use abcb_core::{
    AllowAll, Event, LoggedEvent, LoopError, Message, MockProvider, Provider, Role, Session,
    ToolRegistry, one_turn, read_events, run_loop, system_prompt, write_event,
};
use abcb_models::{OpenAiCompatProvider, check_health};
use abcb_tools::{AddNumbers, Echo, SessionNoteAppend, SessionNoteSearch};
#[cfg(test)]
use clap::CommandFactory;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Default location for project notes used by the file-backed tools: JSONL,
/// at the workspace root (project-scoped memory that persists across sessions,
/// a sibling of `.abcb/sessions/`, not nested under any one session).
const NOTES_PATH: &str = ".abcb/notes.jsonl";

/// Default root for per-session storage. The real provider path may override
/// this with `[memory] dir` from `abcb.toml` via `Config::memory_dir()`.
const DEFAULT_MEMORY_DIR: &str = ".abcb/sessions";

/// Ceiling on agent-loop iterations for the mock path. The real provider path
/// reads `[agent] max_steps` from `abcb.toml` via `Config::max_steps()`.
const DEFAULT_MAX_STEPS: usize = 5;

/// How many times `abcb eval` runs each fixture by default. A nondeterministic
/// model needs repetition: the pass *rate* is the signal, not a single result.
const DEFAULT_EVAL_RUNS: usize = 5;

#[derive(Debug, Parser)]
#[command(name = "abcb")]
#[command(version)]
#[command(about = "AI agent framework for Godot game development")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Check the local abcb development environment.
    Doctor,
    /// Send a single user message and print the assistant reply.
    Chat {
        /// The user message to send.
        message: String,
        /// Use the in-process mock provider (required while no real provider exists).
        #[arg(long)]
        mock: bool,
        /// Append JSONL event records to this file. When omitted, no events are recorded.
        #[arg(long, value_name = "PATH")]
        log: Option<PathBuf>,
    },
    /// Read a JSONL event log and print the recorded event sequence.
    Replay {
        /// Path to a JSONL event log file produced by `abcb chat --log`.
        path: PathBuf,
    },
    /// Run the agent loop against the tool registry and print the final answer.
    Run {
        /// The user message to send.
        message: String,
        /// Use the in-process mock provider (required while no real provider exists).
        #[arg(long)]
        mock: bool,
    },
    /// Run evaluation fixtures against the configured model and print a scorecard.
    Eval {
        /// How many times to run each fixture (the pass rate over repeated runs).
        #[arg(long, default_value_t = DEFAULT_EVAL_RUNS)]
        runs: usize,
    },
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Config {
    project: Option<ProjectConfig>,
    model: Option<ModelConfig>,
    agent: Option<AgentConfig>,
    memory: Option<MemoryConfig>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ProjectConfig {
    name: Option<String>,
}

/// MLX provider endpoint settings. Present only when `[model]` is configured.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ModelConfig {
    /// OpenAI-compatible base URL, e.g. `http://localhost:8083/v1`.
    base_url: String,
    /// Model identifier sent in the request body (the local model path).
    model: String,
    /// Optional health endpoint used by `abcb doctor`.
    health_url: Option<String>,
}

/// Agent-loop tuning. Present only when `[agent]` is configured.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct AgentConfig {
    /// Ceiling on agent-loop iterations. Falls back to `DEFAULT_MAX_STEPS`.
    max_steps: Option<usize>,
}

/// Session-memory settings. Present only when `[memory]` is configured.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct MemoryConfig {
    /// Directory holding per-session state. Falls back to `.abcb/sessions`.
    dir: Option<PathBuf>,
}

impl Config {
    fn project_name(&self) -> Option<&str> {
        self.project.as_ref()?.name.as_deref()
    }

    /// Configured agent step ceiling, or `DEFAULT_MAX_STEPS` if unset.
    fn max_steps(&self) -> usize {
        self.agent
            .as_ref()
            .and_then(|agent| agent.max_steps)
            .unwrap_or(DEFAULT_MAX_STEPS)
    }

    /// Configured session-storage root, or `.abcb/sessions` if unset.
    fn memory_dir(&self) -> &Path {
        self.memory
            .as_ref()
            .and_then(|memory| memory.dir.as_deref())
            .unwrap_or_else(|| Path::new(DEFAULT_MEMORY_DIR))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => run_doctor().await?,
        Command::Chat { message, mock, log } => run_chat(message, mock, log).await?,
        Command::Replay { path } => run_replay(path)?,
        Command::Run { message, mock } => run_run(message, mock).await?,
        Command::Eval { runs } => run_eval_command(runs).await?,
    }

    Ok(())
}

/// Load `abcb.toml`, erroring if it is absent (the real-provider path needs it).
fn load_required_config() -> Result<Config, Box<dyn Error>> {
    let config = load_config(Path::new("abcb.toml"))?
        .ok_or("no abcb.toml found; add a [model] section or pass --mock")?;
    Ok(config)
}

/// Build the configured OpenAI-compatible provider from a `[model]` section.
fn build_provider(config: &Config) -> Result<OpenAiCompatProvider, Box<dyn Error>> {
    let model = config
        .model
        .as_ref()
        .ok_or("abcb.toml has no [model] section; add one or pass --mock")?;
    Ok(OpenAiCompatProvider::new(
        model.base_url.as_str(),
        model.model.as_str(),
    ))
}

fn default_registry(notes_path: PathBuf) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Echo).expect("echo is unique");
    registry
        .register(AddNumbers)
        .expect("add_numbers is unique");
    registry
        .register(SessionNoteAppend::new(notes_path.clone()))
        .expect("session_note_append is unique");
    registry
        .register(SessionNoteSearch::new(notes_path))
        .expect("session_note_search is unique");
    registry
}

/// The directory a session's artifacts live under: `<root>/<session-id>`.
///
/// Pure path composition, with no I/O, so it's directly unit-testable.
fn session_dir(root: &Path, session_id: &str) -> PathBuf {
    root.join(session_id)
}

/// Create the session's storage directory (and any missing parents), returning
/// its path. Idempotent: `create_dir_all` is a no-op if the directory exists.
fn create_session_dir(root: &Path, session_id: &str) -> io::Result<PathBuf> {
    let dir = session_dir(root, session_id);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// How a run ended: the summary's `status`. Derived from the loop's outcome,
/// not the event log (the log doesn't record *why* a run stopped short).
#[derive(Debug, Eq, PartialEq)]
enum RunStatus {
    Completed,
    MaxStepsExceeded,
    Failed,
}

impl RunStatus {
    fn from_outcome(outcome: &Result<String, LoopError>) -> Self {
        match outcome {
            Ok(_) => RunStatus::Completed,
            Err(LoopError::MaxStepsExceeded { .. }) => RunStatus::MaxStepsExceeded,
            Err(_) => RunStatus::Failed,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            RunStatus::Completed => "completed",
            RunStatus::MaxStepsExceeded => "max steps exceeded",
            RunStatus::Failed => "failed",
        }
    }
}

/// A one-run summary: a read-model *projected* from the run's `events.jsonl`
/// (steps, tools), combined with its outcome (status) and endpoint. Derived, not
/// accumulated by the loop; the event log is the single source of truth.
#[derive(Debug, Eq, PartialEq)]
struct RunSummary {
    status: RunStatus,
    steps: usize,
    tools_called: Vec<String>,
    tools_denied: Vec<String>,
    endpoint: String,
}

impl RunSummary {
    fn from_run(
        events: &[LoggedEvent],
        outcome: &Result<String, LoopError>,
        endpoint: String,
    ) -> Self {
        let mut steps = 0;
        let mut tools_called = Vec::new();
        let mut tools_denied = Vec::new();
        for logged in events {
            match &logged.event {
                // One model turn per ModelResponse, including denied, recovered,
                // and malformed turns (each logs a ModelResponse before parsing).
                Event::ModelResponse { .. } => steps += 1,
                Event::ToolResult { tool_name, .. } => tools_called.push(tool_name.clone()),
                Event::ToolDenied { tool_name } => tools_denied.push(tool_name.clone()),
                Event::UserMessage { .. } | Event::FinalAnswer { .. } => {}
            }
        }
        RunSummary {
            status: RunStatus::from_outcome(outcome),
            steps,
            tools_called,
            tools_denied,
            endpoint,
        }
    }
}

impl fmt::Display for RunSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |xs: &[String]| format!("[{}]", xs.join(", "));
        write!(
            f,
            "run summary: status: {}, steps: {}, tools: {}, denied: {}, endpoint: {}",
            self.status.label(),
            self.steps,
            list(&self.tools_called),
            list(&self.tools_denied),
            self.endpoint,
        )
    }
}

/// The endpoint label for the summary: `base_url (model)` for the real provider.
fn endpoint_label(config: &Config) -> String {
    match &config.model {
        Some(model) => format!("{} ({})", model.base_url, model.model),
        None => "unknown".to_string(),
    }
}

async fn run_run(message: String, mock: bool) -> Result<(), Box<dyn Error>> {
    let registry = default_registry(PathBuf::from(NOTES_PATH));

    // The CLI owns the session: stamp it, store it, drive it, then summarize.
    // Seed a system prompt teaching the envelope contract (the run path parses
    // ModelOutput; chat does not, so chat skips this). It's a Role::System
    // message, so it's not logged as a run event and doesn't count as a step.
    let mut session = Session::start();
    session.push_message(Message::new(Role::System, system_prompt(&registry)));
    session.push_message(Message::new(Role::User, message.as_str()));

    // Each branch runs the loop with its own provider, then returns the *captured*
    // outcome (NOT `?`-propagated; a failed run must still be summarized), the
    // endpoint label, and where its event log lives. Storage is orthogonal to the
    // provider, so both write a session dir + events.jsonl.
    let (outcome, endpoint, events_path) = if mock {
        let dir = create_session_dir(Path::new(DEFAULT_MEMORY_DIR), &session.id)?;
        let events_path = dir.join("events.jsonl");
        let mut events = open_event_log(&events_path)?;
        let mut provider = MockProvider::new([format!(
            r#"{{"kind":"final","content":"mock run: you said {message}"}}"#
        )]);
        let outcome = run_loop(
            &mut provider,
            &registry,
            &mut session,
            DEFAULT_MAX_STEPS,
            &mut events,
            &mut AllowAll,
        )
        .await;
        events.flush()?;
        (outcome, "mock".to_string(), events_path)
    } else {
        let config = load_required_config()?;
        let dir = create_session_dir(config.memory_dir(), &session.id)?;
        let events_path = dir.join("events.jsonl");
        let mut events = open_event_log(&events_path)?;
        let endpoint = endpoint_label(&config);
        let mut provider = build_provider(&config)?;
        let outcome = run_loop(
            &mut provider,
            &registry,
            &mut session,
            config.max_steps(),
            &mut events,
            &mut AllowAll,
        )
        .await;
        events.flush()?;
        (outcome, endpoint, events_path)
    };

    // Summarize from the log we just wrote (a read-model over events.jsonl) and
    // print to stderr, so stdout stays the answer alone. Done for *both* success
    // and failure, then surface the outcome (a failed run still exits non-zero).
    let logged = read_events(BufReader::new(File::open(&events_path)?))?;
    let summary = RunSummary::from_run(&logged, &outcome, endpoint);
    eprintln!("{summary}");

    let answer = outcome?;
    println!("{answer}");

    Ok(())
}

/// Open an event log file for appending, buffered.
///
/// `append(true).create(true)` is the same idiom `chat --log` uses; a fresh
/// session dir means this is effectively create-new, but append keeps it safe if
/// a run ever reuses a directory.
fn open_event_log(path: &Path) -> io::Result<BufWriter<File>> {
    let file = OpenOptions::new().append(true).create(true).open(path)?;
    Ok(BufWriter::new(file))
}

/// What a fixture expects of a run. Judged from the run's outcome + summary
/// (which already exist post-run), so a fixture is pure data.
enum Expectation {
    /// The agent loop reached a final answer (didn't error or exhaust steps).
    Completes,
    /// The run invoked the named tool at least once.
    CallsTool { name: String },
    /// The final answer contains this substring.
    FinalContains { text: String },
}

impl Expectation {
    /// A short label for the scorecard, e.g. `calls session_note_append`.
    fn label(&self) -> String {
        match self {
            Expectation::Completes => "completes".to_string(),
            Expectation::CallsTool { name } => format!("calls {name}"),
            Expectation::FinalContains { text } => format!("final contains {text:?}"),
        }
    }
}

/// One evaluation case: a prompt and what a good run of it looks like.
struct EvalCase {
    name: String,
    prompt: String,
    expectation: Expectation,
}

/// The built-in reliability probes. Curated in code (not a data file): these are
/// developer-maintained, type-checked against `Expectation`, and version
/// controlled. The runner takes `&[EvalCase]`, so a file loader can feed it later
/// with no change here. `append-note` is the probe that answers the earlier
/// question: how reliably does the model emit a correct tool call?
fn default_fixtures() -> Vec<EvalCase> {
    vec![
        EvalCase {
            name: "say-hi".to_string(),
            prompt: "Say hi in one short sentence.".to_string(),
            expectation: Expectation::Completes,
        },
        EvalCase {
            name: "append-note".to_string(),
            prompt: "Use a tool to remember this note: buy milk.".to_string(),
            expectation: Expectation::CallsTool {
                name: "session_note_append".to_string(),
            },
        },
        EvalCase {
            name: "add-numbers".to_string(),
            prompt: "What is 3 plus 4? Use a tool, then give the number.".to_string(),
            expectation: Expectation::FinalContains {
                text: "7".to_string(),
            },
        },
    ]
}

/// Judge one run against an expectation. Pure: reads only the outcome and the
/// summary, so it's unit-testable without a model.
fn judge(
    expectation: &Expectation,
    outcome: &Result<String, LoopError>,
    summary: &RunSummary,
) -> bool {
    match expectation {
        Expectation::Completes => outcome.is_ok(),
        Expectation::CallsTool { name } => summary.tools_called.iter().any(|t| t == name),
        Expectation::FinalContains { text } => {
            outcome.as_ref().map(|a| a.contains(text)).unwrap_or(false)
        }
    }
}

/// One eval trial: drive the full agent loop (system prompt + user prompt) once,
/// in memory, and return the outcome plus a summary derived from its events.
async fn eval_trial(
    provider: &mut impl Provider,
    registry: &ToolRegistry,
    prompt: &str,
) -> (Result<String, LoopError>, RunSummary) {
    let mut session = Session::start();
    session.push_message(Message::new(Role::System, system_prompt(registry)));
    session.push_message(Message::new(Role::User, prompt));

    // Events go to an in-memory buffer: eval trials are throwaway, so they don't
    // create session dirs or touch disk like `run` does.
    let mut events: Vec<u8> = Vec::new();
    let outcome = run_loop(
        provider,
        registry,
        &mut session,
        DEFAULT_MAX_STEPS,
        &mut events,
        &mut AllowAll,
    )
    .await;
    let logged = read_events(events.as_slice()).unwrap_or_default();
    let summary = RunSummary::from_run(&logged, &outcome, "eval".to_string());
    (outcome, summary)
}

/// The result of running one fixture `runs` times.
struct FixtureResult {
    name: String,
    label: String,
    passed: usize,
    runs: usize,
}

/// Run each fixture `runs` times and tally how often it met its expectation.
///
/// Generic over the provider so it's unit-tested with `MockProvider` (scripted
/// responses -> deterministic tallies) and run live with `OpenAiCompatProvider`.
async fn run_eval(
    provider: &mut impl Provider,
    registry: &ToolRegistry,
    fixtures: &[EvalCase],
    runs: usize,
) -> Vec<FixtureResult> {
    let mut results = Vec::new();
    for case in fixtures {
        let mut passed = 0;
        for _ in 0..runs {
            let (outcome, summary) = eval_trial(provider, registry, &case.prompt).await;
            if judge(&case.expectation, &outcome, &summary) {
                passed += 1;
            }
        }
        results.push(FixtureResult {
            name: case.name.clone(),
            label: case.expectation.label(),
            passed,
            runs,
        });
    }
    results
}

async fn run_eval_command(runs: usize) -> Result<(), Box<dyn Error>> {
    let config = load_required_config()?;
    let mut provider = build_provider(&config)?;
    // Isolate eval side effects from the project's real notes: point the
    // file-backed tools at a throwaway path under the system temp dir.
    let notes = std::env::temp_dir().join("abcb-eval-notes.jsonl");
    let registry = default_registry(notes);
    let fixtures = default_fixtures();

    let results = run_eval(&mut provider, &registry, &fixtures, runs).await;

    println!("abcb eval: {runs} runs each");
    let mut total_passed = 0;
    let mut total = 0;
    for result in &results {
        println!(
            "  {:<14} {}/{}  {}",
            result.name, result.passed, result.runs, result.label
        );
        total_passed += result.passed;
        total += result.runs;
    }
    println!("overall: {total_passed}/{total}");

    Ok(())
}

async fn run_chat(message: String, mock: bool, log: Option<PathBuf>) -> Result<(), Box<dyn Error>> {
    let reply = if mock {
        let mut provider = MockProvider::new([format!("mock: you said {message}")]);
        one_turn(&mut provider, &message).await?
    } else {
        let config = load_required_config()?;
        let mut provider = build_provider(&config)?;
        one_turn(&mut provider, &message).await?
    };
    println!("{}", reply.content);

    if let Some(path) = log {
        let file = OpenOptions::new().append(true).create(true).open(&path)?;
        let mut writer = BufWriter::new(file);

        write_event(
            &mut writer,
            &Event::UserMessage {
                content: message.clone(),
            },
        )?;
        write_event(
            &mut writer,
            &Event::ModelResponse {
                content: reply.content.clone(),
            },
        )?;
        write_event(
            &mut writer,
            &Event::FinalAnswer {
                content: reply.content,
            },
        )?;
    }

    Ok(())
}

fn run_replay(path: PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let events = read_events(reader)?;

    for (index, logged) in events.iter().enumerate() {
        let (kind, content) = match &logged.event {
            Event::UserMessage { content } => ("user_message", content.clone()),
            Event::ModelResponse { content } => ("model_response", content.clone()),
            Event::ToolResult { tool_name, output } => {
                ("tool_result", format!("{tool_name}: {output}"))
            }
            Event::ToolDenied { tool_name } => ("tool_denied", tool_name.clone()),
            Event::FinalAnswer { content } => ("final_answer", content.clone()),
        };
        println!("[{}] {kind}: {content}", index + 1);
    }

    Ok(())
}

async fn run_doctor() -> Result<(), Box<dyn Error>> {
    println!("abcb doctor");
    println!("workspace: ok");

    let config = match load_config(Path::new("abcb.toml"))? {
        Some(config) => {
            println!("config: found abcb.toml");
            if let Some(name) = config.project_name() {
                println!("project: {name}");
            }
            config
        }
        None => {
            println!("config: abcb.toml not found (ok for now)");
            return Ok(());
        }
    };

    match &config.model {
        Some(model) => {
            println!("model: {} @ {}", model.model, model.base_url);
            check_mlx_health(model).await;
        }
        None => println!("model: no [model] section configured"),
    }

    Ok(())
}

/// Probe the configured MLX `/health` endpoint and print what it reports.
///
/// Diagnostic, not fatal: an unreachable server prints a status line and
/// returns; `doctor`'s job is to report problems, not abort on them.
async fn check_mlx_health(model: &ModelConfig) {
    let Some(health_url) = model.health_url.as_deref() else {
        println!("mlx: no health_url configured (skipping health check)");
        return;
    };

    match check_health(health_url).await {
        Ok(report) => {
            println!("mlx: {} ({health_url})", report.status);
            match report.loaded_model {
                Some(loaded) => println!("mlx model: {loaded}"),
                None => println!("mlx model: (none reported)"),
            }
        }
        Err(e) => println!("mlx: unreachable ({e})"),
    }
}

fn load_config(path: &Path) -> Result<Option<Config>, Box<dyn Error>> {
    if !path.exists() {
        return Ok(None);
    }

    let source = fs::read_to_string(path)?;
    let config = parse_config(&source)?;

    Ok(Some(config))
}

fn parse_config(source: &str) -> Result<Config, toml::de::Error> {
    toml::from_str(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_name_is_abcb() {
        let command = Cli::command();

        assert_eq!(command.get_name(), "abcb");
    }

    #[test]
    fn parses_project_name_from_config() {
        let config = parse_config(
            r#"
            [project]
            name = "abcb"
            "#,
        )
        .expect("config should parse");

        assert_eq!(config.project_name(), Some("abcb"));
    }

    #[test]
    fn parses_empty_config() {
        let config = parse_config("").expect("empty config should parse");

        assert_eq!(config.project_name(), None);
    }

    #[test]
    fn parses_model_config() {
        let config = parse_config(
            r#"
            [model]
            base_url = "http://localhost:8083/v1"
            health_url = "http://localhost:8083/health"
            model = "/path/to/model"
            "#,
        )
        .expect("model config should parse");

        let model = config.model.expect("model section present");
        assert_eq!(model.base_url, "http://localhost:8083/v1");
        assert_eq!(model.model, "/path/to/model");
        assert_eq!(
            model.health_url.as_deref(),
            Some("http://localhost:8083/health")
        );
    }

    #[test]
    fn parses_model_config_without_health_url() {
        let config = parse_config(
            r#"
            [model]
            base_url = "http://localhost:8083/v1"
            model = "/path/to/model"
            "#,
        )
        .expect("model config without health_url should parse");

        let model = config.model.expect("model section present");
        assert_eq!(model.health_url, None);
    }

    #[test]
    fn model_config_requires_base_url_and_model() {
        let err = parse_config(
            r#"
            [model]
            base_url = "http://localhost:8083/v1"
            "#,
        )
        .expect_err("missing model field should error");

        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn parses_agent_max_steps() {
        let config = parse_config(
            r#"
            [agent]
            max_steps = 9
            "#,
        )
        .expect("agent config should parse");

        assert_eq!(config.max_steps(), 9);
    }

    #[test]
    fn max_steps_defaults_to_five_when_absent() {
        let config = parse_config("").expect("empty config should parse");

        assert_eq!(config.max_steps(), DEFAULT_MAX_STEPS);
    }

    #[test]
    fn parses_memory_dir() {
        let config = parse_config(
            r#"
            [memory]
            dir = "/tmp/sessions"
            "#,
        )
        .expect("memory config should parse");

        assert_eq!(config.memory_dir(), Path::new("/tmp/sessions"));
    }

    #[test]
    fn memory_dir_defaults_when_absent() {
        let config = parse_config("").expect("empty config should parse");

        assert_eq!(config.memory_dir(), Path::new(".abcb/sessions"));
    }

    #[test]
    fn chat_subcommand_parses_with_mock_flag() {
        let cli = Cli::try_parse_from(["abcb", "chat", "hi", "--mock"])
            .expect("chat --mock should parse");

        match cli.command {
            Command::Chat { message, mock, log } => {
                assert_eq!(message, "hi");
                assert!(mock);
                assert!(log.is_none());
            }
            other => panic!("expected Chat, got {other:?}"),
        }
    }

    #[test]
    fn chat_subcommand_parses_log_path() {
        let cli = Cli::try_parse_from(["abcb", "chat", "hi", "--mock", "--log", "/tmp/x.jsonl"])
            .expect("chat --mock --log should parse");

        match cli.command {
            Command::Chat { log, .. } => {
                assert_eq!(log.as_deref(), Some(Path::new("/tmp/x.jsonl")));
            }
            other => panic!("expected Chat, got {other:?}"),
        }
    }

    #[test]
    fn replay_subcommand_parses_path() {
        let cli =
            Cli::try_parse_from(["abcb", "replay", "/tmp/x.jsonl"]).expect("replay should parse");

        match cli.command {
            Command::Replay { path } => {
                assert_eq!(path, Path::new("/tmp/x.jsonl"));
            }
            other => panic!("expected Replay, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_chat_without_mock_and_without_config_errors() {
        // No abcb.toml in the crate's test cwd, so the real path has no config.
        let err = run_chat("hi".into(), false, None)
            .await
            .expect_err("should require config or --mock");

        assert!(err.to_string().contains("--mock"));
    }

    #[test]
    fn run_subcommand_parses_with_mock_flag() {
        let cli =
            Cli::try_parse_from(["abcb", "run", "hi", "--mock"]).expect("run --mock should parse");

        match cli.command {
            Command::Run { message, mock } => {
                assert_eq!(message, "hi");
                assert!(mock);
            }
            other => panic!("expected Run, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_run_without_mock_and_without_config_errors() {
        let err = run_run("hi".into(), false)
            .await
            .expect_err("should require config or --mock");

        assert!(err.to_string().contains("--mock"));
    }

    #[test]
    fn build_provider_errors_without_model_section() {
        let config = parse_config("").expect("empty config parses");

        let err = build_provider(&config).expect_err("missing [model] should error");

        assert!(err.to_string().contains("[model]"));
    }

    #[test]
    fn default_registry_has_the_four_starter_tools() {
        let registry = default_registry(PathBuf::from(".abcb/notes.txt"));

        let mut names: Vec<&str> = registry.names().collect();
        names.sort();

        assert_eq!(
            names,
            vec![
                "add_numbers",
                "echo",
                "session_note_append",
                "session_note_search"
            ]
        );
    }

    #[tokio::test]
    async fn run_loop_with_real_tools_appends_a_note_then_finishes() {
        let notes = tempfile::NamedTempFile::new().expect("temp file");
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"session_note_append","arguments":{"note":"buy milk"}}"#.to_string(),
            r#"{"kind":"final","content":"noted"}"#.to_string(),
        ]);
        let registry = default_registry(notes.path().to_path_buf());

        let mut session = Session::start();
        session.push_message(Message::new(Role::User, "remember to buy milk"));
        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session,
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should finish");

        assert_eq!(answer, "noted");
        // The real session_note_append tool ran through the loop and wrote a
        // timestamped JSONL note to disk.
        let contents = std::fs::read_to_string(notes.path()).expect("read notes");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains(r#""text":"buy milk""#));
        assert!(lines[0].contains(r#""at":"#));
    }

    #[test]
    fn run_summary_folds_events_and_reports_outcome() {
        let events = vec![
            LoggedEvent {
                at: 1,
                event: Event::UserMessage {
                    content: "hi".into(),
                },
            },
            LoggedEvent {
                at: 2,
                event: Event::ModelResponse {
                    content: "{...}".into(),
                },
            },
            LoggedEvent {
                at: 3,
                event: Event::ToolResult {
                    tool_name: "session_note_append".into(),
                    output: "ok".into(),
                },
            },
            LoggedEvent {
                at: 4,
                event: Event::ModelResponse {
                    content: "{...}".into(),
                },
            },
            LoggedEvent {
                at: 5,
                event: Event::ToolDenied {
                    tool_name: "danger".into(),
                },
            },
            LoggedEvent {
                at: 6,
                event: Event::ModelResponse {
                    content: "{...}".into(),
                },
            },
            LoggedEvent {
                at: 7,
                event: Event::FinalAnswer {
                    content: "done".into(),
                },
            },
        ];

        let summary = RunSummary::from_run(&events, &Ok("done".into()), "mock".into());

        assert_eq!(summary.status, RunStatus::Completed);
        assert_eq!(summary.steps, 3); // three ModelResponse events
        assert_eq!(
            summary.tools_called,
            vec!["session_note_append".to_string()]
        );
        assert_eq!(summary.tools_denied, vec!["danger".to_string()]);
        assert_eq!(summary.endpoint, "mock");
    }

    #[test]
    fn run_status_reflects_the_loop_outcome() {
        assert_eq!(
            RunStatus::from_outcome(&Ok("x".into())),
            RunStatus::Completed
        );
        assert_eq!(
            RunStatus::from_outcome(&Err(LoopError::MaxStepsExceeded { max_steps: 5 })),
            RunStatus::MaxStepsExceeded
        );
        assert_eq!(
            RunStatus::from_outcome(&Err(LoopError::UnknownTool("x".into()))),
            RunStatus::Failed
        );
    }

    #[test]
    fn run_summary_display_lists_each_field() {
        let summary = RunSummary {
            status: RunStatus::Completed,
            steps: 2,
            tools_called: vec!["echo".into()],
            tools_denied: vec![],
            endpoint: "mock".into(),
        };

        let line = format!("{summary}");
        assert!(line.contains("status: completed"));
        assert!(line.contains("steps: 2"));
        assert!(line.contains("tools: [echo]"));
        assert!(line.contains("denied: []"));
        assert!(line.contains("endpoint: mock"));
    }

    fn summary_with(tools_called: Vec<String>) -> RunSummary {
        RunSummary {
            status: RunStatus::Completed,
            steps: 1,
            tools_called,
            tools_denied: vec![],
            endpoint: "test".into(),
        }
    }

    #[test]
    fn judge_completes_reads_the_outcome() {
        let s = summary_with(vec![]);
        assert!(judge(&Expectation::Completes, &Ok("hi".into()), &s));
        assert!(!judge(
            &Expectation::Completes,
            &Err(LoopError::MaxStepsExceeded { max_steps: 5 }),
            &s
        ));
    }

    #[test]
    fn judge_calls_tool_reads_the_summary() {
        let expect = Expectation::CallsTool {
            name: "session_note_append".into(),
        };
        assert!(judge(
            &expect,
            &Ok("ok".into()),
            &summary_with(vec!["session_note_append".into()])
        ));
        assert!(!judge(&expect, &Ok("ok".into()), &summary_with(vec![])));
    }

    #[test]
    fn judge_final_contains_reads_the_answer() {
        let expect = Expectation::FinalContains { text: "7".into() };
        let s = summary_with(vec![]);
        assert!(judge(&expect, &Ok("the answer is 7".into()), &s));
        assert!(!judge(&expect, &Ok("the answer is six".into()), &s));
        // A run that didn't complete can't contain the text.
        assert!(!judge(
            &expect,
            &Err(LoopError::MaxStepsExceeded { max_steps: 5 }),
            &s
        ));
    }

    #[tokio::test]
    async fn run_eval_tallies_the_pass_rate() {
        let fixtures = vec![EvalCase {
            name: "f".into(),
            prompt: "p".into(),
            expectation: Expectation::Completes,
        }];
        // Two trials, two scripted final envelopes -> both complete.
        let mut provider = MockProvider::new([
            r#"{"kind":"final","content":"a"}"#,
            r#"{"kind":"final","content":"b"}"#,
        ]);
        let registry = ToolRegistry::new();

        let results = run_eval(&mut provider, &registry, &fixtures, 2).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].passed, 2);
        assert_eq!(results[0].runs, 2);
    }

    #[tokio::test]
    async fn run_eval_counts_failures_in_the_rate() {
        let fixtures = vec![EvalCase {
            name: "f".into(),
            prompt: "p".into(),
            expectation: Expectation::Completes,
        }];
        // Only one scripted response: trial 1 completes; trial 2 exhausts the
        // provider and errors -> 1 of 2.
        let mut provider = MockProvider::new([r#"{"kind":"final","content":"a"}"#]);
        let registry = ToolRegistry::new();

        let results = run_eval(&mut provider, &registry, &fixtures, 2).await;

        assert_eq!(results[0].passed, 1);
        assert_eq!(results[0].runs, 2);
    }

    #[test]
    fn session_dir_joins_root_and_id() {
        let dir = session_dir(Path::new(".abcb/sessions"), "sess-1748613022123");
        assert_eq!(dir, PathBuf::from(".abcb/sessions/sess-1748613022123"));
    }

    #[test]
    fn create_session_dir_makes_the_directory_under_the_root() {
        let root = tempfile::tempdir().expect("temp dir");
        let dir = create_session_dir(root.path(), "sess-42").expect("should create");

        assert_eq!(dir, root.path().join("sess-42"));
        assert!(dir.is_dir());
    }

    #[test]
    fn create_session_dir_is_idempotent() {
        let root = tempfile::tempdir().expect("temp dir");
        create_session_dir(root.path(), "sess-42").expect("first create");
        // Calling again on an existing directory must not error.
        create_session_dir(root.path(), "sess-42").expect("second create is a no-op");
    }

    #[tokio::test]
    async fn run_chat_with_log_writes_three_jsonl_events() {
        let log = tempfile::NamedTempFile::new().expect("temp file");

        run_chat("hi".into(), true, Some(log.path().to_path_buf()))
            .await
            .expect("run_chat with --log should succeed");

        let contents = std::fs::read_to_string(log.path()).expect("read log file");
        let lines: Vec<&str> = contents.lines().collect();

        assert_eq!(lines.len(), 3, "expected 3 events, got: {contents:?}");
        assert!(lines[0].contains(r#""kind":"user_message""#));
        assert!(lines[0].contains(r#""content":"hi""#));
        assert!(lines[1].contains(r#""kind":"model_response""#));
        assert!(lines[1].contains("mock: you said hi"));
        assert!(lines[2].contains(r#""kind":"final_answer""#));
        assert!(lines[2].contains("mock: you said hi"));
    }

    #[tokio::test]
    async fn run_chat_appends_to_existing_log_file() {
        let mut log = tempfile::NamedTempFile::new().expect("temp file");
        std::io::Write::write_all(log.as_file_mut(), b"{\"kind\":\"prior\"}\n")
            .expect("seed prior content");

        run_chat("hi".into(), true, Some(log.path().to_path_buf()))
            .await
            .expect("run_chat with --log should succeed");

        let contents = std::fs::read_to_string(log.path()).expect("read log file");
        let lines: Vec<&str> = contents.lines().collect();

        assert_eq!(
            lines.len(),
            4,
            "expected prior + 3 events, got: {contents:?}"
        );
        assert!(lines[0].contains("prior"));
    }

    #[tokio::test]
    async fn chat_log_is_readable_by_read_events() {
        let log = tempfile::NamedTempFile::new().expect("temp file");

        run_chat("hi".into(), true, Some(log.path().to_path_buf()))
            .await
            .expect("run_chat");

        let file = File::open(log.path()).expect("open log");
        let events = read_events(BufReader::new(file)).expect("read_events should parse log");

        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0].event,
            Event::UserMessage { content } if content == "hi"
        ));
        assert!(matches!(
            &events[1].event,
            Event::ModelResponse { content } if content == "mock: you said hi"
        ));
        assert!(matches!(
            &events[2].event,
            Event::FinalAnswer { content } if content == "mock: you said hi"
        ));
    }
}
