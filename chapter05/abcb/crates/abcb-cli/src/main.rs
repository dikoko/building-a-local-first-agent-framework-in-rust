use abcb_core::{Event, MockProvider, one_turn, read_events, write_event};
#[cfg(test)]
use clap::CommandFactory;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

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
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Config {
    project: Option<ProjectConfig>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ProjectConfig {
    name: Option<String>,
}

impl Config {
    fn project_name(&self) -> Option<&str> {
        self.project.as_ref()?.name.as_deref()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => run_doctor()?,
        Command::Chat { message, mock, log } => run_chat(message, mock, log)?,
        Command::Replay { path } => run_replay(path)?,
    }

    Ok(())
}

fn run_chat(message: String, mock: bool, log: Option<PathBuf>) -> Result<(), Box<dyn Error>> {
    if !mock {
        return Err(
            "only --mock is supported right now; pass --mock to use the mock provider".into(),
        );
    }

    let mut provider = MockProvider::new([format!("mock: you said {message}")]);
    let reply = one_turn(&mut provider, &message)?;
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

    for (index, event) in events.iter().enumerate() {
        let (kind, content) = match event {
            Event::UserMessage { content } => ("user_message", content),
            Event::ModelResponse { content } => ("model_response", content),
            Event::FinalAnswer { content } => ("final_answer", content),
        };
        println!("[{}] {kind}: {content}", index + 1);
    }

    Ok(())
}

fn run_doctor() -> Result<(), Box<dyn Error>> {
    println!("abcb doctor");
    println!("workspace: ok");

    match load_config(Path::new("abcb.toml"))? {
        Some(config) => {
            println!("config: found abcb.toml");
            if let Some(name) = config.project_name() {
                println!("project: {name}");
            }
        }
        None => println!("config: abcb.toml not found (ok for now)"),
    }

    Ok(())
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

    #[test]
    fn run_chat_without_mock_flag_errors() {
        let err = run_chat("hi".into(), false, None).expect_err("should require --mock");

        assert!(err.to_string().contains("--mock"));
    }

    #[test]
    fn run_chat_with_log_writes_three_jsonl_events() {
        let log = tempfile::NamedTempFile::new().expect("temp file");

        run_chat("hi".into(), true, Some(log.path().to_path_buf()))
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

    #[test]
    fn run_chat_appends_to_existing_log_file() {
        let mut log = tempfile::NamedTempFile::new().expect("temp file");
        std::io::Write::write_all(log.as_file_mut(), b"{\"kind\":\"prior\"}\n")
            .expect("seed prior content");

        run_chat("hi".into(), true, Some(log.path().to_path_buf()))
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

    #[test]
    fn chat_log_is_readable_by_read_events() {
        let log = tempfile::NamedTempFile::new().expect("temp file");

        run_chat("hi".into(), true, Some(log.path().to_path_buf())).expect("run_chat");

        let file = File::open(log.path()).expect("open log");
        let events = read_events(BufReader::new(file)).expect("read_events should parse log");

        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0],
            Event::UserMessage { content } if content == "hi"
        ));
        assert!(matches!(
            &events[1],
            Event::ModelResponse { content } if content == "mock: you said hi"
        ));
        assert!(matches!(
            &events[2],
            Event::FinalAnswer { content } if content == "mock: you said hi"
        ));
    }
}
