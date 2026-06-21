use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Session {
    pub id: String,
    /// When the session was created, as Unix-epoch milliseconds.
    pub created_at: u64,
    pub messages: Vec<Message>,
}

impl Session {
    /// Construct a session with an explicit id and creation time.
    ///
    /// Pure and deterministic: the timestamp arrives as *data*, not by reading
    /// the clock, so a test can build a session that `assert_eq!`s against a
    /// fixture. Production code that wants "now" calls [`Session::start`].
    pub fn new(id: impl Into<String>, created_at: u64) -> Self {
        Self {
            id: id.into(),
            created_at,
            messages: Vec::new(),
        }
    }

    /// Start a fresh session, stamped with the current time.
    ///
    /// The single impure entry point in the session path: [`now_millis`] is the
    /// one clock read, [`new_session_id`] derives the id from it, then both are
    /// handed to the pure [`Session::new`]. Keeping the impurity here (rather
    /// than inside `new`) is what lets every other construction stay testable.
    pub fn start() -> Self {
        let now = now_millis();
        Session::new(new_session_id(now), now)
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}

/// The current time as Unix-epoch milliseconds.
///
/// The project's single clock-read helper, used wherever a record is stamped
/// (session creation, the event log via [`write_event`], and `abcb-tools` notes).
/// `duration_since` returns a `Result` because the system clock *can* be set
/// before the epoch; that case is not something a caller can act on, so we treat
/// it as `0` rather than propagate.
pub fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// Derive a session id from a creation timestamp.
///
/// Pure (same input → same id) so it's unit-testable. Unique at single-user CLI
/// scale: one session per process, so a collision would need two launches in the
/// same millisecond. Promote to a random id (uuid) if concurrent sessions ever
/// share a process, for example a future daemon.
fn new_session_id(now_millis: u64) -> String {
    format!("sess-{now_millis}")
}

#[derive(Debug)]
pub enum ProviderError {
    /// A scripted provider ran out of responses (test/mock concern).
    NoMoreResponses,
    /// A concrete provider's backend failed (HTTP, JSON, a non-2xx status, …).
    ///
    /// The trait lives in `abcb-core`, but real backends live in other crates
    /// (e.g. `abcb-models`). Rather than pull `reqwest`/`serde_json` errors into
    /// the vocabulary here, each backend maps its own failure into this boxed
    /// error at the trait boundary, keeping core dependency-free and the
    /// crate graph acyclic.
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::NoMoreResponses => write!(f, "no more scripted responses"),
            ProviderError::Backend(e) => write!(f, "provider backend error: {e}"),
        }
    }
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProviderError::NoMoreResponses => None,
            ProviderError::Backend(e) => Some(&**e),
        }
    }
}

// Native `async fn` in a public trait warns (`async_fn_in_trait`) that callers
// can't add a `Send` bound to the returned future. We drive providers
// sequentially on a single-threaded runtime, so that bound isn't needed.
#[allow(async_fn_in_trait)]
pub trait Provider {
    async fn complete(&mut self, session: &Session) -> Result<Message, ProviderError>;
}

pub struct MockProvider {
    scripted: VecDeque<String>,
}

impl MockProvider {
    pub fn new<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            scripted: responses.into_iter().map(Into::into).collect(),
        }
    }
}

impl Provider for MockProvider {
    async fn complete(&mut self, _session: &Session) -> Result<Message, ProviderError> {
        let content = self
            .scripted
            .pop_front()
            .ok_or(ProviderError::NoMoreResponses)?;
        Ok(Message::new(Role::Assistant, content))
    }
}

pub async fn one_turn(
    provider: &mut impl Provider,
    user_message: impl Into<String>,
) -> Result<Message, ProviderError> {
    let mut session = Session::start();
    session.push_message(Message::new(Role::User, user_message));
    provider.complete(&session).await
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    UserMessage { content: String },
    ModelResponse { content: String },
    ToolResult { tool_name: String, output: String },
    ToolDenied { tool_name: String },
    FinalAnswer { content: String },
}

#[derive(Debug)]
pub enum EventLogError {
    Io(io::Error),
    Serde(serde_json::Error),
}

impl fmt::Display for EventLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventLogError::Io(e) => write!(f, "event log io error: {e}"),
            EventLogError::Serde(e) => write!(f, "event log serialization error: {e}"),
        }
    }
}

impl std::error::Error for EventLogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EventLogError::Io(e) => Some(e),
            EventLogError::Serde(e) => Some(e),
        }
    }
}

impl From<io::Error> for EventLogError {
    fn from(e: io::Error) -> Self {
        EventLogError::Io(e)
    }
}

impl From<serde_json::Error> for EventLogError {
    fn from(e: serde_json::Error) -> Self {
        EventLogError::Serde(e)
    }
}

/// A timestamped event as written to the JSONL log.
///
/// [`Event`] itself stays time-free, pure and deterministic to construct, so
/// the loop can build events without reading the clock (which would make the
/// event stream non-deterministic and untestable). The timestamp is stamped at
/// the I/O boundary by [`write_event`], localizing the lone clock read to the
/// write edge: the same impurity-at-the-edge split as [`Session::start`].
///
/// `at` is `#[serde(default)]` so a log written before timestamps existed (or by
/// another tool) still reads back, with `at` defaulting to `0`. The `#[serde(flatten)]`
/// merges the event's own `kind`-tagged fields up to the top level, so a record
/// reads `{"at":1780…,"kind":"model_response","content":"…"}`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LoggedEvent {
    #[serde(default)]
    pub at: u64,
    #[serde(flatten)]
    pub event: Event,
}

/// Append one event to the log as a timestamped JSONL line.
///
/// Stamps the current time here. This is the one impure clock read in the event
/// path. The `&Event` is cloned into a [`LoggedEvent`] for serialization; events
/// are small, and keeping the borrow at the call site simpler is worth the copy.
pub fn write_event(writer: &mut impl Write, event: &Event) -> Result<(), EventLogError> {
    let record = LoggedEvent {
        at: now_millis(),
        event: event.clone(),
    };
    serde_json::to_writer(&mut *writer, &record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub fn read_events(reader: impl BufRead) -> Result<Vec<LoggedEvent>, EventLogError> {
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event: LoggedEvent = serde_json::from_str(trimmed)?;
        events.push(event);
    }
    Ok(events)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelOutput {
    ToolCall {
        tool_name: String,
        arguments: serde_json::Value,
        #[serde(default)]
        note: Option<String>,
    },
    Final {
        content: String,
    },
}

#[derive(Debug)]
pub enum ModelOutputError {
    Parse(serde_json::Error),
}

impl fmt::Display for ModelOutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelOutputError::Parse(e) => write!(f, "failed to parse model output: {e}"),
        }
    }
}

impl std::error::Error for ModelOutputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelOutputError::Parse(e) => Some(e),
        }
    }
}

impl From<serde_json::Error> for ModelOutputError {
    fn from(e: serde_json::Error) -> Self {
        ModelOutputError::Parse(e)
    }
}

impl ModelOutput {
    pub fn parse(raw: &str) -> Result<ModelOutput, ModelOutputError> {
        Ok(serde_json::from_str(extract_json(raw))?)
    }
}

/// Pull the JSON out of a model's raw reply, tolerating the markdown code fences
/// local models habitually wrap it in (often with prose around the fence).
///
/// If a ```` ``` ```` fence is present, return the contents of the first fenced
/// block; otherwise return the trimmed input. This is *fence-aware* (it keys off
/// the fence markers the model emits), not brace-matching, so it won't be fooled
/// by a `}` inside a string value or trailing prose. Prose with no JSON still
/// fails to parse, which the recovery loop and system prompt handle.
fn extract_json(raw: &str) -> &str {
    let Some(open) = raw.find("```") else {
        return raw.trim();
    };

    // Skip the opening fence line, including any language tag (```` ```json ````):
    // everything up to and including the newline that ends the fence line.
    let after_open = &raw[open + 3..];
    let body = match after_open.find('\n') {
        Some(newline) => &after_open[newline + 1..],
        None => after_open,
    };

    // Cut at the closing fence if there is one; otherwise take the rest.
    let inner = match body.find("```") {
        Some(close) => &body[..close],
        None => body,
    };

    inner.trim()
}

#[derive(Debug)]
pub enum ToolError {
    InvalidArguments(serde_json::Error),
    Execution(String),
    Io(io::Error),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::InvalidArguments(e) => write!(f, "invalid tool arguments: {e}"),
            ToolError::Execution(msg) => write!(f, "tool execution failed: {msg}"),
            ToolError::Io(e) => write!(f, "tool io error: {e}"),
        }
    }
}

impl std::error::Error for ToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ToolError::InvalidArguments(e) => Some(e),
            ToolError::Execution(_) => None,
            ToolError::Io(e) => Some(e),
        }
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(e: serde_json::Error) -> Self {
        ToolError::InvalidArguments(e)
    }
}

impl From<io::Error> for ToolError {
    fn from(e: io::Error) -> Self {
        ToolError::Io(e)
    }
}

pub trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError>;
}

#[derive(Debug, Eq, PartialEq)]
pub enum RegistryError {
    DuplicateName(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::DuplicateName(name) => {
                write!(f, "tool already registered: {name}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) -> Result<(), RegistryError> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateName(name));
        }
        self.tools.insert(name, Box::new(tool));
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|boxed| &**boxed)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the system prompt that teaches a local model the agent's output
/// contract: the JSON envelope schema, a worked example of each kind, and the
/// registered tools (sorted, so the prompt is stable and testable).
///
/// Lives in core because core owns the contract ([`ModelOutput`]) and the
/// [`ToolRegistry`] type, keeping the schema description next to the type it
/// describes. The caller injects the returned string as a `Role::System` message
/// (see the CLI's `run`). This is also the seam where retrieved memory would
/// later be injected. That seam is not filled yet.
pub fn system_prompt(registry: &ToolRegistry) -> String {
    let mut tools: Vec<(&str, &str)> = registry
        .names()
        .map(|name| {
            let description = registry
                .get(name)
                .map(|tool| tool.description())
                .unwrap_or("");
            (name, description)
        })
        .collect();
    tools.sort_by_key(|(name, _)| *name);

    let mut prompt = String::from(
        "You are abcb, an agent that acts by emitting JSON.\n\n\
         Respond with EXACTLY ONE JSON object and nothing else: no prose, no markdown code fences.\n\n\
         The object must be one of two shapes:\n\n\
         To call a tool:\n  \
         {\"kind\":\"tool_call\",\"tool_name\":\"<one of the tools below>\",\"arguments\":{...}}\n\n\
         To give your final answer:\n  \
         {\"kind\":\"final\",\"content\":\"<your answer>\"}\n\n\
         Available tools:\n",
    );
    for (name, description) in &tools {
        prompt.push_str("- ");
        prompt.push_str(name);
        prompt.push_str(": ");
        prompt.push_str(description);
        prompt.push('\n');
    }
    prompt.push_str(
        "\nExamples:\n  \
         {\"kind\":\"tool_call\",\"tool_name\":\"echo\",\"arguments\":{\"text\":\"hello\"}}\n  \
         {\"kind\":\"final\",\"content\":\"All done.\"}",
    );
    prompt
}

#[derive(Clone, Debug, PartialEq)]
pub enum StepOutcome {
    Final(String),
    ToolExecuted { tool_name: String, output: String },
    ToolDenied { tool_name: String },
}

#[derive(Debug)]
pub enum LoopError {
    Provider(ProviderError),
    Parse(ModelOutputError),
    UnknownTool(String),
    Tool(ToolError),
    EventLog(EventLogError),
    MaxStepsExceeded { max_steps: usize },
}

impl fmt::Display for LoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoopError::Provider(e) => write!(f, "provider error: {e}"),
            LoopError::Parse(e) => write!(f, "{e}"),
            LoopError::UnknownTool(name) => write!(f, "unknown tool: {name}"),
            LoopError::Tool(e) => write!(f, "{e}"),
            LoopError::EventLog(e) => write!(f, "event log error: {e}"),
            LoopError::MaxStepsExceeded { max_steps } => {
                write!(f, "exceeded max steps ({max_steps}) without a final answer")
            }
        }
    }
}

impl std::error::Error for LoopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoopError::Provider(e) => Some(e),
            LoopError::Parse(e) => Some(e),
            LoopError::UnknownTool(_) => None,
            LoopError::Tool(e) => Some(e),
            LoopError::EventLog(e) => Some(e),
            LoopError::MaxStepsExceeded { .. } => None,
        }
    }
}

impl LoopError {
    /// Classify this error for the agent loop's recovery policy.
    ///
    /// Returns `Some(feedback)` when the model can plausibly fix the problem by
    /// trying again. The feedback is appended to the session as a `Role::Tool`
    /// message so the next turn sees what went wrong. Returns `None` for
    /// environment, provider, or terminal errors that retrying cannot fix.
    pub fn recovery_feedback(&self) -> Option<String> {
        match self {
            LoopError::Parse(e) => Some(format!(
                "your previous output was not valid JSON ({e}); reply with a single valid envelope"
            )),
            LoopError::UnknownTool(name) => Some(format!(
                "unknown tool `{name}`; call one of the registered tools instead"
            )),
            LoopError::Tool(ToolError::InvalidArguments(e)) => Some(format!(
                "invalid tool arguments: {e}; correct them and try again"
            )),
            LoopError::Tool(ToolError::Execution(msg)) => Some(format!(
                "tool execution failed: {msg}; try a different approach"
            )),
            // Environment / provider / terminal errors are not the model's to fix.
            LoopError::Tool(ToolError::Io(_)) => None,
            LoopError::Provider(_) => None,
            LoopError::EventLog(_) => None,
            LoopError::MaxStepsExceeded { .. } => None,
        }
    }
}

impl From<ProviderError> for LoopError {
    fn from(e: ProviderError) -> Self {
        LoopError::Provider(e)
    }
}

impl From<ModelOutputError> for LoopError {
    fn from(e: ModelOutputError) -> Self {
        LoopError::Parse(e)
    }
}

impl From<ToolError> for LoopError {
    fn from(e: ToolError) -> Self {
        LoopError::Tool(e)
    }
}

impl From<EventLogError> for LoopError {
    fn from(e: EventLogError) -> Self {
        LoopError::EventLog(e)
    }
}

/// The decision an [`ApprovalPolicy`] returns for a proposed tool call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalDecision {
    Allow,
    Deny,
}

/// Decides whether a proposed tool call may execute.
///
/// Injected into the agent loop like the event sink: the loop asks before *every*
/// tool call, and the policy decides. Keeping it a trait (not an enum of modes)
/// lets the decision source vary: `AllowAll` in production today, a scripted
/// policy in tests, and eventually an interactive prompt or allow-list
/// without core taking on any UX or I/O assumptions. `&mut self` lets a policy
/// carry state (e.g. "allow the rest of this session" after one approval).
pub trait ApprovalPolicy {
    fn approve(&mut self, tool_name: &str, arguments: &serde_json::Value) -> ApprovalDecision;
}

/// Approves every tool call: the production default while all tools are safe.
pub struct AllowAll;

impl ApprovalPolicy for AllowAll {
    fn approve(&mut self, _tool_name: &str, _arguments: &serde_json::Value) -> ApprovalDecision {
        ApprovalDecision::Allow
    }
}

/// Denies every tool call: a "block everything" safe mode, and the lever tests
/// use to exercise the denial path.
pub struct DenyAll;

impl ApprovalPolicy for DenyAll {
    fn approve(&mut self, _tool_name: &str, _arguments: &serde_json::Value) -> ApprovalDecision {
        ApprovalDecision::Deny
    }
}

/// Run one turn against an existing session: call the provider, record the
/// assistant message, parse its output, and either return the final answer or
/// (with the approval policy's consent) execute the requested tool, appending
/// its result to the session.
pub async fn run_step(
    provider: &mut impl Provider,
    registry: &ToolRegistry,
    session: &mut Session,
    events: &mut impl Write,
    policy: &mut impl ApprovalPolicy,
) -> Result<StepOutcome, LoopError> {
    let reply = provider.complete(session).await?;
    // Log the raw model output *before* parsing it, so a turn that fails to
    // parse (fenced or malformed JSON) is still recorded.
    write_event(
        events,
        &Event::ModelResponse {
            content: reply.content.clone(),
        },
    )?;
    session.push_message(Message::new(Role::Assistant, reply.content.clone()));

    let output = ModelOutput::parse(&reply.content)?;

    match output {
        ModelOutput::Final { content } => Ok(StepOutcome::Final(content)),
        ModelOutput::ToolCall {
            tool_name,
            arguments,
            ..
        } => {
            let tool = registry
                .get(&tool_name)
                .ok_or_else(|| LoopError::UnknownTool(tool_name.clone()))?;

            // Ask the approval policy before executing. On denial the tool never
            // runs; we record the denial and feed it back to the model as a
            // `Role::Tool` message so it can choose a different action: the same
            // "environment talks back" channel as a tool result or recovery
            // feedback. Denial is policy, not error: the loop continues.
            if policy.approve(&tool_name, &arguments) == ApprovalDecision::Deny {
                write_event(
                    events,
                    &Event::ToolDenied {
                        tool_name: tool_name.clone(),
                    },
                )?;
                session.push_message(Message::new(
                    Role::Tool,
                    format!("the tool `{tool_name}` was denied by the approval policy"),
                ));
                return Ok(StepOutcome::ToolDenied { tool_name });
            }

            let output = tool.invoke(&arguments)?;
            write_event(
                events,
                &Event::ToolResult {
                    tool_name: tool_name.clone(),
                    output: output.clone(),
                },
            )?;
            session.push_message(Message::new(Role::Tool, output.clone()));
            Ok(StepOutcome::ToolExecuted { tool_name, output })
        }
    }
}

/// Drive the agent loop over a caller-owned session: repeatedly run steps until
/// the model returns a final answer or `max_steps` is reached.
///
/// The `session` is borrowed `&mut`, not constructed here, so its lifecycle
/// belongs to the caller. That lets the caller name a storage directory after
/// the session's id and inspect the session after an error. The
/// session outlives a returned `Err`. Symmetric with [`run_step`], which has
/// always taken `&mut Session`. The caller is responsible for seeding the
/// session with the initial user message before calling.
pub async fn run_loop(
    provider: &mut impl Provider,
    registry: &ToolRegistry,
    session: &mut Session,
    max_steps: usize,
    events: &mut impl Write,
    policy: &mut impl ApprovalPolicy,
) -> Result<String, LoopError> {
    // Record the seeded user input as the opening event(s). The caller seeded
    // these, so they're the run's starting point. Scanning for
    // `Role::User` keeps the whole event stream owned by `run_loop`, testable
    // end-to-end in core, and the CLI just opens the file and hands over a sink.
    for message in &session.messages {
        if message.role == Role::User {
            write_event(
                events,
                &Event::UserMessage {
                    content: message.content.clone(),
                },
            )?;
        }
    }

    for _ in 0..max_steps {
        match run_step(provider, registry, session, events, policy).await {
            Ok(StepOutcome::Final(answer)) => {
                write_event(
                    events,
                    &Event::FinalAnswer {
                        content: answer.clone(),
                    },
                )?;
                return Ok(answer);
            }
            // A tool ran, or one was denied. Either way the loop continues; the
            // result/denial is already appended to the session for the next turn.
            Ok(StepOutcome::ToolExecuted { .. }) | Ok(StepOutcome::ToolDenied { .. }) => {}
            Err(e) => match e.recovery_feedback() {
                Some(feedback) => session.push_message(Message::new(Role::Tool, feedback)),
                None => return Err(e),
            },
        }
    }

    Err(LoopError::MaxStepsExceeded { max_steps })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_round_trips_as_snake_case_json() {
        let json = serde_json::to_string(&Role::Assistant).expect("role should serialize");

        assert_eq!(json, r#""assistant""#);

        let role: Role = serde_json::from_str(&json).expect("role should deserialize");

        assert_eq!(role, Role::Assistant);
    }

    #[test]
    fn message_round_trips_through_json() {
        let message = Message::new(Role::User, "hello");
        let json = serde_json::to_string(&message).expect("message should serialize");
        let restored: Message = serde_json::from_str(&json).expect("message should deserialize");

        assert_eq!(restored, message);
    }

    #[test]
    fn session_round_trips_through_json() {
        let mut session = Session::new("session-1", 1_748_613_022_123);
        session.push_message(Message::new(Role::System, "You are abcb."));
        session.push_message(Message::new(Role::User, "Create a scene."));

        let json = serde_json::to_string(&session).expect("session should serialize");
        let restored: Session = serde_json::from_str(&json).expect("session should deserialize");

        assert_eq!(restored, session);
    }

    #[test]
    fn session_new_records_the_given_timestamp() {
        let session = Session::new("session-1", 1_748_613_022_123);

        assert_eq!(session.id, "session-1");
        assert_eq!(session.created_at, 1_748_613_022_123);
        assert!(session.messages.is_empty());
    }

    #[test]
    fn new_session_id_is_derived_from_the_timestamp() {
        assert_eq!(new_session_id(1_748_613_022_123), "sess-1748613022123");
    }

    #[test]
    fn start_stamps_a_real_time_and_derives_the_id_from_it() {
        let session = Session::start();

        // We can't assert the exact clock value, but the invariant `start`
        // upholds is testable: the id is derived from the stamped time, and the
        // clock read produced a real (post-epoch) moment.
        assert!(session.created_at > 0);
        assert_eq!(session.id, format!("sess-{}", session.created_at));
    }

    #[tokio::test]
    async fn mock_provider_returns_scripted_responses_in_order() {
        let mut provider = MockProvider::new(["first", "second"]);
        let session = Session::new("s", 0);

        let first = provider.complete(&session).await.expect("first response");
        assert_eq!(first, Message::new(Role::Assistant, "first"));

        let second = provider.complete(&session).await.expect("second response");
        assert_eq!(second, Message::new(Role::Assistant, "second"));
    }

    #[tokio::test]
    async fn mock_provider_errors_when_exhausted() {
        let mut provider = MockProvider::new(["only"]);
        let session = Session::new("s", 0);

        provider.complete(&session).await.expect("first response");
        let err = provider
            .complete(&session)
            .await
            .expect_err("provider should be exhausted");
        assert!(matches!(err, ProviderError::NoMoreResponses));
    }

    #[tokio::test]
    async fn one_turn_returns_assistant_reply_from_provider() {
        let mut provider = MockProvider::new(["bot reply"]);

        let reply = one_turn(&mut provider, "hi")
            .await
            .expect("one_turn should produce a reply");

        assert_eq!(reply, Message::new(Role::Assistant, "bot reply"));
    }

    #[test]
    fn event_user_message_serializes_with_snake_case_kind_tag() {
        let event = Event::UserMessage {
            content: "hi".into(),
        };

        let json = serde_json::to_string(&event).expect("event should serialize");

        assert_eq!(json, r#"{"kind":"user_message","content":"hi"}"#);
    }

    #[test]
    fn event_round_trips_through_json() {
        let original = Event::FinalAnswer {
            content: "done".into(),
        };

        let json = serde_json::to_string(&original).expect("event should serialize");
        let restored: Event = serde_json::from_str(&json).expect("event should deserialize");

        assert_eq!(restored, original);
    }

    #[test]
    fn write_event_writes_one_timestamped_newline_terminated_line() {
        let mut buf: Vec<u8> = Vec::new();
        let event = Event::ModelResponse {
            content: "ok".into(),
        };

        write_event(&mut buf, &event).expect("write_event should succeed");

        let text = std::str::from_utf8(&buf).expect("utf8");
        // Exactly one JSONL line, newline-terminated. We don't assert the exact
        // string because `at` is stamped from the clock and varies per run.
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), 1);

        // The record flattens the event and carries a stamped `at`.
        let restored = read_events(buf.as_slice()).expect("read back");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].event, event);
        assert!(restored[0].at > 0, "write_event should stamp a real time");
    }

    #[test]
    fn write_event_supports_appending_multiple_events_as_jsonl() {
        let mut buf: Vec<u8> = Vec::new();

        write_event(
            &mut buf,
            &Event::UserMessage {
                content: "hi".into(),
            },
        )
        .expect("first write");
        write_event(
            &mut buf,
            &Event::FinalAnswer {
                content: "bye".into(),
            },
        )
        .expect("second write");

        let text = std::str::from_utf8(&buf).expect("utf8");
        let lines: Vec<&str> = text.lines().collect();

        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("user_message"));
        assert!(lines[1].contains("final_answer"));
    }

    #[test]
    fn read_events_parses_jsonl_bytes_into_events() {
        let input: &[u8] = b"{\"kind\":\"user_message\",\"content\":\"hi\"}\n\
                             {\"kind\":\"final_answer\",\"content\":\"done\"}\n";

        let events = read_events(input).expect("read_events should parse");

        assert_eq!(events.len(), 2);
        // Lines without an `at` field still parse; `at` defaults to 0.
        assert_eq!(events[0].at, 0);
        assert_eq!(
            events[0].event,
            Event::UserMessage {
                content: "hi".into()
            }
        );
        assert_eq!(
            events[1].event,
            Event::FinalAnswer {
                content: "done".into()
            }
        );
    }

    #[test]
    fn read_events_skips_blank_lines() {
        let input: &[u8] = b"\n{\"kind\":\"user_message\",\"content\":\"hi\"}\n\n";

        let events = read_events(input).expect("read_events should parse");

        assert_eq!(events.len(), 1);
    }

    #[test]
    fn read_events_errors_on_malformed_line() {
        let input: &[u8] = b"not json at all\n";

        let result = read_events(input);

        assert!(matches!(result, Err(EventLogError::Serde(_))));
    }

    #[test]
    fn write_then_read_round_trips_events() {
        let original = vec![
            Event::UserMessage {
                content: "hi".into(),
            },
            Event::ModelResponse {
                content: "ok".into(),
            },
            Event::FinalAnswer {
                content: "done".into(),
            },
        ];

        let mut buf: Vec<u8> = Vec::new();
        for event in &original {
            write_event(&mut buf, event).expect("write");
        }

        let restored = read_events(buf.as_slice()).expect("read");

        let restored_events: Vec<Event> = restored.iter().map(|le| le.event.clone()).collect();
        assert_eq!(restored_events, original);
        // Every record got a real timestamp on the way out.
        assert!(restored.iter().all(|le| le.at > 0));
    }

    #[derive(Deserialize)]
    struct StubEchoArgs {
        text: String,
    }

    struct StubEcho;

    impl Tool for StubEcho {
        fn name(&self) -> &str {
            "stub_echo"
        }

        fn description(&self) -> &str {
            "Echoes the `text` field of its arguments."
        }

        fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
            let typed: StubEchoArgs = serde_json::from_value(args.clone())?;
            Ok(typed.text)
        }
    }

    #[test]
    fn tool_exposes_name_and_description() {
        let tool = StubEcho;

        assert_eq!(tool.name(), "stub_echo");
        assert!(tool.description().to_lowercase().contains("echoes"));
    }

    #[test]
    fn tool_invoke_returns_string_output() {
        let tool = StubEcho;
        let args = serde_json::json!({"text": "hello"});

        let output = tool.invoke(&args).expect("invoke should succeed");

        assert_eq!(output, "hello");
    }

    #[test]
    fn tool_invoke_with_wrong_arg_shape_errors_as_invalid_arguments() {
        let tool = StubEcho;
        let args = serde_json::json!({"wrong_field": 42});

        let err = tool.invoke(&args).expect_err("invoke should fail");

        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn tool_error_converts_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "missing file");

        let tool_err: ToolError = io_err.into();

        assert!(matches!(tool_err, ToolError::Io(_)));
    }

    struct StubNoop;

    impl Tool for StubNoop {
        fn name(&self) -> &str {
            "stub_noop"
        }

        fn description(&self) -> &str {
            "Does nothing and returns an empty string."
        }

        fn invoke(&self, _args: &serde_json::Value) -> Result<String, ToolError> {
            Ok(String::new())
        }
    }

    #[test]
    fn registry_starts_empty() {
        let registry = ToolRegistry::new();

        assert_eq!(registry.names().count(), 0);
    }

    #[test]
    fn register_then_get_returns_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register(StubEcho)
            .expect("first register should succeed");

        let tool = registry
            .get("stub_echo")
            .expect("tool should be registered");
        let args = serde_json::json!({"text": "hi"});

        assert_eq!(tool.invoke(&args).expect("invoke should succeed"), "hi");
    }

    #[test]
    fn get_unknown_name_returns_none() {
        let registry = ToolRegistry::new();

        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn register_duplicate_name_errors() {
        let mut registry = ToolRegistry::new();
        registry.register(StubEcho).expect("first register");

        let err = registry
            .register(StubEcho)
            .expect_err("second register should fail");

        assert_eq!(err, RegistryError::DuplicateName("stub_echo".into()));
    }

    #[test]
    fn names_lists_registered_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(StubEcho).expect("register echo");
        registry.register(StubNoop).expect("register noop");

        let mut names: Vec<&str> = registry.names().collect();
        names.sort();

        assert_eq!(names, vec!["stub_echo", "stub_noop"]);
    }

    #[test]
    fn parse_tool_call_envelope() {
        let raw = r#"{"kind":"tool_call","tool_name":"echo","arguments":{"text":"hi"},"note":"thinking"}"#;

        let output = ModelOutput::parse(raw).expect("should parse");

        assert_eq!(
            output,
            ModelOutput::ToolCall {
                tool_name: "echo".into(),
                arguments: serde_json::json!({"text": "hi"}),
                note: Some("thinking".into()),
            }
        );
    }

    #[test]
    fn parse_tool_call_without_note_defaults_to_none() {
        let raw = r#"{"kind":"tool_call","tool_name":"echo","arguments":{}}"#;

        let output = ModelOutput::parse(raw).expect("should parse");

        assert_eq!(
            output,
            ModelOutput::ToolCall {
                tool_name: "echo".into(),
                arguments: serde_json::json!({}),
                note: None,
            }
        );
    }

    #[test]
    fn parse_final_envelope() {
        let raw = r#"{"kind":"final","content":"all done"}"#;

        let output = ModelOutput::parse(raw).expect("should parse");

        assert_eq!(
            output,
            ModelOutput::Final {
                content: "all done".into()
            }
        );
    }

    #[test]
    fn parse_rejects_invalid_json() {
        let err = ModelOutput::parse("not json").expect_err("should fail");

        assert!(matches!(err, ModelOutputError::Parse(_)));
    }

    #[test]
    fn parse_rejects_unknown_kind() {
        let raw = r#"{"kind":"banana","content":"x"}"#;

        let err = ModelOutput::parse(raw).expect_err("should fail");

        assert!(matches!(err, ModelOutputError::Parse(_)));
    }

    #[test]
    fn parse_tolerates_a_json_code_fence() {
        let raw = "```json\n{\"kind\":\"final\",\"content\":\"done\"}\n```";

        let output = ModelOutput::parse(raw).expect("fenced JSON should parse");

        assert_eq!(
            output,
            ModelOutput::Final {
                content: "done".into()
            }
        );
    }

    #[test]
    fn parse_tolerates_a_bare_fence_with_surrounding_prose() {
        // The common local-model habit: a sentence, then a fenced block.
        let raw = "Here is my response:\n```\n{\"kind\":\"final\",\"content\":\"done\"}\n```";

        let output = ModelOutput::parse(raw).expect("prose + fence should parse");

        assert_eq!(
            output,
            ModelOutput::Final {
                content: "done".into()
            }
        );
    }

    #[test]
    fn parse_still_accepts_clean_json_and_trims_whitespace() {
        let raw = "  {\"kind\":\"final\",\"content\":\"done\"}  ";

        let output = ModelOutput::parse(raw).expect("clean JSON should still parse");

        assert_eq!(
            output,
            ModelOutput::Final {
                content: "done".into()
            }
        );
    }

    #[test]
    fn parse_rejects_prose_with_no_json() {
        // Fence-stripping is a safety net, not magic: prose with no JSON still
        // fails, which the recovery loop and system prompt are there to handle.
        let err = ModelOutput::parse("Sure, I can help with that!").expect_err("no json");

        assert!(matches!(err, ModelOutputError::Parse(_)));
    }

    #[test]
    fn system_prompt_states_the_contract_and_lists_tools() {
        let registry = registry_with_stub_echo();

        let prompt = system_prompt(&registry);

        // The two envelope shapes.
        assert!(prompt.contains(r#""kind":"tool_call""#));
        assert!(prompt.contains(r#""kind":"final""#));
        // The "no fences/prose" instruction the fence-tolerant parser backstops.
        assert!(prompt.contains("no markdown code fences"));
        // The registered tool, name and description.
        assert!(prompt.contains("- stub_echo: Echoes the `text` field of its arguments."));
    }

    fn registry_with_stub_echo() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register(StubEcho).expect("register stub_echo");
        registry
    }

    fn session_with_user(message: &str) -> Session {
        let mut session = Session::new("test", 0);
        session.push_message(Message::new(Role::User, message));
        session
    }

    #[tokio::test]
    async fn run_step_returns_final_for_final_envelope() {
        let mut provider = MockProvider::new([r#"{"kind":"final","content":"all done"}"#]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let outcome = run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("step should succeed");

        assert_eq!(outcome, StepOutcome::Final("all done".into()));
    }

    #[tokio::test]
    async fn run_step_executes_tool_for_tool_call_envelope() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
        ]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let outcome = run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("step should succeed");

        assert_eq!(
            outcome,
            StepOutcome::ToolExecuted {
                tool_name: "stub_echo".into(),
                output: "pong".into(),
            }
        );
    }

    #[tokio::test]
    async fn run_step_appends_assistant_and_tool_messages_to_session() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
        ]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("step should succeed");

        // user + assistant (the tool call) + tool (the result)
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[1].role, Role::Assistant);
        assert_eq!(session.messages[2].role, Role::Tool);
        assert_eq!(session.messages[2].content, "pong");
    }

    #[tokio::test]
    async fn run_step_errors_on_unknown_tool() {
        let mut provider =
            MockProvider::new([r#"{"kind":"tool_call","tool_name":"nonexistent","arguments":{}}"#]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let err = run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect_err("step should fail");

        assert!(matches!(err, LoopError::UnknownTool(name) if name == "nonexistent"));
    }

    #[tokio::test]
    async fn run_step_errors_on_invalid_json() {
        let mut provider = MockProvider::new(["not valid json"]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let err = run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect_err("step should fail");

        assert!(matches!(err, LoopError::Parse(_)));
    }

    #[tokio::test]
    async fn run_step_errors_on_bad_tool_arguments() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"wrong":1}}"#,
        ]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let err = run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect_err("step should fail");

        assert!(matches!(err, LoopError::Tool(_)));
    }

    #[tokio::test]
    async fn run_loop_returns_final_immediately() {
        let mut provider = MockProvider::new([r#"{"kind":"final","content":"done"}"#]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should succeed");

        assert_eq!(answer, "done");
    }

    #[tokio::test]
    async fn run_loop_executes_tool_then_returns_final() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
            r#"{"kind":"final","content":"finished"}"#,
        ]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should succeed");

        assert_eq!(answer, "finished");
    }

    #[tokio::test]
    async fn run_loop_runs_multiple_tools_then_final() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"one"}}"#,
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"two"}}"#,
            r#"{"kind":"final","content":"both done"}"#,
        ]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should succeed");

        assert_eq!(answer, "both done");
    }

    #[tokio::test]
    async fn run_loop_errors_when_max_steps_exceeded() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"a"}}"#,
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"b"}}"#,
        ]);
        let registry = registry_with_stub_echo();

        let err = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            2,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect_err("loop should fail");

        assert!(matches!(err, LoopError::MaxStepsExceeded { max_steps: 2 }));
    }

    #[tokio::test]
    async fn run_loop_recovers_from_parse_error() {
        let mut provider = MockProvider::new([
            "this is not json",
            r#"{"kind":"final","content":"recovered"}"#,
        ]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should recover");

        assert_eq!(answer, "recovered");
    }

    #[tokio::test]
    async fn run_loop_recovers_from_unknown_tool() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"nonexistent","arguments":{}}"#,
            r#"{"kind":"final","content":"recovered"}"#,
        ]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should recover");

        assert_eq!(answer, "recovered");
    }

    #[tokio::test]
    async fn run_loop_recovers_from_bad_arguments() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"wrong":1}}"#,
            r#"{"kind":"final","content":"recovered"}"#,
        ]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should recover");

        assert_eq!(answer, "recovered");
    }

    #[tokio::test]
    async fn run_loop_aborts_on_provider_error() {
        // One scripted tool call, then the provider is exhausted on the next turn.
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"x"}}"#,
        ]);
        let registry = registry_with_stub_echo();

        let err = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect_err("loop should abort");

        assert!(matches!(err, LoopError::Provider(_)));
    }

    #[tokio::test]
    async fn run_loop_recovery_is_bounded_by_max_steps() {
        let mut provider = MockProvider::new(["garbage one", "garbage two"]);
        let registry = registry_with_stub_echo();

        let err = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            2,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect_err("loop should fail");

        assert!(matches!(err, LoopError::MaxStepsExceeded { max_steps: 2 }));
    }

    #[tokio::test]
    async fn run_loop_writes_the_full_event_stream_as_jsonl() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
            r#"{"kind":"final","content":"done"}"#,
        ]);
        let registry = registry_with_stub_echo();
        let mut log: Vec<u8> = Vec::new();

        run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut log,
            &mut AllowAll,
        )
        .await
        .expect("loop should succeed");

        let events: Vec<Event> = read_events(log.as_slice())
            .expect("log should parse")
            .into_iter()
            .map(|logged| logged.event)
            .collect();

        assert_eq!(
            events,
            vec![
                Event::UserMessage {
                    content: "hi".into()
                },
                // turn 1: the tool-call envelope (logged before parsing), then the result
                Event::ModelResponse {
                    content:
                        r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#
                            .into()
                },
                Event::ToolResult {
                    tool_name: "stub_echo".into(),
                    output: "pong".into()
                },
                // turn 2: the final envelope, then the loop's extracted final answer
                Event::ModelResponse {
                    content: r#"{"kind":"final","content":"done"}"#.into()
                },
                Event::FinalAnswer {
                    content: "done".into()
                },
            ]
        );
    }

    #[tokio::test]
    async fn run_step_denied_by_policy_skips_the_tool_and_feeds_back() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
        ]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let outcome = run_step(
            &mut provider,
            &registry,
            &mut session,
            &mut io::sink(),
            &mut DenyAll,
        )
        .await
        .expect("denial is policy, not error; the step still succeeds");

        assert_eq!(
            outcome,
            StepOutcome::ToolDenied {
                tool_name: "stub_echo".into()
            }
        );
        // The tool never ran (no "pong" result anywhere); the last message is the
        // denial, fed back as Role::Tool so the next turn can choose differently.
        assert!(!session.messages.iter().any(|m| m.content == "pong"));
        let last = session.messages.last().expect("a message");
        assert_eq!(last.role, Role::Tool);
        assert!(last.content.contains("denied"));
    }

    #[tokio::test]
    async fn run_loop_logs_denial_and_keeps_going() {
        let mut provider = MockProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
            r#"{"kind":"final","content":"ok then"}"#,
        ]);
        let registry = registry_with_stub_echo();
        let mut log: Vec<u8> = Vec::new();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut log,
            &mut DenyAll,
        )
        .await
        .expect("loop reaches a final answer despite the denial");

        assert_eq!(answer, "ok then");

        let events: Vec<Event> = read_events(log.as_slice())
            .expect("log parses")
            .into_iter()
            .map(|logged| logged.event)
            .collect();
        assert_eq!(
            events,
            vec![
                Event::UserMessage {
                    content: "hi".into()
                },
                Event::ModelResponse {
                    content:
                        r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#
                            .into()
                },
                // denied, not executed: a ToolDenied event, no ToolResult
                Event::ToolDenied {
                    tool_name: "stub_echo".into()
                },
                Event::ModelResponse {
                    content: r#"{"kind":"final","content":"ok then"}"#.into()
                },
                Event::FinalAnswer {
                    content: "ok then".into()
                },
            ]
        );
    }

    #[test]
    fn recovery_feedback_classifies_errors() {
        // Recoverable: model can plausibly fix these by trying again.
        assert!(
            LoopError::UnknownTool("x".into())
                .recovery_feedback()
                .is_some()
        );
        assert!(
            LoopError::Tool(ToolError::Execution("boom".into()))
                .recovery_feedback()
                .is_some()
        );

        // Not recoverable: environment / provider / terminal conditions.
        let io = ToolError::Io(io::Error::new(io::ErrorKind::PermissionDenied, "denied"));
        assert!(LoopError::Tool(io).recovery_feedback().is_none());
        assert!(
            LoopError::Provider(ProviderError::NoMoreResponses)
                .recovery_feedback()
                .is_none()
        );
        assert!(
            LoopError::MaxStepsExceeded { max_steps: 3 }
                .recovery_feedback()
                .is_none()
        );
    }

    /// A provider that records every session it is asked to complete, so tests
    /// can assert what the model actually saw on each turn.
    struct RecordingProvider {
        scripted: VecDeque<String>,
        seen: Vec<Session>,
    }

    impl RecordingProvider {
        fn new<I, S>(responses: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            Self {
                scripted: responses.into_iter().map(Into::into).collect(),
                seen: Vec::new(),
            }
        }
    }

    impl Provider for RecordingProvider {
        async fn complete(&mut self, session: &Session) -> Result<Message, ProviderError> {
            self.seen.push(session.clone());
            let content = self
                .scripted
                .pop_front()
                .ok_or(ProviderError::NoMoreResponses)?;
            Ok(Message::new(Role::Assistant, content))
        }
    }

    #[tokio::test]
    async fn run_loop_feeds_tool_result_back_to_model() {
        let mut provider = RecordingProvider::new([
            r#"{"kind":"tool_call","tool_name":"stub_echo","arguments":{"text":"pong"}}"#,
            r#"{"kind":"final","content":"done"}"#,
        ]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(
            &mut provider,
            &registry,
            &mut session_with_user("hi"),
            5,
            &mut io::sink(),
            &mut AllowAll,
        )
        .await
        .expect("loop should succeed");
        assert_eq!(answer, "done");

        // The provider was called twice. The second call's session must include
        // the tool result, proving the loop fed it back for the model to see.
        assert_eq!(provider.seen.len(), 2);
        let second_turn = &provider.seen[1];
        assert!(
            second_turn
                .messages
                .iter()
                .any(|m| m.role == Role::Tool && m.content == "pong"),
            "second turn should see the tool result; saw: {:?}",
            second_turn.messages
        );
    }
}
