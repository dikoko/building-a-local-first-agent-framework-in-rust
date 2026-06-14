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
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
        }
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }
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
    /// error at the trait boundary — keeping core dependency-free and the
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
    let mut session = Session::new("one-turn");
    session.push_message(Message::new(Role::User, user_message));
    provider.complete(&session).await
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    UserMessage { content: String },
    ModelResponse { content: String },
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

pub fn write_event(writer: &mut impl Write, event: &Event) -> Result<(), EventLogError> {
    serde_json::to_writer(&mut *writer, event)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub fn read_events(reader: impl BufRead) -> Result<Vec<Event>, EventLogError> {
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(trimmed)?;
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
        Ok(serde_json::from_str(raw)?)
    }
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

#[derive(Clone, Debug, PartialEq)]
pub enum StepOutcome {
    Final(String),
    ToolExecuted { tool_name: String, output: String },
}

#[derive(Debug)]
pub enum LoopError {
    Provider(ProviderError),
    Parse(ModelOutputError),
    UnknownTool(String),
    Tool(ToolError),
    MaxStepsExceeded { max_steps: usize },
}

impl fmt::Display for LoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoopError::Provider(e) => write!(f, "provider error: {e}"),
            LoopError::Parse(e) => write!(f, "{e}"),
            LoopError::UnknownTool(name) => write!(f, "unknown tool: {name}"),
            LoopError::Tool(e) => write!(f, "{e}"),
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
            LoopError::MaxStepsExceeded { .. } => None,
        }
    }
}

impl LoopError {
    /// Classify this error for the agent loop's recovery policy.
    ///
    /// Returns `Some(feedback)` when the model can plausibly fix the problem by
    /// trying again — the feedback is appended to the session as a `Role::Tool`
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

/// Run one turn against an existing session: call the provider, record the
/// assistant message, parse its output, and either return the final answer or
/// execute the requested tool (appending its result to the session).
pub async fn run_step(
    provider: &mut impl Provider,
    registry: &ToolRegistry,
    session: &mut Session,
) -> Result<StepOutcome, LoopError> {
    let reply = provider.complete(session).await?;
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
            let output = tool.invoke(&arguments)?;
            session.push_message(Message::new(Role::Tool, output.clone()));
            Ok(StepOutcome::ToolExecuted { tool_name, output })
        }
    }
}

/// Drive the agent loop: seed a session with the user message, then repeatedly
/// run steps until the model returns a final answer or `max_steps` is reached.
pub async fn run_loop(
    provider: &mut impl Provider,
    registry: &ToolRegistry,
    user_message: impl Into<String>,
    max_steps: usize,
) -> Result<String, LoopError> {
    let mut session = Session::new("run-loop");
    session.push_message(Message::new(Role::User, user_message));

    for _ in 0..max_steps {
        match run_step(provider, registry, &mut session).await {
            Ok(StepOutcome::Final(answer)) => return Ok(answer),
            Ok(StepOutcome::ToolExecuted { .. }) => {}
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
        let mut session = Session::new("session-1");
        session.push_message(Message::new(Role::System, "You are abcb."));
        session.push_message(Message::new(Role::User, "Create a scene."));

        let json = serde_json::to_string(&session).expect("session should serialize");
        let restored: Session = serde_json::from_str(&json).expect("session should deserialize");

        assert_eq!(restored, session);
    }

    #[tokio::test]
    async fn mock_provider_returns_scripted_responses_in_order() {
        let mut provider = MockProvider::new(["first", "second"]);
        let session = Session::new("s");

        let first = provider.complete(&session).await.expect("first response");
        assert_eq!(first, Message::new(Role::Assistant, "first"));

        let second = provider.complete(&session).await.expect("second response");
        assert_eq!(second, Message::new(Role::Assistant, "second"));
    }

    #[tokio::test]
    async fn mock_provider_errors_when_exhausted() {
        let mut provider = MockProvider::new(["only"]);
        let session = Session::new("s");

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
    fn write_event_appends_newline_terminated_json_line() {
        let mut buf: Vec<u8> = Vec::new();
        let event = Event::ModelResponse {
            content: "ok".into(),
        };

        write_event(&mut buf, &event).expect("write_event should succeed");

        assert_eq!(
            std::str::from_utf8(&buf).expect("utf8"),
            "{\"kind\":\"model_response\",\"content\":\"ok\"}\n"
        );
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
        assert_eq!(
            events[0],
            Event::UserMessage {
                content: "hi".into()
            }
        );
        assert_eq!(
            events[1],
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

        assert_eq!(restored, original);
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

    fn registry_with_stub_echo() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register(StubEcho).expect("register stub_echo");
        registry
    }

    fn session_with_user(message: &str) -> Session {
        let mut session = Session::new("test");
        session.push_message(Message::new(Role::User, message));
        session
    }

    #[tokio::test]
    async fn run_step_returns_final_for_final_envelope() {
        let mut provider = MockProvider::new([r#"{"kind":"final","content":"all done"}"#]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let outcome = run_step(&mut provider, &registry, &mut session)
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

        let outcome = run_step(&mut provider, &registry, &mut session)
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

        run_step(&mut provider, &registry, &mut session)
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

        let err = run_step(&mut provider, &registry, &mut session)
            .await
            .expect_err("step should fail");

        assert!(matches!(err, LoopError::UnknownTool(name) if name == "nonexistent"));
    }

    #[tokio::test]
    async fn run_step_errors_on_invalid_json() {
        let mut provider = MockProvider::new(["not valid json"]);
        let registry = registry_with_stub_echo();
        let mut session = session_with_user("hi");

        let err = run_step(&mut provider, &registry, &mut session)
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

        let err = run_step(&mut provider, &registry, &mut session)
            .await
            .expect_err("step should fail");

        assert!(matches!(err, LoopError::Tool(_)));
    }

    #[tokio::test]
    async fn run_loop_returns_final_immediately() {
        let mut provider = MockProvider::new([r#"{"kind":"final","content":"done"}"#]);
        let registry = registry_with_stub_echo();

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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

        let err = run_loop(&mut provider, &registry, "hi", 2)
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

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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

        let err = run_loop(&mut provider, &registry, "hi", 5)
            .await
            .expect_err("loop should abort");

        assert!(matches!(err, LoopError::Provider(_)));
    }

    #[tokio::test]
    async fn run_loop_recovery_is_bounded_by_max_steps() {
        let mut provider = MockProvider::new(["garbage one", "garbage two"]);
        let registry = registry_with_stub_echo();

        let err = run_loop(&mut provider, &registry, "hi", 2)
            .await
            .expect_err("loop should fail");

        assert!(matches!(err, LoopError::MaxStepsExceeded { max_steps: 2 }));
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

        let answer = run_loop(&mut provider, &registry, "hi", 5)
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
