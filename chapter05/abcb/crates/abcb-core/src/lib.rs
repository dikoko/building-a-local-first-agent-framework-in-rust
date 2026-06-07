use std::collections::VecDeque;
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

#[derive(Debug, Eq, PartialEq)]
pub enum ProviderError {
    NoMoreResponses,
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::NoMoreResponses => write!(f, "no more scripted responses"),
        }
    }
}

impl std::error::Error for ProviderError {}

pub trait Provider {
    fn complete(&mut self, session: &Session) -> Result<Message, ProviderError>;
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
    fn complete(&mut self, _session: &Session) -> Result<Message, ProviderError> {
        let content = self
            .scripted
            .pop_front()
            .ok_or(ProviderError::NoMoreResponses)?;
        Ok(Message::new(Role::Assistant, content))
    }
}

pub fn one_turn(
    provider: &mut impl Provider,
    user_message: impl Into<String>,
) -> Result<Message, ProviderError> {
    let mut session = Session::new("one-turn");
    session.push_message(Message::new(Role::User, user_message));
    provider.complete(&session)
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
            EventLogError::Serde(e) => write!(f, "event log json error: {e}"),
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

    #[test]
    fn mock_provider_returns_scripted_responses_in_order() {
        let mut provider = MockProvider::new(["first", "second"]);
        let session = Session::new("s");

        let first = provider.complete(&session).expect("first response");
        assert_eq!(first, Message::new(Role::Assistant, "first"));

        let second = provider.complete(&session).expect("second response");
        assert_eq!(second, Message::new(Role::Assistant, "second"));
    }

    #[test]
    fn mock_provider_errors_when_exhausted() {
        let mut provider = MockProvider::new(["only"]);
        let session = Session::new("s");

        provider.complete(&session).expect("first response");
        let err = provider
            .complete(&session)
            .expect_err("provider should be exhausted");
        assert!(matches!(err, ProviderError::NoMoreResponses));
    }

    #[test]
    fn one_turn_returns_assistant_reply_from_provider() {
        let mut provider = MockProvider::new(["bot reply"]);

        let reply = one_turn(&mut provider, "hi").expect("one_turn should produce a reply");

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
}
