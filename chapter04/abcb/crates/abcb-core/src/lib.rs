use std::collections::VecDeque;
use std::fmt;

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
}
