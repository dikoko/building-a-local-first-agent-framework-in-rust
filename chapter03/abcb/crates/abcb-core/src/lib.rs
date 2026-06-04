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
}
