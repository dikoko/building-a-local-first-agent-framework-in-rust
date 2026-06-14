//! HTTP providers for OpenAI-compatible chat endpoints.
//!
//! [`OpenAiCompatProvider`] speaks the `/v1/chat/completions` protocol, which is
//! shared by many local model runtimes (MLX's server, llama.cpp's server,
//! Ollama's OpenAI shim, vLLM, LM Studio, …). The runtime is a *deployment*
//! detail chosen by config (`base_url` + `model`), not a separate provider type:
//! we split providers by wire protocol, not by backend brand.

use abcb_core::{Message, Provider, ProviderError, Role, Session};
use serde::{Deserialize, Serialize};

/// A provider that talks to any OpenAI-compatible `/chat/completions` endpoint.
#[derive(Debug)]
pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    /// OpenAI-compatible base URL, e.g. `http://localhost:8083/v1`.
    base_url: String,
    /// Model identifier sent in each request (the local model path, P3-T04).
    model: String,
}

impl OpenAiCompatProvider {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    /// Build the request body for one completion from the session's history.
    fn build_request<'a>(&'a self, session: &'a Session) -> ChatRequest<'a> {
        ChatRequest {
            model: &self.model,
            messages: session
                .messages
                .iter()
                .map(|m| WireMessage {
                    role: wire_role(&m.role),
                    content: &m.content,
                })
                .collect(),
            // Non-streaming responses for now (P3-T05).
            stream: false,
        }
    }
}

impl Provider for OpenAiCompatProvider {
    async fn complete(&mut self, session: &Session) -> Result<Message, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let request = self.build_request(session);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ProviderError::Backend(Box::new(e)))?;

        // Read the status before consuming the body, so a non-2xx can report it.
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ProviderError::Backend(Box::new(e)))?;

        if !status.is_success() {
            return Err(ProviderError::Backend(
                format!("HTTP {status} from {url}: {body}").into(),
            ));
        }

        let content = parse_response(&body)?;
        Ok(Message::new(Role::Assistant, content))
    }
}

/// Map our internal [`Role`] onto an OpenAI wire role.
///
/// `Tool` maps to `"user"` deliberately: local chat templates typically only
/// understand user/assistant/system, and OpenAI's real `"tool"` role requires a
/// `tool_call_id` we don't carry. Feeding tool results back as a user turn is
/// the standard local-model workaround.
fn wire_role(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "user",
    }
}

/// Extract the assistant text from a `/chat/completions` response body.
fn parse_response(body: &str) -> Result<String, ProviderError> {
    let parsed: ChatResponse =
        serde_json::from_str(body).map_err(|e| ProviderError::Backend(Box::new(e)))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .ok_or_else(|| ProviderError::Backend("response contained no choices".into()))
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    stream: bool,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> OpenAiCompatProvider {
        OpenAiCompatProvider::new("http://localhost:8083/v1", "/models/gemma4")
    }

    #[test]
    fn wire_role_maps_tool_to_user() {
        assert_eq!(wire_role(&Role::User), "user");
        assert_eq!(wire_role(&Role::Assistant), "assistant");
        assert_eq!(wire_role(&Role::System), "system");
        assert_eq!(wire_role(&Role::Tool), "user");
    }

    #[test]
    fn build_request_serializes_model_messages_and_stream() {
        let mut session = Session::new("s");
        session.push_message(Message::new(Role::User, "hi"));
        session.push_message(Message::new(Role::Tool, "tool result"));

        let value = serde_json::to_value(provider().build_request(&session))
            .expect("request should serialize");

        assert_eq!(value["model"], "/models/gemma4");
        assert_eq!(value["stream"], false);
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"], "hi");
        // Tool messages go out on the wire as user turns.
        assert_eq!(value["messages"][1]["role"], "user");
        assert_eq!(value["messages"][1]["content"], "tool result");
    }

    #[test]
    fn parse_response_extracts_first_choice_content() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"hello there"}}]}"#;

        let content = parse_response(body).expect("should parse");

        assert_eq!(content, "hello there");
    }

    #[test]
    fn parse_response_errors_on_empty_choices() {
        let body = r#"{"choices":[]}"#;

        let err = parse_response(body).expect_err("no choices should error");

        assert!(matches!(err, ProviderError::Backend(_)));
    }

    #[test]
    fn parse_response_errors_on_malformed_json() {
        let err = parse_response("not json").expect_err("malformed should error");

        assert!(matches!(err, ProviderError::Backend(_)));
    }
}
