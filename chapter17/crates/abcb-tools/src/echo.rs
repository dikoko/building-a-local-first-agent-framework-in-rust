use abcb_core::{Tool, ToolError};
use serde::Deserialize;

#[derive(Deserialize)]
struct EchoArgs {
    text: String,
}

pub struct Echo;

impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the `text` field of its arguments back as the output."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: EchoArgs = serde_json::from_value(args.clone())?;
        Ok(typed.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_returns_text_arg() {
        let tool = Echo;
        let args = serde_json::json!({"text": "hello"});

        let output = tool.invoke(&args).expect("invoke");

        assert_eq!(output, "hello");
    }

    #[test]
    fn echo_errors_on_missing_text_arg() {
        let tool = Echo;
        let args = serde_json::json!({});

        let err = tool.invoke(&args).expect_err("invoke should fail");

        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn echo_has_name_and_description() {
        let tool = Echo;

        assert_eq!(tool.name(), "echo");
        assert!(!tool.description().is_empty());
    }
}
