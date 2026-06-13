use abcb_core::{Tool, ToolError};
use serde::Deserialize;

#[derive(Deserialize)]
struct AddArgs {
    a: f64,
    b: f64,
}

pub struct AddNumbers;

impl Tool for AddNumbers {
    fn name(&self) -> &str {
        "add_numbers"
    }

    fn description(&self) -> &str {
        "Adds two numbers `a` and `b` and returns their sum."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: AddArgs = serde_json::from_value(args.clone())?;
        let sum = typed.a + typed.b;
        Ok(format!("{sum}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_numbers_returns_sum_of_floats() {
        let tool = AddNumbers;
        let args = serde_json::json!({"a": 1.5, "b": 2.25});

        let output = tool.invoke(&args).expect("invoke");

        assert_eq!(output, "3.75");
    }

    #[test]
    fn add_numbers_works_with_integer_literals() {
        let tool = AddNumbers;
        let args = serde_json::json!({"a": 2, "b": 3});

        let output = tool.invoke(&args).expect("invoke");

        assert_eq!(output, "5");
    }

    #[test]
    fn add_numbers_errors_on_missing_arg() {
        let tool = AddNumbers;
        let args = serde_json::json!({"a": 1.0});

        let err = tool.invoke(&args).expect_err("invoke should fail");

        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }
}
