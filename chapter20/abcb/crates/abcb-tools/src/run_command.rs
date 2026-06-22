use std::path::PathBuf;
use std::process::Command;

use abcb_core::{Tool, ToolError};
use serde::Deserialize;

#[derive(Deserialize)]
struct RunCommandArgs {
    command: String,
}

/// Runs a shell command in the working directory supplied at construction
/// time. Unlike the filesystem tools, this is deliberately *not* sandboxed —
/// it can do anything the user can. The control point is the loop's
/// `ApprovalPolicy`, which sees every call before it runs; that's why the tool
/// takes a plain command string (ergonomic for the model) rather than an
/// injection-safe argv array.
pub struct RunCommand {
    working_dir: PathBuf,
}

impl RunCommand {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for RunCommand {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Runs a shell command in the project directory and returns its combined stdout and stderr."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: RunCommandArgs = serde_json::from_value(args.clone())?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(&typed.command)
            .current_dir(&self.working_dir)
            .output()?; // only a failure to *spawn* is an Io error

        // Combine both streams the way a human reading a terminal sees them.
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));

        // A non-zero exit is NOT a tool error — the command ran, so the tool
        // succeeded. But surface the status so the model can tell: some failing
        // commands (e.g. `false`) produce no output at all.
        if !output.status.success() {
            let status = output.status.code().map_or_else(
                || "terminated by signal".to_string(),
                |c| format!("exit code {c}"),
            );
            combined.push_str(&format!("\n[command failed: {status}]"));
        }

        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn run_command_returns_command_output() {
        let dir = tempdir().expect("tempdir");
        let tool = RunCommand::new(dir.path().to_path_buf());

        let out = tool
            .invoke(&serde_json::json!({"command": "echo hello"}))
            .expect("run");

        assert_eq!(out, "hello\n");
    }

    #[test]
    fn run_command_runs_in_the_working_directory() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("marker.txt"), "in the dir").expect("write");
        let tool = RunCommand::new(dir.path().to_path_buf());

        // A *relative* path resolves against the child's cwd, so this only
        // works if `current_dir` took effect — proves it without depending on
        // how `pwd` prints symlinked temp paths.
        let out = tool
            .invoke(&serde_json::json!({"command": "cat marker.txt"}))
            .expect("run");

        assert_eq!(out, "in the dir");
    }

    #[test]
    fn run_command_surfaces_a_nonzero_exit() {
        let dir = tempdir().expect("tempdir");
        let tool = RunCommand::new(dir.path().to_path_buf());

        let out = tool
            .invoke(&serde_json::json!({"command": "exit 3"}))
            .expect("a failing command is still a successful tool call");

        assert!(out.contains("exit code 3"), "got: {out:?}");
    }
}
