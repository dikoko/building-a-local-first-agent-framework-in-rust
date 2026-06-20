use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use abcb_core::{Tool, ToolError};
use serde::Deserialize;

#[derive(Deserialize)]
struct AppendArgs {
    note: String,
}

pub struct SessionNoteAppend {
    path: PathBuf,
}

impl SessionNoteAppend {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Tool for SessionNoteAppend {
    fn name(&self) -> &str {
        "session_note_append"
    }

    fn description(&self) -> &str {
        "Appends `note` as a new line in the session notes file."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: AppendArgs = serde_json::from_value(args.clone())?;
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;
        writeln!(file, "{}", typed.note)?;
        Ok("note appended".into())
    }
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

pub struct SessionNoteSearch {
    path: PathBuf,
}

impl SessionNoteSearch {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Tool for SessionNoteSearch {
    fn name(&self) -> &str {
        "session_note_search"
    }

    fn description(&self) -> &str {
        "Returns notes containing `query` as a substring, one match per line."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: SearchArgs = serde_json::from_value(args.clone())?;

        if !self.path.exists() {
            return Ok("no matches".into());
        }

        let content = fs::read_to_string(&self.path)?;
        let matches: Vec<&str> = content
            .lines()
            .filter(|line| line.contains(&typed.query))
            .collect();

        if matches.is_empty() {
            Ok("no matches".into())
        } else {
            Ok(matches.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn append_writes_note_as_new_line() {
        let file = NamedTempFile::new().expect("temp file");
        let tool = SessionNoteAppend::new(file.path().to_path_buf());

        tool.invoke(&serde_json::json!({"note": "remember this"}))
            .expect("append");

        let contents = fs::read_to_string(file.path()).expect("read");
        assert_eq!(contents, "remember this\n");
    }

    #[test]
    fn append_preserves_existing_notes() {
        let mut file = NamedTempFile::new().expect("temp file");
        Write::write_all(file.as_file_mut(), b"first\n").expect("seed");

        let tool = SessionNoteAppend::new(file.path().to_path_buf());
        tool.invoke(&serde_json::json!({"note": "second"}))
            .expect("append");

        let contents = fs::read_to_string(file.path()).expect("read");
        assert_eq!(contents, "first\nsecond\n");
    }

    #[test]
    fn search_finds_substring_match() {
        let mut file = NamedTempFile::new().expect("temp file");
        Write::write_all(file.as_file_mut(), b"buy milk\nfeed cat\nbuy bread\n").expect("seed");

        let tool = SessionNoteSearch::new(file.path().to_path_buf());
        let output = tool
            .invoke(&serde_json::json!({"query": "buy"}))
            .expect("search");

        assert_eq!(output, "buy milk\nbuy bread");
    }

    #[test]
    fn search_returns_no_matches_when_query_absent() {
        let mut file = NamedTempFile::new().expect("temp file");
        Write::write_all(file.as_file_mut(), b"hello\n").expect("seed");

        let tool = SessionNoteSearch::new(file.path().to_path_buf());
        let output = tool
            .invoke(&serde_json::json!({"query": "missing"}))
            .expect("search");

        assert_eq!(output, "no matches");
    }

    #[test]
    fn search_returns_no_matches_when_file_missing() {
        let path = PathBuf::from("/tmp/abcb-this-path-should-not-exist-xyz.txt");
        assert!(!path.exists(), "test precondition: path should not exist");

        let tool = SessionNoteSearch::new(path);
        let output = tool
            .invoke(&serde_json::json!({"query": "anything"}))
            .expect("search");

        assert_eq!(output, "no matches");
    }
}
