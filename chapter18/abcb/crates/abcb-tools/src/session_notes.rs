use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use abcb_core::{Tool, ToolError, now_millis};
use serde::{Deserialize, Serialize};

/// One note as stored in the JSONL notes file: the text plus the time it was
/// written (Unix-epoch millis). Mirrors the event log's `LoggedEvent` envelope:
/// a timestamped record, one per line, and stamps the time with `now_millis`,
/// the same clock-at-the-edge helper the event log uses.
#[derive(Debug, Deserialize, Serialize)]
struct Note {
    at: u64,
    text: String,
}

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
        "Appends `note` to the project notes as a timestamped JSONL record."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: AppendArgs = serde_json::from_value(args.clone())?;
        let note = Note {
            at: now_millis(),
            text: typed.note,
        };
        // Serializing a `{u64, String}` record is infallible in practice; map the
        // theoretical error to `Execution` rather than the `?`-default
        // `InvalidArguments`, which would mislabel an internal failure as bad args.
        let line = serde_json::to_string(&note).map_err(|e| ToolError::Execution(e.to_string()))?;

        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
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
        "Returns notes whose text contains `query` as a substring, one per line."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: SearchArgs = serde_json::from_value(args.clone())?;

        if !self.path.exists() {
            return Ok("no matches".into());
        }

        let content = fs::read_to_string(&self.path)?;
        // Parse each JSONL line into a `Note` and match on its text. Lines that
        // don't parse are skipped rather than erroring: notes are best-effort
        // memory, so a stray/corrupt line shouldn't break a search. (Contrast the
        // event log, which errors on malformed lines because it's an audit record.)
        let matches: Vec<String> = content
            .lines()
            .filter_map(|line| serde_json::from_str::<Note>(line).ok())
            .filter(|note| note.text.contains(&typed.query))
            .map(|note| note.text)
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

    /// Read the tool's file back as parsed notes (the test module can see the
    /// private `path` field because it's a child module of this one).
    fn stored_notes(path: &PathBuf) -> Vec<Note> {
        let content = fs::read_to_string(path).expect("read");
        content
            .lines()
            .map(|line| serde_json::from_str::<Note>(line).expect("note line parses"))
            .collect()
    }

    #[test]
    fn append_writes_note_as_timestamped_jsonl_record() {
        let file = NamedTempFile::new().expect("temp file");
        let tool = SessionNoteAppend::new(file.path().to_path_buf());

        tool.invoke(&serde_json::json!({"note": "remember this"}))
            .expect("append");

        let notes = stored_notes(&tool.path);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].text, "remember this");
        assert!(notes[0].at > 0, "append should stamp a real time");
    }

    #[test]
    fn append_preserves_existing_notes() {
        let file = NamedTempFile::new().expect("temp file");
        let tool = SessionNoteAppend::new(file.path().to_path_buf());

        tool.invoke(&serde_json::json!({"note": "first"}))
            .expect("first");
        tool.invoke(&serde_json::json!({"note": "second"}))
            .expect("second");

        let notes = stored_notes(&tool.path);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].text, "first");
        assert_eq!(notes[1].text, "second");
    }

    #[test]
    fn search_finds_substring_match() {
        let file = NamedTempFile::new().expect("temp file");
        let append = SessionNoteAppend::new(file.path().to_path_buf());
        for note in ["buy milk", "feed cat", "buy bread"] {
            append
                .invoke(&serde_json::json!({ "note": note }))
                .expect("append");
        }

        let search = SessionNoteSearch::new(file.path().to_path_buf());
        let output = search
            .invoke(&serde_json::json!({"query": "buy"}))
            .expect("search");

        assert_eq!(output, "buy milk\nbuy bread");
    }

    #[test]
    fn search_returns_no_matches_when_query_absent() {
        let file = NamedTempFile::new().expect("temp file");
        let append = SessionNoteAppend::new(file.path().to_path_buf());
        append
            .invoke(&serde_json::json!({"note": "hello"}))
            .expect("append");

        let search = SessionNoteSearch::new(file.path().to_path_buf());
        let output = search
            .invoke(&serde_json::json!({"query": "missing"}))
            .expect("search");

        assert_eq!(output, "no matches");
    }

    #[test]
    fn search_returns_no_matches_when_file_missing() {
        let path = PathBuf::from("/tmp/abcb-this-path-should-not-exist-xyz.jsonl");
        assert!(!path.exists(), "test precondition: path should not exist");

        let tool = SessionNoteSearch::new(path);
        let output = tool
            .invoke(&serde_json::json!({"query": "anything"}))
            .expect("search");

        assert_eq!(output, "no matches");
    }
}
