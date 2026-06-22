use std::fs;
use std::path::{Path, PathBuf};

use abcb_core::{Tool, ToolError};
use serde::Deserialize;

/// Resolve `requested` against `root` and guarantee the result stays inside
/// `root`. `canonicalize` is the security primitive here: it resolves `..`
/// segments and follows symlinks, so a `../../etc/passwd` traversal *and* a
/// symlink pointing out of the tree both reduce to a real path we then
/// prefix-check against the (also canonicalized) root.
///
/// Because `canonicalize` touches the filesystem, the path must exist — a
/// missing file surfaces as `ToolError::Io` via `?`. A path that resolves
/// *outside* the root is a policy violation rather than an I/O fault, so it's
/// `ToolError::Execution`, which `LoopError::recovery_feedback` treats as
/// model-recoverable: the model can retry with a path inside the project.
fn confine_to_root(root: &Path, requested: &str) -> Result<PathBuf, ToolError> {
    // `join` with an absolute `requested` discards `root` entirely — that's
    // fine, the canonicalized result is still prefix-checked below, so an
    // absolute path outside the root is rejected just like a `..` escape.
    let resolved = root.join(requested).canonicalize()?;
    let canonical_root = root.canonicalize()?;
    if resolved.starts_with(&canonical_root) {
        Ok(resolved)
    } else {
        Err(ToolError::Execution(format!(
            "path `{requested}` is outside the project root"
        )))
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
}

/// Reads a UTF-8 text file, confined to the project root supplied at
/// construction time.
pub struct ReadFile {
    root: PathBuf,
}

impl ReadFile {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the UTF-8 text file at `path` (relative to the project root) and returns its contents."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: ReadFileArgs = serde_json::from_value(args.clone())?;
        let path = confine_to_root(&self.root, &typed.path)?;
        Ok(fs::read_to_string(path)?)
    }
}

#[derive(Deserialize)]
struct ListDirArgs {
    path: String,
}

/// Lists the entries of a directory, confined to the project root.
pub struct ListDir {
    root: PathBuf,
}

impl ListDir {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Tool for ListDir {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "Lists the entries of the directory at `path` (relative to the project root), one per line, sorted."
    }

    fn invoke(&self, args: &serde_json::Value) -> Result<String, ToolError> {
        let typed: ListDirArgs = serde_json::from_value(args.clone())?;
        let dir = confine_to_root(&self.root, &typed.path)?;

        let mut names = Vec::new();
        for entry in fs::read_dir(dir)? {
            names.push(entry?.file_name().to_string_lossy().into_owned());
        }
        // `read_dir` yields entries in an OS-dependent order; sort so the
        // output is deterministic and testable (same reason `system_prompt`
        // sorts its tool list).
        names.sort();
        Ok(names.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn read_file_reads_a_file_inside_the_root() {
        let root = tempdir().expect("tempdir");
        fs::write(root.path().join("hello.txt"), "hi there").expect("write");

        let tool = ReadFile::new(root.path().to_path_buf());
        let out = tool
            .invoke(&serde_json::json!({"path": "hello.txt"}))
            .expect("read");

        assert_eq!(out, "hi there");
    }

    #[test]
    fn read_file_rejects_a_path_outside_the_root() {
        let root = tempdir().expect("root");
        let outside = tempdir().expect("outside");
        let secret = outside.path().join("secret.txt");
        fs::write(&secret, "top secret").expect("write");

        let tool = ReadFile::new(root.path().to_path_buf());
        let err = tool
            .invoke(&serde_json::json!({"path": secret.to_str().unwrap()}))
            .expect_err("an absolute path outside the root must be rejected");

        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[test]
    fn list_dir_lists_entries_sorted() {
        let root = tempdir().expect("root");
        fs::write(root.path().join("b.txt"), "").expect("b");
        fs::write(root.path().join("a.txt"), "").expect("a");
        fs::create_dir(root.path().join("sub")).expect("sub");

        let tool = ListDir::new(root.path().to_path_buf());
        let out = tool
            .invoke(&serde_json::json!({"path": "."}))
            .expect("list");

        assert_eq!(out, "a.txt\nb.txt\nsub");
    }

    #[test]
    fn list_dir_rejects_a_path_outside_the_root() {
        let root = tempdir().expect("root");
        let outside = tempdir().expect("outside");

        let tool = ListDir::new(root.path().to_path_buf());
        let err = tool
            .invoke(&serde_json::json!({"path": outside.path().to_str().unwrap()}))
            .expect_err("listing a directory outside the root must be rejected");

        assert!(matches!(err, ToolError::Execution(_)));
    }
}
