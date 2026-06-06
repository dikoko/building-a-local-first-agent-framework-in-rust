use abcb_core::{MockProvider, one_turn};
#[cfg(test)]
use clap::CommandFactory;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Parser)]
#[command(name = "abcb")]
#[command(version)]
#[command(about = "AI agent framework for Godot game development")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Check the local abcb development environment.
    Doctor,
    /// Send a single user message and print the assistant reply.
    Chat {
        /// The user message to send.
        message: String,
        /// Use the in-process mock provider (required while no real provider exists).
        #[arg(long)]
        mock: bool,
    },
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Config {
    project: Option<ProjectConfig>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct ProjectConfig {
    name: Option<String>,
}

impl Config {
    fn project_name(&self) -> Option<&str> {
        self.project.as_ref()?.name.as_deref()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => run_doctor()?,
        Command::Chat { message, mock } => run_chat(message, mock)?,
    }

    Ok(())
}

fn run_chat(message: String, mock: bool) -> Result<(), Box<dyn Error>> {
    if !mock {
        return Err(
            "only --mock is supported right now; pass --mock to use the mock provider".into(),
        );
    }

    let mut provider = MockProvider::new([format!("mock: you said {message}")]);
    let reply = one_turn(&mut provider, &message)?;
    println!("{}", reply.content);

    Ok(())
}

fn run_doctor() -> Result<(), Box<dyn Error>> {
    println!("abcb doctor");
    println!("workspace: ok");

    match load_config(Path::new("abcb.toml"))? {
        Some(config) => {
            println!("config: found abcb.toml");
            if let Some(name) = config.project_name() {
                println!("project: {name}");
            }
        }
        None => println!("config: abcb.toml not found (ok for now)"),
    }

    Ok(())
}

fn load_config(path: &Path) -> Result<Option<Config>, Box<dyn Error>> {
    if !path.exists() {
        return Ok(None);
    }

    let source = fs::read_to_string(path)?;
    let config = parse_config(&source)?;

    Ok(Some(config))
}

fn parse_config(source: &str) -> Result<Config, toml::de::Error> {
    toml::from_str(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_name_is_abcb() {
        let command = Cli::command();

        assert_eq!(command.get_name(), "abcb");
    }

    #[test]
    fn parses_project_name_from_config() {
        let config = parse_config(
            r#"
            [project]
            name = "abcb"
            "#,
        )
        .expect("config should parse");

        assert_eq!(config.project_name(), Some("abcb"));
    }

    #[test]
    fn parses_empty_config() {
        let config = parse_config("").expect("empty config should parse");

        assert_eq!(config.project_name(), None);
    }

    #[test]
    fn chat_subcommand_parses_with_mock_flag() {
        let cli = Cli::try_parse_from(["abcb", "chat", "hi", "--mock"])
            .expect("chat --mock should parse");

        match cli.command {
            Command::Chat { message, mock } => {
                assert_eq!(message, "hi");
                assert!(mock);
            }
            other => panic!("expected Chat, got {other:?}"),
        }
    }

    #[test]
    fn run_chat_without_mock_flag_errors() {
        let err = run_chat("hi".into(), false).expect_err("should require --mock");

        assert!(err.to_string().contains("--mock"));
    }
}
