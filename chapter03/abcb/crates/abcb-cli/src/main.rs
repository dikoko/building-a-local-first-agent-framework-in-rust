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
    }

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
}
