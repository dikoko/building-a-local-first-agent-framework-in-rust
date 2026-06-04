# Building a Local-First Agent Framework in Rust: Sample Code

This repository contains the public sample code for the book/blog series
*Building a Local-First Agent Framework in Rust*.

Each chapter has its own folder. The code is intentionally duplicated by chapter
so each folder can be read and run as the state of the project at that point in
the series.

## Chapters

- `chapter01/`: Introduction. No standalone sample code yet.
- `chapter02/`: Setting up the Rust workspace and first CLI.
- `chapter03/`: Modeling messages, roles, and sessions in `abcb-core`.

## Running a Chapter Sample

```sh
cd chapter03/abcb
cargo run -- doctor
./scripts/check.sh
```
