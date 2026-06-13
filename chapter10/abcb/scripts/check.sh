#!/usr/bin/env bash
set -euo pipefail

cargo test --workspace
cargo fmt --check
cargo clippy --workspace -- -D warnings
