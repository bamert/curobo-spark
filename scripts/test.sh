#!/usr/bin/env bash
# Validate cuRobo on this machine.
cd "$(dirname "$0")/.."
uv run pytest tests -v "$@"
