#!/usr/bin/env bash
# Interactive cuRobo viewer on port 8080 (or $1). Stop with Ctrl+C.
cd "$(dirname "$0")/.."
uv run python -m curobo.examples.getting_started.motion_planning --visualize --port "${1:-8080}"
