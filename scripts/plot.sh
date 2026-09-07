#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
exec uv run --no-sync python -m looped_transformer_comparison.cli plot "$@"
