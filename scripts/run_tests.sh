#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
MEMORY_PYTHON="${MEMORY_PYTHON:-python3}"
if [ -x .venv/bin/python ]; then MEMORY_PYTHON=.venv/bin/python; fi
if [ "${MEMORY_TEST_INSTALLED:-0}" = 1 ]; then
  exec "$MEMORY_PYTHON" -I -B -m pytest "$@"
fi
exec "$MEMORY_PYTHON" -B -m pytest "$@"
