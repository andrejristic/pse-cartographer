#!/bin/sh
# Build the virtual environment for running PSE Cartographer from source (app/.venv).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
"$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' || {
    echo "need Python 3.10 or newer; found: $("$py" -V 2>&1)" >&2; exit 1; }
[ -d "$here/.venv" ] || "$py" -m venv "$here/.venv"
"$here/.venv/bin/pip" install --upgrade pip >/dev/null
"$here/.venv/bin/pip" install -r "$here/requirements.txt"
"$here/.venv/bin/python" "$here/main.py" --selftest
echo "ready: $here/run.sh"
