#!/bin/sh
# Launch PSE Cartographer from source (after ./setup.sh).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
py="$here/.venv/bin/python"
[ -x "$py" ] || { echo "run ./setup.sh first" >&2; exit 1; }
cd "$here"
exec "$py" main.py "$@"
