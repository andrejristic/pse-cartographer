#!/usr/bin/env bash
# Build PSECartographer-Setup-<version>.exe (PyInstaller bundle + Inno Setup), on Windows.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$HERE")"; REPO="$(dirname "$APP")"
PY="${PYTHON:-$APP/.venv/Scripts/python.exe}"
VERSION="${1:-$("$PY" -c 'import sys; sys.path.insert(0, r"'"$(cygpath -w "$APP")"'"); import cartographer; print(cartographer.__version__)')}"
PKG="pse-cartographer"
STAGE="$HERE/stage-win"; OUT="$REPO/dist"
ISCC="${ISCC:-/c/Program Files (x86)/Inno Setup 6/ISCC.exe}"
echo "== PyInstaller bundle ($VERSION)"
rm -rf "$APP/build/pyi" "$STAGE"
cd "$APP"                      # pse_a / cartographer must be importable for --collect-*
export PYTHONPATH="$(cygpath -w "$APP")"
"$PY" -m PyInstaller --noconfirm --clean --onedir --windowed --name "$PKG" \
  --icon "$HERE/$PKG.ico" \
  --distpath "$APP/build/pyi/dist" --workpath "$APP/build/pyi/work" --specpath "$APP/build/pyi" \
  --paths "$APP" \
  --collect-submodules pse_a --collect-data pse_a \
  --collect-submodules cartographer \
  --collect-submodules anthropic --collect-all tokenizers \
  --collect-all matplotlib \
  --hidden-import scipy.optimize --hidden-import scipy.stats \
  --exclude-module torch --exclude-module tensorflow --exclude-module PySide6.QtWebEngineCore \
  "$APP/main.py"
APPDIR="$STAGE/$PKG"; mkdir -p "$APPDIR"
cp -a "$APP/build/pyi/dist/$PKG/." "$APPDIR/"
cp "$HERE/$PKG.ico" "$APPDIR/"; cp "$REPO/LICENSE" "$APPDIR/LICENSE.txt"; cp "$REPO/NOTICE" "$APPDIR/NOTICE.txt"
cp "$REPO/GUIDE-EN.md" "$REPO/VODIC-SR.md" "$APPDIR/"
echo "== Inno Setup"; mkdir -p "$OUT"
MSYS2_ARG_CONV_EXCL="*" "$ISCC" "/DAppVersion=$VERSION" "/DSourceDir=$(cygpath -w "$APPDIR")" "/DOutDir=$(cygpath -w "$OUT")" \
  "$(cygpath -w "$HERE/$PKG.iss")"
echo "built $OUT/PSECartographer-Setup-$VERSION.exe"
