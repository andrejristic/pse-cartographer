#!/usr/bin/env bash
# Build pse-cartographer_<version>_amd64.deb — a self-contained PyInstaller bundle.
# No Python, pip or network is needed on the target machine; no PyTorch inside.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$HERE")"; REPO="$(dirname "$APP")"
PY="${PYTHON:-$APP/.venv/bin/python}"; PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"   # absolute, symlink kept (venv)
VERSION="${1:-$("$PY" -c 'import sys; sys.path.insert(0, "'"$APP"'"); import cartographer; print(cartographer.__version__)')}"
ARCH="${ARCH:-amd64}"; PKG="pse-cartographer"
STAGE="$HERE/stage"; OUT="$REPO/dist"

echo "== PyInstaller bundle ($VERSION)"
rm -rf "$APP/build/pyi" "$STAGE"
cd "$APP"                      # pse_a / cartographer must be importable for --collect-*
export PYTHONPATH="$APP${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m PyInstaller --noconfirm --clean --onedir --name "$PKG" \
  --distpath "$APP/build/pyi/dist" --workpath "$APP/build/pyi/work" --specpath "$APP/build/pyi" \
  --paths "$APP" \
  --collect-submodules pse_a --collect-data pse_a \
  --collect-submodules cartographer \
  --collect-submodules anthropic --collect-all tokenizers \
  --collect-all matplotlib \
  --hidden-import scipy.optimize --hidden-import scipy.stats \
  --exclude-module torch --exclude-module tensorflow --exclude-module PySide6.QtWebEngineCore \
  "$APP/main.py"

APPDIR="$STAGE/opt/$PKG"; mkdir -p "$APPDIR"
cp -a "$APP/build/pyi/dist/$PKG/." "$APPDIR/"
cp "$REPO/LICENSE" "$APPDIR/LICENSE"; cp "$REPO/NOTICE" "$APPDIR/NOTICE"
cp "$REPO/GUIDE-EN.md" "$REPO/VODIC-SR.md" "$APPDIR/"

mkdir -p "$STAGE/usr/bin" "$STAGE/usr/share/applications" \
         "$STAGE/usr/share/icons/hicolor/256x256/apps" "$STAGE/usr/share/doc/$PKG" "$STAGE/DEBIAN"
ln -s "/opt/$PKG/$PKG" "$STAGE/usr/bin/$PKG"
cp "$HERE/$PKG.png" "$STAGE/usr/share/icons/hicolor/256x256/apps/$PKG.png"
cp "$REPO/LICENSE" "$REPO/NOTICE" "$STAGE/usr/share/doc/$PKG/"
cat > "$STAGE/usr/share/applications/$PKG.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=PSE Cartographer
GenericName=Local model J-space cartography
Comment=Open a local GGUF, map its J-space, watch it move as you feed prompts
Exec=$PKG
Icon=$PKG
Terminal=false
Categories=Development;Science;Education;
Keywords=LLM;GGUF;J-space;interpretability;PSE;cartography;
DESK
SIZE="$(du -sk "$STAGE/opt" "$STAGE/usr" | awk '{s+=$1} END{print s}')"
cat > "$STAGE/DEBIAN/control" <<CTL
Package: $PKG
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Installed-Size: $SIZE
Maintainer: Andrej Ristic <andrej.s.ristic@gmail.com>
Depends: libc6 (>= 2.34), libgl1, libegl1, libfontconfig1, libxkbcommon0, libdbus-1-3
Homepage: https://github.com/andrejristic/pse-cartographer
Description: PSE Cartographer - live J-space cartography of a local GGUF model
 Open a local GGUF language model, map which concepts it holds "in mind" at
 every layer (J-space), feed it prompts and watch that move; plus the full
 PSE-A toolset (spectral anatomy, precision comparison, thought waterfall,
 concept dictionary, perplexity ladder, sealed dossier, swarm and inbox) and
 Findings & Conclusions written by a spokesperson model of your choice.
 Runs on the CPU with its own numpy runtime (no PyTorch, no GPU needed).
CTL
echo "== dpkg-deb"; mkdir -p "$OUT"
DEB="$OUT/${PKG}_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group -Zxz --build "$STAGE" "$DEB"; echo "built $DEB"
