#!/usr/bin/env bash
# Build a .deb for PSE Cartographer.
#
# Cartographer reuses the PSE-A engine (`pse_a`) from the sibling "PSE J-space/app" folder.
# For a self-contained package we bundle BOTH: the Cartographer app under
# /opt/pse-cartographer/app and the engine under /opt/pse-cartographer/engine, and the
# launcher points PSE_ENGINE at the bundled engine.
#
# Run on the target machine (dpkg-deb + python3-venv present). Needs network at install
# time for the first-run pip install. Torch/transformers are NOT bundled — install them
# from the app's Patient tab when you want to dissect (huge, CPU/GPU-specific).
set -euo pipefail

VERSION="0.1.0"; ARCH="all"; PKG="pse-cartographer"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"                 # PSE Cartographer/
ENGINE="$(cd "$ROOT/../PSE J-space/app" && pwd)"   # shared engine (contains pse_a/)
STAGE="$HERE/build/${PKG}_${VERSION}_${ARCH}"

echo "==> engine at: $ENGINE"
[ -d "$ENGINE/pse_a" ] || { echo "ERROR: engine pse_a not found next to Cartographer"; exit 1; }

rm -rf "$HERE/build"
mkdir -p "$STAGE/DEBIAN" "$STAGE/opt/${PKG}" "$STAGE/usr/bin" \
         "$STAGE/usr/share/applications" "$STAGE/usr/share/icons/hicolor/256x256/apps"

echo "==> staging app + bundled engine"
cp -r "$ROOT/app" "$STAGE/opt/${PKG}/app"
mkdir -p "$STAGE/opt/${PKG}/engine"
cp -r "$ENGINE/pse_a" "$STAGE/opt/${PKG}/engine/pse_a"
find "$STAGE/opt/${PKG}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

cat > "$STAGE/DEBIAN/control" <<EOF
Package: ${PKG}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Depends: python3 (>= 3.10), python3-venv, python3-pip, libgl1
Maintainer: Andrej Ristic <andrej.s.ristic@gmail.com>
Description: PSE Cartographer — live J-space cartography of a local GGUF
 Enter a local GGUF model, estimate its J-space basis once, feed it prompts and watch the
 J-space move; produces a plain-language report. Built on the PSE-A engine (bundled).
EOF

cat > "$STAGE/usr/bin/${PKG}" <<'EOF'
#!/usr/bin/env bash
APPDIR="/opt/pse-cartographer/app"
VENV="/opt/pse-cartographer/venv"
export PSE_ENGINE="/opt/pse-cartographer/engine"   # bundled shared engine
if [ ! -x "$VENV/bin/python" ]; then
    echo "PSE Cartographer: environment missing. Reinstall the package." >&2; exit 1
fi
cd "$APPDIR"; exec "$VENV/bin/python" "$APPDIR/main.py" "$@"
EOF
chmod 0755 "$STAGE/usr/bin/${PKG}"

cp "$HERE/pse-cartographer.desktop" "$STAGE/usr/share/applications/${PKG}.desktop"
[ -f "$HERE/pse-cartographer.png" ] && \
  cp "$HERE/pse-cartographer.png" "$STAGE/usr/share/icons/hicolor/256x256/apps/${PKG}.png"

cat > "$STAGE/DEBIAN/postinst" <<'EOF'
#!/usr/bin/env bash
set -e
APPROOT="/opt/pse-cartographer"; VENV="$APPROOT/venv"
echo "PSE Cartographer: creating environment…"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install -r "$APPROOT/app/requirements.txt"
update-desktop-database >/dev/null 2>&1 || true
echo "Installed. Launch: pse-cartographer   (torch for dissection installs from the app)"
EOF
chmod 0755 "$STAGE/DEBIAN/postinst"

cat > "$STAGE/DEBIAN/prerm" <<'EOF'
#!/usr/bin/env bash
set -e
rm -rf /opt/pse-cartographer/venv
EOF
chmod 0755 "$STAGE/DEBIAN/prerm"

dpkg-deb --root-owner-group --build "$STAGE"
OUT="$HERE/${PKG}_${VERSION}_${ARCH}.deb"; mv "$STAGE.deb" "$OUT"
echo "==> DONE: $OUT"
echo "Install:  sudo apt install $OUT"
