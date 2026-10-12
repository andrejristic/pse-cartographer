#!/usr/bin/env python3
"""
PSE Cartographer — entry point.

A separate application that reuses the PSE-A engine (`pse_a`) as a library and adds live
J-space cartography of a local GGUF patient: enter the model, estimate its J-space basis
once, feed it prompts, and watch the J-space move — with a plain-language ("narodski")
report at the end.

The working model (spokesperson) is chosen on first launch: Claude / Gemini / Mistral /
Groq via API key, or a local GGUF via a path. The patient is ALWAYS a local GGUF.
"""
import sys


def main():
    # C0: make the shared engine importable before anything else.
    from cartographer.engine_path import ensure_engine_on_path
    ensure_engine_on_path()

    from PySide6.QtWidgets import QApplication
    from cartographer.gui.main_window import CartographerWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PSE Cartographer")
    app.setOrganizationName("ZeravaTech")
    win = CartographerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
