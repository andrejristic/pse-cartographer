#!/usr/bin/env python3
"""
PSE Cartographer — entry point.

One desktop application that carries the PSE-A engine (`pse_a`, bundled, numpy runtime — no
PyTorch) and adds live J-space cartography of a local GGUF patient: enter the model, estimate
its J-space basis once, feed it prompts, and watch the J-space move — with a plain-language
("narodski") report at the end, Findings & Conclusions over everything measured, and the full
PSE-A toolset (spectral anatomy, precision comparison, thought waterfall, concept dictionary,
J-space map, perplexity ladder, dossier, swarm & inbox).

The working model (spokesperson) is chosen on first launch: Claude / Gemini / Mistral / Groq
and others via an API key, a local server, or a local GGUF. The patient is ALWAYS a local GGUF.

    pse-cartographer               start the GUI
    pse-cartographer --selftest    check the installation (no network, no models, no display)
    pse-cartographer --cli ...     the engine's headless commands (doctor, l1, jspace, ...)
"""
import os
import sys


def _tune_threads():
    """BLAS on physical cores: hyperthreads slow GEMM down (measured on an i3-2330M: the J-space
    backward takes 11.5 s on 2 threads vs 16.3 s on 4). Must run before numpy is imported;
    a value the user set is never overridden."""
    if any(os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        return
    try:
        from pse_a.shell.doctor import _cpu
        _, physical, logical, _ = _cpu()
        if physical == logical and logical >= 4 and sys.platform == "win32":
            physical = logical // 2          # no /proc/cpuinfo: assume 2 threads per core
        n = max(1, physical or 1)
    except Exception:
        return
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[k] = str(n)


_tune_threads()


def selftest() -> int:
    from pse_a.tests import selftest as st
    rc = st.main(None)
    # GUI smoke test without a display: build the window over a temporary data folder
    import tempfile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        os.environ["PSE_CARTOGRAPHER_DATA"] = d
        os.environ["XDG_CONFIG_HOME"] = d
        try:
            from PySide6.QtWidgets import QApplication
            from cartographer.gui.main_window import CartographerWindow
            app = QApplication.instance() or QApplication(sys.argv[:1])
            w = CartographerWindow(first_run=False)
            ok = w.tabs.count() == 11
            w.close()
            del app
            print(f"  {'GUI builds (offscreen)':34} {'OK' if ok else 'FAIL'}")
            rc = rc or (0 if ok else 1)
        except Exception as e:                                # noqa: BLE001
            print(f"  {'GUI builds (offscreen)':34} ERROR  {type(e).__name__}: {e}")
            rc = 1
    return rc


def main():
    if "--version" in sys.argv:
        from cartographer import __version__
        print(__version__)
        return 0
    if "--selftest" in sys.argv:
        return selftest()
    if "--cli" in sys.argv:
        from pse_a.cli import main as cli_main
        i = sys.argv.index("--cli")
        return cli_main(sys.argv[i + 1:])

    from PySide6.QtWidgets import QApplication
    from cartographer.gui.main_window import CartographerWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PSE Cartographer")
    app.setOrganizationName("ZeravaTech")
    win = CartographerWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
