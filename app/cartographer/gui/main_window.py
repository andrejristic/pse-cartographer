"""
Cartographer main window (C3).

Subclasses the shared engine's analyzer window — so inside Cartographer you can do
EVERYTHING the existing app does (open a model, Model Card, L1 spectral map, ESD,
Waterfall, Features, J-space, Dossier, Swarm, chat) — and adds:
  - the first-run spokesperson picker (C1), which nudges toward choosing,
  - a dedicated "Cartographer" tab group whose first tab asks for the patient GGUF path (C4),
  - a spokesperson menu that includes Gemini (C2).

Nothing in the engine is modified; this only extends it.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMessageBox

from pse_a.gui.main_window import MainWindow          # the shared analyzer window

from ..spokesperson import make_backend
from .cartographer_tabs import CartographerWidget
from .spokesperson_dialog import SpokespersonDialog


class CartographerWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSE Cartographer")

        # add the dedicated cartography flow as its own top tab
        self.cartographer = CartographerWidget(self)
        self.tabs.addTab(self.cartographer, "★ Cartographer")

        # a spokesperson menu that includes Gemini (the engine's own menu omits it)
        m = self.menuBar().addMenu("&Cartographer")
        act = QAction("Choose spokesperson…", self)
        act.triggered.connect(self._choose_spokesperson)
        m.addAction(act)

        self._first_run_done = False
        QTimer.singleShot(0, self._first_run)          # pop picker after the window shows

    def _first_run(self):
        if self._first_run_done:
            return
        self._first_run_done = True
        dlg = SpokespersonDialog(self)
        dlg.exec()
        if dlg.backend is not None:
            self.psel.backend = dlg.backend
            self.chat.narrate(f"<p style='color:#888'>Spokesperson: "
                              f"<b>{dlg.backend.name}</b> "
                              f"({'ready' if dlg.backend.available() else 'set a key/path'}).</p>")
        else:
            self.chat.narrate("<p style='color:#a97'>No spokesperson chosen — measurements "
                              "only. Choose one anytime via <b>Cartographer → Choose "
                              "spokesperson…</b> for the conversation and the report.</p>")

    def _choose_spokesperson(self):
        dlg = SpokespersonDialog(self)
        dlg.exec()
        if dlg.backend is not None:
            self.psel.backend = dlg.backend
            self.chat.narrate(f"<p style='color:#888'>Spokesperson set to "
                              f"<b>{dlg.backend.name}</b>.</p>")
