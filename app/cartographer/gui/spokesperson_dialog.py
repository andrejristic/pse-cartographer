"""
First-run spokesperson picker (C1).

On first launch the app nudges the operator to choose a working model. It names plainly
what is available WITHOUT one (raw measurements only) and what is lost (chat, the
plain-language report, the short expert notes on each result, the active cartographer).
Choosing is not forced — "Defer" works — but the dialog pushes toward choosing.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout,
)

from ..spokesperson import CHOICES, make_backend

_NUDGE = """<h3>Choose a working model (spokesperson)</h3>
<p>The spokesperson drives the conversation and writes the plain-language report.
Without one you can still use the app, but <b>only for raw measurements</b>:</p>
<ul>
  <li>✅ open a model, Model Card + capability semaphore</li>
  <li>✅ L1 spectral map + ESD histograms</li>
  <li>✅ (with torch) the Cartographer's numbers &amp; charts — basis, per-prompt J-space,
      drift/diff, ignition</li>
</ul>
<p style='color:#a11'>Without a spokesperson you will NOT get: the conversation,
the plain-language ("narodski") report, the short expert note on each result, or the
active cartographer. Those are what make the results readable.</p>
<p>Pick one now (recommended), or defer and set it later from the menu.</p>"""


class SpokespersonDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSE Cartographer — spokesperson")
        self.setModal(True)
        self.resize(560, 460)
        self.backend = None            # result: a backend, or None if deferred

        lay = QVBoxLayout(self)
        nudge = QLabel(_NUDGE)
        nudge.setWordWrap(True)
        nudge.setTextFormat(Qt.RichText)
        lay.addWidget(nudge)

        form = QFormLayout()
        self.choice = QComboBox()
        for key, label, _env in CHOICES:
            self.choice.addItem(label, key)
        self.choice.currentIndexChanged.connect(self._on_choice)
        form.addRow("Backend:", self.choice)

        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("API key (leave blank to read from environment)")
        form.addRow("API key:", self.key)

        self.model = QLineEdit()
        self.model.setPlaceholderText("model id (optional — sensible default used)")
        form.addRow("Model:", self.model)

        self.path = QLineEdit()
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("local GGUF path (for the local option)")
        self.browse = QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        form.addRow(self.path, self.browse)
        lay.addLayout(form)

        buttons = QDialogButtonBox()
        self.use_btn = buttons.addButton("Use this spokesperson",
                                         QDialogButtonBox.AcceptRole)
        self.defer_btn = buttons.addButton("Defer (measurements only)",
                                           QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._on_choice()

    def _current_key(self) -> str:
        return self.choice.currentData()

    def _on_choice(self):
        is_local = self._current_key() == "local"
        self.key.setEnabled(not is_local)
        self.path.setEnabled(is_local)
        self.browse.setEnabled(is_local)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Local GGUF spokesperson", "",
                                           "GGUF (*.gguf)")
        if p:
            self.path.setText(p)

    def _accept(self):
        choice = self._current_key()
        try:
            self.backend = make_backend(
                choice,
                api_key=(self.key.text().strip() or None),
                model=(self.model.text().strip() or None),
                model_path=(self.path.text().strip() or None),
            )
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Spokesperson", f"{type(e).__name__}: {e}")
            return
        self.accept()
