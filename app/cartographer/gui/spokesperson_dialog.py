"""
First-run spokesperson picker (C1).

On first launch the app nudges the operator to choose a working model. It names plainly what
is available WITHOUT one (raw measurements only) and what is lost (chat, the plain-language
report, the short expert notes, Findings & Conclusions, the active cartographer). Choosing is
not forced — "Defer" works — but the dialog pushes toward choosing. The choice (and, unless
unticked, the key) is remembered in ~/.config/pse-cartographer/llm.json with owner-only rights.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout)

from pse_a.shell import llm

from ..spokesperson import Spokesperson, provider_choices

_NUDGE = """<h3>Choose a working model (spokesperson)</h3>
<p>The spokesperson drives the conversation and writes the plain-language reports.
Without one you can still use the app, but <b>only for raw measurements</b>:</p>
<ul>
  <li>✅ open a model, Model Card + capability semaphore</li>
  <li>✅ spectral map, precision comparison, thought waterfall, concept dictionary, J-space</li>
  <li>✅ the Cartographer's numbers &amp; charts — basis, per-prompt J-space, drift/diff, ignition</li>
</ul>
<p style='color:#a11'>Without a spokesperson you will NOT get: the conversation, the plain-language
("narodski") report, the short expert note on each result, Findings &amp; Conclusions, or the
active cartographer. Those are what make the results readable.</p>
<p>Only summaries of sealed results are sent to an online spokesperson — never the model, its
weights or activations. A local GGUF spokesperson runs offline on this computer (slow on a CPU;
up to ~3B is practical).</p>"""


class SpokespersonDialog(QDialog):
    def __init__(self, parent=None, current: Spokesperson | None = None):
        super().__init__(parent)
        self.setWindowTitle("PSE Cartographer — spokesperson")
        self.setModal(True)
        self.resize(620, 560)
        self.backend = None
        self.remember = True
        cfg = llm.load_config()
        self._cfg = cfg

        lay = QVBoxLayout(self)
        nudge = QLabel(_NUDGE)
        nudge.setWordWrap(True)
        nudge.setTextFormat(Qt.RichText)
        lay.addWidget(nudge)

        form = QFormLayout()
        self.choice = QComboBox()
        for pid, label in provider_choices():
            self.choice.addItem(label, pid)
        form.addRow("Provider:", self.choice)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        form.addRow("API key:", self.key)
        self.where = QLabel()
        self.where.setOpenExternalLinks(True)
        form.addRow("", self.where)
        self.base = QLineEdit()
        form.addRow("Server address:", self.base)
        mrow = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumWidth(280)
        fetch = QPushButton("Fetch model list")
        fetch.clicked.connect(self._fetch)
        mrow.addWidget(self.model, 1)
        mrow.addWidget(fetch)
        form.addRow("Model:", mrow)
        grow = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("local instruct/chat GGUF")
        self.browse = QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        grow.addWidget(self.path, 1)
        grow.addWidget(self.browse)
        form.addRow("GGUF:", grow)
        self.remember_chk = QCheckBox("Remember the key on this computer (owner-only file)")
        self.remember_chk.setChecked(cfg.get("remember_keys", True))
        form.addRow("", self.remember_chk)
        lay.addLayout(form)

        buttons = QDialogButtonBox()
        self.test_btn = buttons.addButton("Test", QDialogButtonBox.ActionRole)
        self.test_btn.clicked.connect(self._test)
        buttons.addButton("Use this spokesperson", QDialogButtonBox.AcceptRole)
        buttons.addButton("Defer (measurements only)", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.choice.currentIndexChanged.connect(self._on_choice)
        if current is not None:
            i = self.choice.findData(current.pid)
            if i >= 0:
                self.choice.setCurrentIndex(i)
        self._on_choice()

    def _pid(self) -> str:
        return self.choice.currentData()

    def _on_choice(self):
        pid = self._pid()
        _, grp, kind, base, env, url = llm.PROVIDERS[pid]
        e = llm.entry(self._cfg, pid)
        is_local = kind == "local_gguf"
        self.key.setEnabled(bool(env) or pid == "custom")
        self.key.setText(e.get("key", ""))
        self.key.setPlaceholderText(f"or set {env} in the environment" if env else "not needed")
        self.where.setText(f"<a href='{url}'>where to get a key</a>" if url and env else "")
        self.base.setEnabled(kind == "openai_compat")
        self.base.setText(e.get("base_url", base))
        self.model.setEnabled(not is_local)
        self.model.clear()
        if kind == "anthropic":
            self.model.addItem(llm.CLAUDE_DEFAULT)
        if e.get("model"):
            self.model.setCurrentText(e["model"])
        self.path.setEnabled(is_local)
        self.browse.setEnabled(is_local)
        self.path.setText(e.get("gguf", ""))

    def _entry(self) -> dict:
        pid = self._pid()
        kind = llm.PROVIDERS[pid][2]
        e = {"key": self.key.text().strip()}
        if kind == "openai_compat":
            e["base_url"] = self.base.text().strip()
        if kind != "local_gguf":
            e["model"] = self.model.currentText().strip()
        else:
            e["gguf"] = self.path.text().strip()
        env = llm.PROVIDERS[pid][4]
        if not e["key"] and env:
            e["key"] = __import__("os").environ.get(env, "")
        return e

    def _fetch(self):
        try:
            names = llm.list_models(self._pid(), self._entry())
        except Exception as ex:                               # noqa: BLE001
            QMessageBox.warning(self, "Model list", str(ex))
            return
        cur = self.model.currentText()
        self.model.clear()
        self.model.addItems(names)
        if cur in names:
            self.model.setCurrentText(cur)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Local GGUF spokesperson", "", "GGUF (*.gguf)")
        if p:
            self.path.setText(p)

    def _test(self):
        sp = Spokesperson(self._pid(), self._entry())
        if not sp.available():
            QMessageBox.information(self, "Test", "Fill in the key / model / path first.")
            return
        try:
            txt = sp.complete("Answer in one short sentence.",
                              [{"role": "user", "content": "Say that you are ready."}], 60)
            QMessageBox.information(self, "Test", f"{sp.name}:\n\n{txt.strip()[:400]}")
        except Exception as ex:                               # noqa: BLE001
            QMessageBox.warning(self, "Test", str(ex))

    def _accept(self):
        sp = Spokesperson(self._pid(), self._entry())
        if not sp.available():
            QMessageBox.warning(self, "Spokesperson", "This choice is not complete yet "
                                                      "(key, model, server address or GGUF path).")
            return
        self.backend = sp
        self.remember = self.remember_chk.isChecked()
        self.accept()
