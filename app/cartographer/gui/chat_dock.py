"""The conversation — "the spine" ([SPEC] 10.1): talk with the spokesperson about the sealed
results. The model only ever sees summaries of sealed results (never weights or
activations); every answer passes the AZ-8 scrub, also while it is still being typed."""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from pse_a.shell import psel

from .common import esc, html_view


class _Talk(QObject):
    part = Signal(str)
    done = Signal(str)

    def __init__(self, lib, sp, text):
        super().__init__()
        self.lib, self.sp, self.text = lib, sp, text

    def run(self):
        try:
            final = ""
            for kind, txt in psel.respond_stream(self.lib, self.sp.pid, self.sp.entry, self.text):
                if kind == "partial":
                    self.part.emit(txt)
                else:
                    final = txt
            self.done.emit(final)
        except Exception as e:                                 # noqa: BLE001
            self.done.emit(f"Could not get an answer: {e}")


class ChatPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._t = None
        lay = QVBoxLayout(self)
        self.view = html_view()
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask about the results…")
        self.input.returnPressed.connect(self._send)
        b = QPushButton("Send")
        b.clicked.connect(self._send)
        row.addWidget(self.input, 1)
        row.addWidget(b)
        lay.addLayout(row)
        self._log = []
        self.reload()

    def reload(self):
        self._log = [(m["role"], m["text"]) for m in psel.history(self.main.lib, 40)]
        self._render()

    def narrate(self, html: str):
        self._log.append(("system", html))
        self._render()

    def _render(self, partial: str | None = None):
        out = []
        for role, txt in self._log:
            if role == "system":
                out.append(txt)
            elif role == "user":
                out.append(f"<p style='color:#246'><b>You:</b> {esc(txt)}</p>")
            else:
                out.append(f"<p><b>Spokesperson:</b> {esc(txt).replace(chr(10), '<br>')}</p>")
        if partial is not None:
            out.append(f"<p><b>Spokesperson:</b> {esc(partial).replace(chr(10), '<br>')} …</p>")
        self.view.setHtml("".join(out))
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

    def _send(self):
        text = self.input.text().strip()
        if not text or self._t is not None:
            return
        sp = self.main.voice.backend
        if not (sp and sp.available()):
            self.narrate("<p style='color:#a97'>No spokesperson is set — choose one via "
                         "<b>Cartographer → Choose spokesperson…</b></p>")
            return
        self.input.clear()
        self._log.append(("user", text))
        self._render()
        t = QThread()
        w = _Talk(self.main.lib, sp, text)
        w.moveToThread(t)
        t.started.connect(w.run)
        w.part.connect(self._partial)
        w.done.connect(self._finished)
        self._t = (t, w)
        t.start()

    def _partial(self, s):                              # GUI thread (slot of a QObject)
        self._render(s)

    def _finished(self, s):
        self._log.append(("assistant", s))
        self._render()
        t, _ = self._t
        t.quit()
        t.wait()
        self._t = None
