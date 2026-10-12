"""Dossier ([SPEC] Sec 11): the archive of sealed runs per model, in two registers.

Mathematical register: the artifacts in full. Human register: the engine's deterministic
story, each sentence tagged with its artifact. Notes are append-only, attributed and anchored
to an artifact hash (AZ-6): there is no edit or delete — a correction is a new run. A comparison
view puts two dossiers side by side.
"""
from __future__ import annotations

import getpass
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from pse_a.shell import narrative
from pse_a.shell.prognostics import run_kind_label

from .common import esc, html_view, jdump, when


class DossierTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.mh = None
        lay = QVBoxLayout(self)
        sp = QSplitter(Qt.Horizontal)
        self.runs = QListWidget()
        self.runs.currentItemChanged.connect(self._show)
        sp.addWidget(self.runs)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.reg = QTabWidget()
        self.human = html_view()
        self.math = html_view()
        self.reg.addTab(self.human, "Human register")
        self.reg.addTab(self.math, "Mathematical register")
        rl.addWidget(self.reg, 1)
        nrow = QHBoxLayout()
        self.anchor = QComboBox()
        self.note = QLineEdit()
        self.note.setPlaceholderText("add a note to this sealed run (append-only, cannot be edited)")
        add = QPushButton("Add note")
        add.clicked.connect(self._add_note)
        nrow.addWidget(QLabel("on"))
        nrow.addWidget(self.anchor)
        nrow.addWidget(self.note, 1)
        nrow.addWidget(add)
        rl.addLayout(nrow)
        sp.addWidget(right)
        sp.setSizes([320, 800])
        lay.addWidget(sp, 2)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Compare with:"))
        self.other = QComboBox()
        crow.addWidget(self.other, 1)
        cb = QPushButton("Side by side")
        cb.clicked.connect(self._compare)
        crow.addWidget(cb)
        lay.addLayout(crow)
        self.cmp = html_view()
        lay.addWidget(self.cmp, 1)

    def refresh(self, model_hash):
        self.mh = model_hash
        self.runs.clear()
        if model_hash:
            for r in self.main.lib.dossier(model_hash)[::-1]:
                it = QListWidgetItem(f"{run_kind_label(r['kind'])}\n{when(r['sealed'])} · {r['run_id'][:8]}"
                                     + (f" · {len(r['notes'])} note(s)" if r["notes"] else ""))
                it.setData(Qt.UserRole, r)
                self.runs.addItem(it)
        self.other.clear()
        for m in self.main.lib.models():
            if m["model_hash"] != model_hash:
                self.other.addItem(m["display_name"], m["model_hash"])
        if self.runs.count():
            self.runs.setCurrentRow(0)
        else:
            self.human.setHtml("<p>No sealed runs for this model yet.</p>")
            self.math.clear()

    def _show(self, it, _prev=None):
        if it is None:
            return
        r = it.data(Qt.UserRole)
        lib = self.main.lib
        load = lambda h: json.loads(lib.get_blob(h))
        story = narrative.story(r["kind"], r["artifacts"], load)
        notes = lib.db.execute("SELECT author, text, created, anchor_hash FROM notes WHERE run_id=? "
                               "ORDER BY created", (r["run_id"],)).fetchall()
        self.human.setHtml(
            f"<h3>{esc(run_kind_label(r['kind']))}</h3><p style='color:#888'>run {r['run_id']} · sealed "
            f"{when(r['sealed'])} · config {r['config_hash'][:12]} · findings in knowledge: "
            f"{r['n_registry']}" + (f" · supersedes {r['supersedes']}" if r["supersedes"] else "") + "</p>"
            + "".join(f"<p>{esc(s)}</p>" for s in story)
            + ("<h4>Notes</h4>" + "".join(f"<p><b>{esc(a)}</b> · {when(t)} · on [{h[:8]}]<br>{esc(x)}</p>"
                                          for a, x, t, h in notes) if notes else ""))
        parts = []
        for name, h in r["artifacts"].items():
            raw = lib.get_blob(h)
            try:
                txt = jdump(json.loads(raw))
            except Exception:
                txt = f"(binary, {len(raw)} bytes)" if not name.endswith(".md") else raw.decode("utf-8")
            if len(txt) > 60000:
                txt = txt[:60000] + "\n… (truncated in view; export it from Findings)"
            parts.append(f"<h4>{esc(name)} [{h[:8]}]</h4><pre>{esc(txt)}</pre>")
        self.math.setHtml("".join(parts))
        self.anchor.clear()
        for name, h in r["artifacts"].items():
            self.anchor.addItem(f"{name} [{h[:8]}]", (r["run_id"], h))

    def _add_note(self):
        txt = self.note.text().strip()
        data = self.anchor.currentData()
        if not txt or not data:
            return
        run_id, h = data
        self.main.lib.add_note(run_id, getpass.getuser(), h, txt)
        self.note.clear()
        row = self.runs.currentRow()
        self.refresh(self.mh)
        self.runs.setCurrentRow(row)

    def _compare(self):
        b = self.other.currentData()
        if not (self.mh and b):
            return
        lib = self.main.lib
        load = lambda h: json.loads(lib.get_blob(h))

        def col(mh):
            out = []
            for r in lib.dossier(mh)[::-1]:
                st = narrative.story(r["kind"], r["artifacts"], load)
                out.append(f"<h4>{esc(run_kind_label(r['kind']))} · {when(r['sealed'])}</h4>"
                           + "".join(f"<p>{esc(s)}</p>" for s in st))
            return "".join(out) or "<p>(no sealed runs)</p>"

        name = lambda mh: lib.db.execute("SELECT display_name FROM models WHERE model_hash=?",
                                          (mh,)).fetchone()[0]
        self.cmp.setHtml(f"<table width='100%' cellpadding=6><tr><th>{esc(name(self.mh))}</th>"
                         f"<th>{esc(name(b))}</th></tr><tr><td valign=top>{col(self.mh)}</td>"
                         f"<td valign=top>{col(b)}</td></tr></table>")
