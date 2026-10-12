"""Model Card (L0) + launching the analyses.

[SPEC] 10.3: "metadata, architecture, and the capability semaphore — the machine's abilities
declared up front; no mid-run surprises" (AZ-2), with estimated times (I-2). Every analysis of
PSE-A is started from here and runs as a sealed PSE mission in the background.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from pse_a.core.capability import semaphore
from pse_a.core.source.canonical import build_map
from pse_a.core.source.gguf import open_gguf
from pse_a.profiles import PROFILES

from .common import esc, html_view

LAYER_TXT = {"L0": "open the model", "L1": "spectral anatomy", "L2": "thought by layer",
             "L3": "concept dictionary", "L4": "J-space"}


class ModelTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.path = None
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("path to a local .gguf (disk or USB)")
        b = QPushButton("Open GGUF…")
        b.clicked.connect(self._browse)
        go = QPushButton("Show card")
        go.clicked.connect(lambda: self.open_path(self.path_edit.text().strip()))
        row.addWidget(self.path_edit, 1)
        row.addWidget(b)
        row.addWidget(go)
        lay.addLayout(row)
        self.card = html_view()
        lay.addWidget(self.card, 3)

        box = QGroupBox("Run an analysis on this model (each one is a sealed mission)")
        f = QFormLayout(box)
        self.l1_btn = QPushButton("L1 · spectral sweep of every weight matrix")
        self.l1_btn.clicked.connect(self._run_l1)
        f.addRow(self.l1_btn)
        r2 = QHBoxLayout()
        self.lens_prompts = QLineEdit()
        self.lens_prompts.setPlaceholderText("prompts separated by |  (empty = built-in ambiguous set)")
        self.l2_btn = QPushButton("L2 · thought waterfall")
        self.l2_btn.clicked.connect(self._run_l2)
        r2.addWidget(self.lens_prompts, 1)
        r2.addWidget(self.l2_btn)
        f.addRow(r2)
        r4 = QHBoxLayout()
        self.probes = QSpinBox()
        self.probes.setRange(8, 512)
        self.probes.setValue(48)
        self.l4_btn = QPushButton("L4 · J-space map (every layer)")
        self.l4_btn.clicked.connect(self._run_l4)
        r4.addWidget(QLabel("probes"))
        r4.addWidget(self.probes)
        r4.addStretch(1)
        r4.addWidget(self.l4_btn)
        f.addRow(r4)
        r3 = QHBoxLayout()
        self.sae_layer = QSpinBox()
        self.sae_layer.setRange(0, 999)
        self.profile = QComboBox()
        for p in PROFILES:
            self.profile.addItem(p, p)
        self.corpus = QLineEdit()
        self.corpus.setPlaceholderText("corpus .txt (empty = built-in GPL-3.0 text)")
        cb = QPushButton("…")
        cb.clicked.connect(lambda: self._pick_into(self.corpus, "Text corpus", "Text (*.txt)"))
        self.l3_btn = QPushButton("L3 · concept dictionary")
        self.l3_btn.clicked.connect(self._run_l3)
        for w in (QLabel("layer"), self.sae_layer, QLabel("profile"), self.profile, self.corpus, cb,
                  self.l3_btn):
            r3.addWidget(w)
        f.addRow(r3)
        rq = QHBoxLayout()
        self.qd_other = QLineEdit()
        self.qd_other.setPlaceholderText("the same model in another precision (.gguf)")
        qb = QPushButton("…")
        qb.clicked.connect(lambda: self._pick_into(self.qd_other, "Other precision", "GGUF (*.gguf)"))
        self.qd_btn = QPushButton("Compare precisions (this = reference)")
        self.qd_btn.clicked.connect(self._run_qdiff)
        rq.addWidget(self.qd_other, 1)
        rq.addWidget(qb)
        rq.addWidget(self.qd_btn)
        f.addRow(rq)
        rp = QHBoxLayout()
        self.ppl_dir = QLineEdit()
        self.ppl_dir.setPlaceholderText("folder with several precisions of one model (incl. an f16)")
        pb = QPushButton("…")
        pb.clicked.connect(self._pick_dir)
        self.ppl_btn = QPushButton("Perplexity ladder")
        self.ppl_btn.clicked.connect(self._run_ppl)
        rp.addWidget(self.ppl_dir, 1)
        rp.addWidget(pb)
        rp.addWidget(self.ppl_btn)
        f.addRow(rp)
        lay.addWidget(box, 2)
        self.buttons = [self.l1_btn, self.l2_btn, self.l4_btn, self.l3_btn, self.qd_btn, self.ppl_btn]
        self._enable(False)

    # ------------------------------------------------------------ helpers
    def _enable(self, on: bool):
        for b in self.buttons:
            b.setEnabled(on and not self.main.jobs.busy)

    def _pick_into(self, edit, title, flt):
        p, _ = QFileDialog.getOpenFileName(self, title, "", flt)
        if p:
            edit.setText(p)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "GGUF ladder folder")
        if d:
            self.ppl_dir.setText(d)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open a local GGUF", "", "GGUF (*.gguf)")
        if p:
            self.open_path(p)

    # ------------------------------------------------------------ card
    def open_path(self, path: str):
        if not path or not os.path.isfile(path):
            self.card.setHtml("<p>Give the path to an existing .gguf file.</p>")
            return
        self.path = path
        self.path_edit.setText(path)
        try:
            g = open_gguf(path)
            mm = build_map(g)
        except Exception as e:                                 # noqa: BLE001
            self.card.setHtml(f"<h3>Cannot open this file</h3><p>{esc(e)}</p>")
            self._enable(False)
            return
        mh = g.model_hash()
        self.main.remember_path(mh, path, os.path.basename(path), mm)
        caps = semaphore(g, mm, self.main.rates(), self.main.machine().ram_bytes)
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values()) or 1
        n_params = tot
        sem = "".join(
            f"<tr><td><b>{k}</b></td><td>{LAYER_TXT[k]}</td><td>"
            + ("<span style='color:#2a7'>available</span>" if caps.layers.get(k)
               else f"<span style='color:#c33'>unavailable</span> — {esc(caps.blockers.get(k, ''))}")
            + "</td></tr>" for k in ("L0", "L1", "L2", "L3", "L4"))
        est = "".join(f"<li>{esc(k)}: <b>{esc(v)}</b></li>" for k, v in caps.estimates.items())
        notes = "".join(f"<li>{esc(n)}</li>" for n in caps.notes)
        types = ", ".join(f"{k} {100 * v['elements'] / tot:.0f}%" for k, v in inv.items())
        self.card.setHtml(
            f"<h2>{esc(os.path.basename(path))}</h2>"
            f"<p>{esc(mm.summary())}</p>"
            f"<p>{n_params / 1e9:.2f} B parameters · {g._file_size / 1e9:.2f} GB on disk · "
            f"types: {esc(types)}<br><span style='color:#888'>model_hash {mh[:16]}</span></p>"
            f"<h3>Capability semaphore (this machine)</h3><table cellpadding=3>{sem}</table>"
            + (f"<h3>Estimated time</h3><ul>{est}</ul>" if est else "")
            + (f"<h3>Notes</h3><ul>{notes}</ul>" if notes else ""))
        self.sae_layer.setMaximum(max(mm.n_layers - 1, 0))
        self.sae_layer.setValue(mm.n_layers // 2)
        prof = self.main.profile()
        self.profile.setCurrentText(prof.name)
        self.caps = caps
        self._enable(True)
        self.l2_btn.setEnabled(self.l2_btn.isEnabled() and caps.layers.get("L2", False))
        self.l3_btn.setEnabled(self.l3_btn.isEnabled() and caps.layers.get("L3", False))
        self.l4_btn.setEnabled(self.l4_btn.isEnabled() and caps.layers.get("L4", False))
        self.l1_btn.setEnabled(self.l1_btn.isEnabled() and caps.layers.get("L1", False))
        self.main.select_model(mh)

    # ------------------------------------------------------------ missions
    def _mission(self, label, fn):
        import re
        frac = re.compile(r"\[\s*(\d+)/(\d+)\]|chunk (\d+)/(\d+)")

        def wrapped(prog):
            def p(msg):
                m = frac.search(str(msg))
                i, n = ((int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4))) if m else (0, 0))
                prog(str(msg).strip()[:90], i, n)
            return fn(p)
        self.main.run_job(label, wrapped, on_done=lambda r: self.main.mission_finished(label, r))

    def _run_l1(self):
        from pse_a.shell.l1_mission import run_l1
        p = self.path
        self._mission("L1 spectral sweep", lambda pr: run_l1(p, self.main.lib, progress=pr))

    def _run_l2(self):
        from pse_a.core.lens.jlens import load_ambiguous
        from pse_a.shell.lens_missions import run_logit_lens
        p = self.path
        prompts = [x.strip() for x in self.lens_prompts.text().split("|") if x.strip()] or load_ambiguous()
        self._mission("L2 thought waterfall",
                      lambda pr: run_logit_lens(p, self.main.lib, prompts, progress=pr))

    def _run_l4(self):
        from pse_a.shell.lens_missions import run_jspace
        p, n = self.path, self.probes.value()
        nl = build_map(open_gguf(p)).n_layers
        self._mission("L4 J-space map",
                      lambda pr: run_jspace(p, self.main.lib, list(range(1, nl)), n, progress=pr))

    def _run_l3(self):
        from pse_a.shell.sae_mission import CORPUS_NAME, run_sae
        p, L = self.path, self.sae_layer.value()
        prof = PROFILES[self.profile.currentData()]
        cpath = self.corpus.text().strip()
        text = open(cpath, encoding="utf-8", errors="ignore").read() if cpath else None
        name = os.path.basename(cpath) if cpath else CORPUS_NAME
        self._mission("L3 concept dictionary",
                      lambda pr: run_sae(p, self.main.lib, L, prof, text, name, progress=pr))

    def _run_qdiff(self):
        from pse_a.shell.qdiff_mission import run_qdiff
        a, b = self.path, self.qd_other.text().strip()
        if not b:
            return
        self._mission("Precision comparison", lambda pr: run_qdiff(a, b, self.main.lib, progress=pr))

    def _run_ppl(self):
        from pse_a.shell.ppl_mission import run_ppl_ladder
        d = self.ppl_dir.text().strip()
        if not d:
            return
        self._mission("Perplexity ladder", lambda pr: run_ppl_ladder(d, self.main.lib, progress=pr))
