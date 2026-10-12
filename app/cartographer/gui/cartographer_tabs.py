"""
Cartographer tab group (C4 + C7 render).

The dedicated flow for dissecting a local GGUF patient. The FIRST tab asks for the GGUF path
(disk or USB) and shows what this machine can do with it; the rest render the J-space
cartography: Basis (which layers were mapped, with which probe words), Session (per-prompt
readouts), Drift / Diff (the heart — how the J-space moves) and Report (the plain-language
report + expert note). Earlier sealed cartography runs can be reopened from the run picker.

Heavy work runs in the window's job runner so the UI never freezes.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from .. import drift as drift_mod
from .. import routing as routing_mod
from ..session import readouts_from_payload
from .common import Canvas, RunPicker, esc


class CartographerWidget(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._bundle = None
        outer = QVBoxLayout(self)
        self.picker = RunPicker(("cartography",), "Sealed cartography run:")
        self.picker.changed.connect(self._load_run)
        outer.addWidget(self.picker)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self.tabs.addTab(self._build_patient_tab(), "Patient")
        self.tabs.addTab(self._build_basis_tab(), "Basis")
        self.tabs.addTab(self._build_session_tab(), "Session")
        self.tabs.addTab(self._build_drift_tab(), "Drift / Diff")
        self.tabs.addTab(self._build_report_tab(), "Report")
        self.tabs.addTab(self._build_experts_tab(), "Experts")

    def refresh(self, model_hash):
        self.picker.fill(self.main.lib, model_hash)
        p = self.main.path_of(model_hash) if model_hash else None
        if p and not self.path_edit.text():
            self.path_edit.setText(p)

    # -- Patient tab (first: give me the GGUF path) ------------------------- #
    def _build_patient_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<h3>Patient — the local GGUF to dissect</h3>"))
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("path to a local .gguf (disk or USB)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_patient)
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        lay.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        opt = QHBoxLayout()
        opt.addWidget(QLabel("probe words"))
        self.n_probes = QSpinBox()
        self.n_probes.setRange(16, 512)
        self.n_probes.setValue(128)
        opt.addWidget(self.n_probes)
        self.active_chk = QCheckBox("Active cartographer (the spokesperson writes the next prompt)")
        opt.addWidget(self.active_chk)
        opt.addStretch(1)
        lay.addLayout(opt)
        lay.addWidget(QLabel("Prompts to feed (one per line; empty = the built-in ambiguous battery):"))
        self.battery = QPlainTextEdit()
        self.battery.setMaximumHeight(140)
        lay.addWidget(self.battery)
        self.analyze_btn = QPushButton("Analyze patient")
        self.analyze_btn.clicked.connect(self._analyze)
        lay.addWidget(self.analyze_btn)
        lay.addWidget(QLabel(
            "<p style='color:#888'>Order: spectral anatomy (L1, once per model) → J-space basis for "
            "every layer → readout of each prompt → drift → report → seal. The basis is cached, so "
            "a second session on the same model is fast.</p>"))
        lay.addStretch(1)
        return w

    def _browse_patient(self):
        p, _ = QFileDialog.getOpenFileName(self, "Patient GGUF", "", "GGUF (*.gguf)")
        if p:
            self.path_edit.setText(p)
            self.main.model_tab.open_path(p)

    def _analyze(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            self.status.setText("Give a GGUF path first.")
            return
        from ..orchestrator import DEFAULT_BATTERY, Orchestrator
        battery = [l.strip() for l in self.battery.toPlainText().splitlines() if l.strip()] or DEFAULT_BATTERY
        if self.active_chk.isChecked() and not self.main.voice.has_voice():
            self.status.setText("Active cartographer needs a spokesperson — choose one from the "
                                "Cartographer menu, or uncheck it.")
            return
        orch = Orchestrator(self.main.lib, self.main.voice, self.main.data_root)
        n, active = self.n_probes.value(), self.active_chk.isChecked()
        m = self.main.machine()
        self.status.setText("Working… (progress in the status bar)")
        self.main.run_job(
            "J-space cartography",
            lambda prog: orch.analyze_patient(path, battery=battery, active=active, n_probes=n,
                                              progress=prog, rates=self.main.rates(),
                                              ram_bytes=m.ram_bytes),
            on_done=self._on_done, on_fail=self._on_failed)

    def _on_failed(self, msg):
        self.status.setText(f"Analysis not done: {esc(msg)}")

    def _on_done(self, bundle):
        self._bundle = bundle
        seal = ("sealed into the dossier" if bundle.get("sealed")
                else f"measured but not sealed — {bundle.get('refused', '')}")
        self.main.mission_finished("J-space cartography", bundle)   # refreshes every panel
        self.status.setText(f"Done: {len(bundle['readouts'])} prompts, {len(bundle['layers_mapped'])} "
                            f"layers; {seal}.")
        # show THIS run, not whatever the refreshed run picker selected
        self.picker.box.blockSignals(True)
        i = self.picker.box.findData(bundle.get("run_id"))
        if i >= 0:
            self.picker.box.setCurrentIndex(i)
        else:
            self.picker.box.insertItem(0, "this session (not sealed)", "")
            self.picker.box.setCurrentIndex(0)
        self.picker.box.blockSignals(False)
        self._render_all(bundle)
        self.tabs.setCurrentIndex(3)                      # jump to Drift / Diff — the heart

    def _load_run(self, run_id):
        if not run_id:
            return
        art, _ = self.main.lib.artifact_json(run_id, "cartography.json")
        rep, _ = self.main.lib.artifact_json(run_id, "report.json")
        if art is None:
            return
        b = {"model": {"display_name": art.get("model"), "n_layers": art.get("n_layers")},
             "layers_mapped": art["layers_mapped"], "probes": art.get("probes", []),
             "readouts": readouts_from_payload(art), "session_summary": art.get("summary", {}),
             "narodski_report": (rep or {}).get("plain_report", ""),
             "expert_note": (rep or {}).get("expert_note", ""), "voice": (rep or {}).get("voice"),
             "sealed": True, "run_id": run_id}
        self._bundle = b
        self._render_all(b)

    def _render_all(self, b):
        self._render_basis(b)
        self._render_session(b)
        self._render_drift(b)
        self._render_report(b)
        self._render_experts(b)

    # -- Basis tab ---------------------------------------------------------- #
    def _build_basis_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.basis_body = QTextEdit()
        self.basis_body.setReadOnly(True)
        lay.addWidget(self.basis_body)
        return w

    def _render_basis(self, b):
        probes = b.get("probes")
        if probes is None and b.get("readouts"):
            probes = []
        self.basis_body.setHtml(
            f"<h3>J-space basis</h3><p>Mapped layers: <b>{b['layers_mapped']}</b> · model "
            f"{esc(b['model']['display_name'])} ({b['model']['n_layers']} layers)"
            + (" · basis loaded from cache" if b.get("basis_cached") else "") + ".</p>"
            "<p style='color:#888'>For each layer, the averaged lens rows W_U·J_l of the probe words "
            "are estimated once (one backward pass per calibration prompt covers every layer) and "
            "cached; per-prompt readouts reuse them.</p>"
            + (f"<p><b>Probe words</b>: {esc(', '.join(probes))}</p>" if probes else ""))

    # -- Session tab -------------------------------------------------------- #
    def _build_session_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.session_body = QTextEdit()
        self.session_body.setReadOnly(True)
        lay.addWidget(self.session_body)
        return w

    def _render_session(self, b):
        parts = ["<h3>Per-prompt readouts</h3><p style='color:#888'>Concepts are read at the "
                 "ignition layer when one is found, otherwise at the layer with the largest J-space "
                 "share.</p>"]
        for r in b["readouts"]:
            lr = drift_mod.representative_layer(r)
            chips = ", ".join(esc(lbl) for lbl, _ in (lr.top_concepts[:12] if lr else []))
            ign = f"ignition @ L{r.ignition_layer}" if r.ignition_layer is not None else "no ignition"
            parts.append(f"<p><b>{esc(r.prompt)}</b><br>{ign} · read at L{lr.layer if lr else '?'} · "
                         f"share {lr.var_frac if lr else 0:.3f} · holds: {chips}</p>")
        self.session_body.setHtml("".join(parts))

    # -- Drift / Diff tab (the heart) --------------------------------------- #
    def _build_drift_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.canvas = Canvas(7, 4)
        lay.addWidget(self.canvas)
        row = QHBoxLayout()
        self.diff_a = QComboBox()
        self.diff_b = QComboBox()
        diff_btn = QPushButton("Diff A → B")
        diff_btn.clicked.connect(self._show_diff)
        row.addWidget(QLabel("A:"))
        row.addWidget(self.diff_a)
        row.addWidget(QLabel("B:"))
        row.addWidget(self.diff_b)
        row.addWidget(diff_btn)
        lay.addLayout(row)
        self.diff_body = QTextEdit()
        self.diff_body.setReadOnly(True)
        self.diff_body.setMaximumHeight(150)
        lay.addWidget(self.diff_body)
        return w

    def _render_drift(self, b):
        outs = b["readouts"]
        d = drift_mod.drift(outs)
        igs = drift_mod.ignition_track(outs)
        f = self.canvas.clear()
        ax = f.add_subplot(111)
        if d:
            ax.bar(range(1, len(outs)), d, color="#4c78a8", label="drift vs previous")
        ax.set_ylim(0, 1)
        ax.set_xlabel("prompt index")
        ax.set_ylabel("J-space drift vs previous (0..1)")
        ax.set_title("How the held concepts move, prompt by prompt")
        ax2 = ax.twinx()
        ax2.plot(range(len(outs)), [(-1 if i is None else i) for i in igs], "o--", color="#e45756",
                 label="ignition layer (−1 = none)")
        ax2.set_ylabel("ignition layer")
        ax2.legend(fontsize=8, loc="upper right")
        self.canvas.draw()
        self.diff_a.clear()
        self.diff_b.clear()
        for i, r in enumerate(outs):
            self.diff_a.addItem(f"{i}: {r.prompt[:30]}", i)
            self.diff_b.addItem(f"{i}: {r.prompt[:30]}", i)
        if len(outs) > 1:
            self.diff_b.setCurrentIndex(1)
        self.diff_body.clear()

    def _show_diff(self):
        if not self._bundle:
            return
        outs = self._bundle["readouts"]
        a, b = self.diff_a.currentData(), self.diff_b.currentData()
        if a is None or b is None:
            return
        di = drift_mod.diff(outs[a], outs[b])
        self.diff_body.setHtml(
            f"<b>Turned ON:</b> {esc(', '.join(di['turned_on'])) or '—'}<br>"
            f"<b>Turned OFF:</b> {esc(', '.join(di['turned_off'])) or '—'}<br>"
            f"<b>Shared:</b> {esc(', '.join(di['shared'])) or '—'}<br>"
            f"<b>Ignition shift:</b> {di['ignition_shift']}")

    # -- Report tab --------------------------------------------------------- #
    def _build_report_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.report_body = QTextEdit()
        self.report_body.setReadOnly(True)
        lay.addWidget(self.report_body)
        return w

    def _render_report(self, b):
        seal = "🔒 sealed" if b.get("sealed") else "⚠ not sealed"
        voice = (f"written by {esc(b['voice'])}" if b.get("voice")
                 else "no spokesperson — deterministic text")
        self.report_body.setHtml(
            f"<h2>Report — {esc(b['model']['display_name'])} · {seal}</h2>"
            f"<p style='color:#888'>{voice}</p>"
            f"<h3>What happened (plain language)</h3>"
            f"<p>{esc(b.get('narodski_report', '')).replace(chr(10), '<br>')}</p>"
            f"<h3>Expert note</h3><p style='color:#555'>{esc(b.get('expert_note', '')).replace(chr(10), '<br>')}</p>")

    # -- Experts tab (mixture-of-experts routing) --------------------------- #
    def _build_experts_tab(self):
        from PySide6.QtWidgets import QSplitter, QTableWidget
        from PySide6.QtCore import Qt
        w = QWidget()
        lay = QVBoxLayout(w)
        self.exp_note = QLabel()
        self.exp_note.setWordWrap(True)
        lay.addWidget(self.exp_note)
        row = QHBoxLayout()
        row.addWidget(QLabel("Routing for:"))
        self.exp_prompt = QComboBox()
        self.exp_prompt.currentIndexChanged.connect(self._draw_experts)
        row.addWidget(self.exp_prompt, 1)
        lay.addLayout(row)
        sp = QSplitter(Qt.Vertical)
        self.exp_canvas = Canvas(8, 4)
        sp.addWidget(self.exp_canvas)
        self.exp_table = QTableWidget()
        sp.addWidget(self.exp_table)
        lay.addWidget(sp, 1)
        return w

    def _render_experts(self, b):
        outs = b["readouts"]
        self._exp_b = b
        moe = routing_mod.has_routing(outs)
        self.exp_prompt.setEnabled(moe)
        self.exp_table.setVisible(moe)
        self.exp_canvas.setVisible(moe)
        if not moe:
            self.exp_note.setText("This model is dense: every token goes through the same MLP. Expert "
                                  "routing appears here for mixture-of-experts models (Granite MoE, "
                                  "OLMoE).")
            return
        s = routing_mod.summary(outs)
        corr = s.get("drift_correlation")
        self.exp_note.setText(
            f"<b>Mixture of experts:</b> each token goes to {s['experts_per_token']} of "
            f"{s['n_experts']} experts at every layer. Routing drift between prompts "
            f"{s['mean_routing_drift']:.2f} on average (0 = same experts, 1 = all different)"
            + (f"; it correlates with the concept drift at {corr:.2f}." if corr is not None else "."))
        self.exp_prompt.blockSignals(True)
        self.exp_prompt.clear()
        self.exp_prompt.addItem("all prompts", None)
        for i, r in enumerate(outs):
            self.exp_prompt.addItem(f"{i}: {r.prompt[:40]}", i)
        self.exp_prompt.blockSignals(False)
        self._draw_experts()
        from PySide6.QtWidgets import QTableWidgetItem
        ec = routing_mod.expert_concepts(outs)
        rows = [(L, e, cs) for L in sorted(ec) for e, cs in sorted(ec[L].items())]
        self.exp_table.setColumnCount(3)
        self.exp_table.setHorizontalHeaderLabels(["layer", "expert", "concepts held when this expert is chosen"])
        self.exp_table.setRowCount(len(rows))
        for i, (L, e, cs) in enumerate(rows):
            for c, v in enumerate([f"L{L}", str(e), ", ".join(f"{t}" for t, _ in cs)]):
                self.exp_table.setItem(i, c, QTableWidgetItem(v))
        self.exp_table.resizeColumnsToContents()

    def _draw_experts(self):
        b = getattr(self, "_exp_b", None)
        if not b or not routing_mod.has_routing(b["readouts"]):
            return
        outs = b["readouts"]
        which = self.exp_prompt.currentData()
        U = routing_mod.usage_matrix(outs, which)
        layers = sorted(outs[0].routing)
        f = self.exp_canvas.clear()
        ax = f.add_subplot(1, 2, 1)
        im = ax.imshow(U, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_xlabel("expert")
        ax.set_ylabel("layer")
        ax.set_yticks(range(0, len(layers), max(1, len(layers) // 8)),
                      [layers[k] for k in range(0, len(layers), max(1, len(layers) // 8))])
        ax.set_title("Share of routed tokens per expert", fontsize=9)
        f.colorbar(im, ax=ax, fraction=0.04)
        ax2 = f.add_subplot(1, 2, 2)
        rd = routing_mod.routing_drift(outs)
        jd = drift_mod.drift(outs)
        x = range(1, len(outs))
        ax2.plot(x, rd, "o-", color="#e45756", label="routing drift")
        ax2.plot(x, jd, "s--", color="#4c78a8", label="concept drift")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("prompt index")
        ax2.set_title("Experts vs concepts, prompt to prompt", fontsize=9)
        ax2.legend(fontsize=7)
        self.exp_canvas.draw()
