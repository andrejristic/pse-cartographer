"""
Cartographer tab group (C4 + C7 render).

The dedicated flow for dissecting a local GGUF patient. The FIRST tab asks for the GGUF
path (disk or USB) and checks torch support; the rest render the live J-space cartography:
Basis (which layers mapped), Session (per-prompt readouts), Drift/Diff (the heart — how the
J-space moves), and Report (the narodski report + expert note).

Heavy work runs in a worker thread so the UI never freezes.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QProgressBar, QPushButton, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QCheckBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .. import drift as drift_mod
from .torch_install import TorchInstaller, commands_text, torch_available


class AnalyzeWorker(QObject):
    progress = Signal(str, int, int)     # stage, i, n
    done = Signal(object)                # bundle
    failed = Signal(str)

    def __init__(self, orchestrator, path, active):
        super().__init__()
        self.orch = orchestrator
        self.path = path
        self.active = active

    def run(self):
        try:
            bundle = self.orch.analyze_patient(
                self.path, active=self.active,
                progress=lambda stage, i, n: self.progress.emit(stage, i, n))
            self.done.emit(bundle)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class CartographerWidget(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main                 # CartographerWindow (holds lib + psel)
        self._orch = None
        self._bundle = None
        self._threads = []

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self.tabs.addTab(self._build_patient_tab(), "Patient")
        self.tabs.addTab(self._build_basis_tab(), "Basis")
        self.tabs.addTab(self._build_session_tab(), "Session")
        self.tabs.addTab(self._build_drift_tab(), "Drift / Diff")
        self.tabs.addTab(self._build_report_tab(), "Report")

    # -- Patient tab (first: give me the GGUF path) ------------------------- #
    def _build_patient_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<h3>Patient — local GGUF to dissect</h3>"))
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("path to a local .gguf (disk or USB)")
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse_patient)
        row.addWidget(self.path_edit); row.addWidget(browse)
        lay.addLayout(row)

        self.torch_status = QLabel()
        lay.addWidget(self.torch_status)
        self.install_log = QPlainTextEdit(); self.install_log.setReadOnly(True)
        self.install_log.setMaximumHeight(160); self.install_log.setVisible(False)
        lay.addWidget(self.install_log)
        self.install_btn = QPushButton("Install dissection support (torch + transformers)")
        self.install_btn.clicked.connect(self._install_torch)
        lay.addWidget(self.install_btn)

        self.active_chk = QCheckBox("Active cartographer (spokesperson generates the probes)")
        lay.addWidget(self.active_chk)
        self.analyze_btn = QPushButton("Analyze patient")
        self.analyze_btn.clicked.connect(self._analyze)
        lay.addWidget(self.analyze_btn)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        lay.addWidget(self.progress)
        lay.addStretch(1)
        self._refresh_torch()
        return w

    def _refresh_torch(self):
        if torch_available():
            self.torch_status.setText("✅ torch + transformers present — dissection enabled.")
            self.install_btn.setVisible(False)
        else:
            self.torch_status.setText(
                "⚠ torch/transformers not installed — L0/L1 still work, but J-space "
                "cartography needs them. Copy-paste, or use the button:\n\n"
                + commands_text(cpu=True))
            self.install_btn.setVisible(True)

    def _browse_patient(self):
        p, _ = QFileDialog.getOpenFileName(self, "Patient GGUF", "", "GGUF (*.gguf)")
        if p:
            self.path_edit.setText(p)

    def _install_torch(self):
        self.install_log.setVisible(True); self.install_btn.setEnabled(False)
        inst = TorchInstaller(cpu=True, parent=self)
        inst.line.connect(lambda s: self.install_log.appendPlainText(s))
        inst.finished.connect(self._install_done)
        inst.start()

    def _install_done(self, ok):
        self.install_btn.setEnabled(True)
        self._refresh_torch()
        self.install_log.appendPlainText("\n== done ==" if ok else "\n== still missing ==")

    def _analyze(self):
        path = self.path_edit.text().strip()
        if not path:
            self.torch_status.setText("Give a GGUF path first."); return
        if self._orch is None:
            from ..orchestrator import Orchestrator
            self._orch = Orchestrator(self.main.lib, self.main.psel)
        self.progress.setVisible(True); self.analyze_btn.setEnabled(False)
        w = AnalyzeWorker(self._orch, path, self.active_chk.isChecked())
        t = QThread(self)
        w.moveToThread(t); t.started.connect(w.run)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_done); w.failed.connect(self._on_failed)
        self._threads.append((t, w)); t.start()

    def _on_progress(self, stage, i, n):
        self.progress.setMaximum(n); self.progress.setValue(i)
        self.main.statusBar().showMessage(f"{stage}: {i}/{n}", 2000)

    def _on_failed(self, msg):
        self.progress.setVisible(False); self.analyze_btn.setEnabled(True)
        self.torch_status.setText(f"Analysis unavailable: {msg}\n\n"
                                  + commands_text(cpu=True))

    def _on_done(self, bundle):
        self.progress.setVisible(False); self.analyze_btn.setEnabled(True)
        self._bundle = bundle
        self._render_basis(bundle); self._render_session(bundle)
        self._render_drift(bundle); self._render_report(bundle)
        self.tabs.setCurrentIndex(3)     # jump to Drift / Diff — the heart

    # -- Basis tab ---------------------------------------------------------- #
    def _build_basis_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.basis_body = QTextEdit(); self.basis_body.setReadOnly(True)
        lay.addWidget(self.basis_body); return w

    def _render_basis(self, b):
        self.basis_body.setHtml(
            f"<h3>J-space basis</h3><p>Mapped layers: <b>{b['layers_mapped']}</b> · "
            f"model {b['model']['display_name']} ({b['model']['n_layers']} layers).</p>"
            f"<p style='color:#888'>The basis (averaged lens rows) is estimated once and "
            f"cached; per-prompt readouts reuse it.</p>")

    # -- Session tab -------------------------------------------------------- #
    def _build_session_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.session_body = QTextEdit(); self.session_body.setReadOnly(True)
        lay.addWidget(self.session_body); return w

    def _render_session(self, b):
        parts = ["<h3>Per-prompt readouts</h3>"]
        for r in b["readouts"]:
            lr = drift_mod.representative_layer(r)
            chips = ", ".join(lbl for lbl, _ in (lr.top_concepts[:12] if lr else []))
            parts.append(f"<p><b>{r.prompt}</b><br>ignition@L{r.ignition_layer} · "
                         f"holds: {chips}</p>")
        self.session_body.setHtml("".join(parts))

    # -- Drift / Diff tab (the heart) --------------------------------------- #
    def _build_drift_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.fig = Figure(figsize=(7, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig); lay.addWidget(self.canvas)
        row = QHBoxLayout()
        self.diff_a = QComboBox(); self.diff_b = QComboBox()
        diff_btn = QPushButton("Diff A → B"); diff_btn.clicked.connect(self._show_diff)
        row.addWidget(QLabel("A:")); row.addWidget(self.diff_a)
        row.addWidget(QLabel("B:")); row.addWidget(self.diff_b); row.addWidget(diff_btn)
        lay.addLayout(row)
        self.diff_body = QTextEdit(); self.diff_body.setReadOnly(True)
        self.diff_body.setMaximumHeight(150); lay.addWidget(self.diff_body)
        return w

    def _render_drift(self, b):
        outs = b["readouts"]
        d = drift_mod.drift(outs)
        igs = drift_mod.ignition_track(outs)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if d:
            ax.bar(range(1, len(outs)), d, color="#4c78a8")
        ax.set_ylim(0, 1); ax.set_xlabel("prompt index")
        ax.set_ylabel("J-space drift vs previous (0..1)")
        ax.set_title("How the held concepts move, prompt by prompt")
        ax2 = ax.twinx()
        ax2.plot(range(len(outs)), [(-1 if i is None else i) for i in igs],
                 "o--", color="#e45756", label="ignition layer")
        ax2.set_ylabel("ignition layer"); ax2.legend(fontsize=8)
        self.canvas.draw()
        self.diff_a.clear(); self.diff_b.clear()
        for i, r in enumerate(outs):
            self.diff_a.addItem(f"{i}: {r.prompt[:30]}", i)
            self.diff_b.addItem(f"{i}: {r.prompt[:30]}", i)
        if len(outs) > 1:
            self.diff_b.setCurrentIndex(1)

    def _show_diff(self):
        if not self._bundle:
            return
        outs = self._bundle["readouts"]
        a, b = self.diff_a.currentData(), self.diff_b.currentData()
        if a is None or b is None:
            return
        di = drift_mod.diff(outs[a], outs[b])
        self.diff_body.setHtml(
            f"<b>Turned ON:</b> {', '.join(di['turned_on']) or '—'}<br>"
            f"<b>Turned OFF:</b> {', '.join(di['turned_off']) or '—'}<br>"
            f"<b>Shared:</b> {', '.join(di['shared']) or '—'}<br>"
            f"<b>Ignition shift:</b> {di['ignition_shift']}")

    # -- Report tab --------------------------------------------------------- #
    def _build_report_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.report_body = QTextEdit(); self.report_body.setReadOnly(True)
        lay.addWidget(self.report_body); return w

    def _render_report(self, b):
        seal = "🔒 sealed" if b.get("sealed") else "⚠ not sealed"
        self.report_body.setHtml(
            f"<h2>Report — {b['model']['display_name']} · {seal}</h2>"
            f"<h3>What happened (plain language)</h3>"
            f"<p>{b.get('narodski_report','').replace(chr(10), '<br>')}</p>"
            f"<h3>Expert note</h3><p style='color:#555'>{b.get('expert_note','')}</p>")
