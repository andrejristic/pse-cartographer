"""
Cartographer main window (C3).

Everything PSE-A does, as one desktop application, plus the cartography flow:
  left   — the models in the dossier (and "Open GGUF…")
  centre — Model Card, ★ Cartographer, Spectral Map, Precision, Thought Waterfall, Features,
           J-space, Perplexity, Dossier, Findings & Conclusions, Swarm & Inbox
  right  — the conversation with the spokesperson ("the conversation is the spine")
  bottom — the running unit, its progress, the inbox badge and the Guardian shield

The window is a view over SHARED_LIB ([PSE-A] 10.1); missions run one at a time in the job
runner, and when one finishes every panel re-reads the dossier.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QDockWidget, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSplitter, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from pse_a import __version__ as ENGINE_VERSION
from pse_a.core.cost import MachineRates
from pse_a.shell import doctor
from pse_a.shell.guardian_view import shield_state
from pse_a.shell.shared_lib import SharedLib

from .. import __version__
from .. import spokesperson as spk
from ..spokesperson import Voice
from .cartographer_tabs import CartographerWidget
from .chat_dock import ChatPanel
from .common import JobRunner, esc
from .dossier_tab import DossierTab
from .findings_tab import FindingsTab
from .model_tab import ModelTab
from .result_tabs import (FeaturesTab, JSpaceTab, PplTab, PrecisionTab, SpectralTab,
                          WaterfallTab)
from .spokesperson_dialog import SpokespersonDialog
from .swarm_tab import SwarmTab

SHIELD = {"gray": "#999", "green": "#2a7", "yellow": "#d90", "red": "#c33"}


class _Probe(QObject):
    done = Signal(object)

    def run(self):
        self.done.emit(doctor.probe(benchmark=True))


class CartographerWindow(QMainWindow):
    def __init__(self, first_run: bool = True):
        super().__init__()
        self.setWindowTitle(f"PSE Cartographer {__version__}")
        self.resize(1400, 880)
        self.data_root = doctor.data_root()
        self.lib = SharedLib(self.data_root)
        self.voice = Voice(self.lib, spk.load_saved(self.lib))
        self._machine = doctor.probe(benchmark=False)
        self._benched = False
        self.jobs = JobRunner(self)
        self.current = None

        # ---- left: models
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Models</b>"))
        self.models = QListWidget()
        left.setMinimumWidth(220)
        self.models.currentItemChanged.connect(self._model_clicked)
        ll.addWidget(self.models, 1)
        ob = QPushButton("Open GGUF…")
        ob.clicked.connect(lambda: (self.tabs.setCurrentWidget(self.model_tab), self.model_tab._browse()))
        ll.addWidget(ob)

        # ---- centre: panels
        self.tabs = QTabWidget()
        self.model_tab = ModelTab(self)
        self.cartographer = CartographerWidget(self)
        self.spectral = SpectralTab(self)
        self.precision = PrecisionTab(self)
        self.waterfall = WaterfallTab(self)
        self.features = FeaturesTab(self)
        self.jspace = JSpaceTab(self)
        self.ppl = PplTab(self)
        self.dossier = DossierTab(self)
        self.findings = FindingsTab(self)
        self.swarm = SwarmTab(self)
        for w, name in ((self.model_tab, "Model Card"), (self.cartographer, "★ Cartographer"),
                        (self.spectral, "Spectral Map"), (self.precision, "Precision"),
                        (self.waterfall, "Thought Waterfall"), (self.features, "Features"),
                        (self.jspace, "J-space"), (self.ppl, "Perplexity"),
                        (self.dossier, "Dossier"), (self.findings, "Findings && Conclusions"),
                        (self.swarm, "Swarm && Inbox")):
            self.tabs.addTab(w, name)
        self.panels = [self.cartographer, self.spectral, self.precision, self.waterfall,
                       self.features, self.jspace, self.ppl, self.dossier, self.findings, self.swarm]
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(left)
        sp.addWidget(self.tabs)
        sp.setSizes([230, 1100])
        self.setCentralWidget(sp)

        # ---- right: the conversation
        self.chat = ChatPanel(self)
        dock = QDockWidget("Conversation", self)
        dock.setWidget(self.chat)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.chat_dock = dock

        # ---- bottom: status
        self.job_label = QLabel("")
        self.job_bar = QProgressBar()
        self.job_bar.setMaximumWidth(260)
        self.job_bar.setVisible(False)
        self.inbox_label = QLabel()
        self.shield = QLabel()
        self.statusBar().addWidget(self.job_label, 1)
        self.statusBar().addPermanentWidget(self.job_bar)
        self.statusBar().addPermanentWidget(self.inbox_label)
        self.statusBar().addPermanentWidget(self.shield)
        self.jobs.progress.connect(self._progress)
        self.jobs.busy_changed.connect(self._busy)

        # ---- menus
        fm = self.menuBar().addMenu("&File")
        a = QAction("Open GGUF…", self)
        a.triggered.connect(lambda: (self.tabs.setCurrentWidget(self.model_tab), self.model_tab._browse()))
        fm.addAction(a)
        a = QAction("Open data folder", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.data_root)))
        fm.addAction(a)
        fm.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        fm.addAction(a)
        m = self.menuBar().addMenu("&Cartographer")
        a = QAction("Choose spokesperson…", self)
        a.triggered.connect(self._choose_spokesperson)
        m.addAction(a)
        a = QAction("Machine report (doctor)…", self)
        a.triggered.connect(self._doctor)
        m.addAction(a)
        a = QAction("Show conversation", self)
        a.triggered.connect(lambda: self.chat_dock.show())
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        for label, fname in (("Guide (English)", "GUIDE-EN.md"), ("Vodič (srpski)", "VODIC-SR.md")):
            a = QAction(label, self)
            a.triggered.connect(lambda _=False, f=fname: self._show_doc(f))
            h.addAction(a)
        a = QAction("About / licence", self)
        a.triggered.connect(self._about)
        h.addAction(a)

        self._fill_models()
        self.update_badges()
        self._start_benchmark()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_badges)
        self._timer.start(15000)
        if first_run:
            QTimer.singleShot(0, self._first_run)

    # ------------------------------------------------------------ machine
    def machine(self):
        return self._machine

    def rates(self) -> MachineRates:
        return doctor.rates_of(self._machine) if self._benched else MachineRates(20.0, 1e-9)

    def profile(self):
        return doctor.select_profile(self._machine)

    def _start_benchmark(self):
        t = QThread(self)
        p = _Probe()
        p.moveToThread(t)
        t.started.connect(p.run)

        p.done.connect(self._bench_done)
        self._bench = (t, p)
        t.start()

    def closeEvent(self, ev):
        t = getattr(self, "_bench", (None,))[0]
        if t is not None:
            t.quit()
            t.wait(15000)
        if self.jobs.busy:
            self.jobs._thread.quit()
            self.jobs._thread.wait(2000)
        super().closeEvent(ev)

    def _bench_done(self, r):                          # GUI thread (slot of a QObject)
        self._machine, self._benched = r, True
        self._bench[0].quit()
        if self.model_tab.path:
            self.model_tab.open_path(self.model_tab.path)

    # ------------------------------------------------------------ models
    def _paths_file(self) -> Path:
        return Path(self.data_root) / "paths.json"

    def remember_path(self, mh, path, name, mm):
        try:
            d = json.loads(self._paths_file().read_text(encoding="utf-8"))
        except Exception:
            d = {}
        d[mh] = path
        self._paths_file().write_text(json.dumps(d, indent=1))
        self.lib.register_model(mh, name, mm.architecture, {})
        self._fill_models(select=mh)

    def path_of(self, mh):
        try:
            return json.loads(self._paths_file().read_text(encoding="utf-8")).get(mh)
        except Exception:
            return None

    def _fill_models(self, select=None):
        keep = select or self.current
        self.models.blockSignals(True)
        self.models.clear()
        for m in self.lib.models():
            n = len(self.lib.runs_of(m["model_hash"], "sealed"))
            it = QListWidgetItem(f"{m['display_name']}\n{m['architecture']} · {n} sealed")
            it.setData(Qt.UserRole, m["model_hash"])
            self.models.addItem(it)
            if m["model_hash"] == keep:
                self.models.setCurrentItem(it)
        self.models.blockSignals(False)

    def _model_clicked(self, it, _prev=None):
        if it is not None:
            self.select_model(it.data(Qt.UserRole))
            p = self.path_of(self.current)
            if p and os.path.isfile(p) and p != self.model_tab.path:
                self.model_tab.open_path(p)

    def select_model(self, mh):
        self.current = mh
        for i in range(self.models.count()):
            if self.models.item(i).data(Qt.UserRole) == mh:
                self.models.blockSignals(True)
                self.models.setCurrentRow(i)
                self.models.blockSignals(False)
        for p in self.panels:
            p.refresh(mh)

    # ------------------------------------------------------------ jobs
    def run_job(self, label, fn, on_done=None, on_fail=None):
        return self.jobs.start(label, fn, on_done=on_done, on_fail=on_fail, parent_widget=self)

    def _busy(self, on):
        self.job_bar.setVisible(on)
        self.job_label.setText(f"Working: {self.jobs.label}" if on else "")
        self.model_tab._enable(self.model_tab.path is not None)
        self.cartographer.analyze_btn.setEnabled(not on)
        self.findings.gen_btn.setEnabled(not on)

    def _progress(self, stage, i, n):
        self.job_label.setText(f"{self.jobs.label} — {stage}")
        if n:
            self.job_bar.setMaximum(n)
            self.job_bar.setValue(i)
        else:
            self.job_bar.setMaximum(0)

    def mission_finished(self, label, r):
        from pse_a.shell import prognostics as pg
        if isinstance(r, dict):
            if r.get("sealed"):
                msg = f"<p style='color:#2a7'>{esc(label)}: sealed into the dossier.</p>"
            else:
                why = r.get("refused", "")
                reasons = pg.gate_reasons(why)
                msg = (f"<p style='color:#c33'>{esc(label)}: "
                       f"{esc(pg.refusal(reasons) if reasons else why)}</p>")
            for p in r.get("prognoses", []) or []:
                msg += f"<p style='color:#a97'>{esc(p)}</p>"
            self.chat.narrate(msg)
        self._fill_models()
        if self.current:
            self.select_model(self.current)
        self.update_badges()

    def update_badges(self):
        n = len(self.lib.inbox(unseen_only=True))
        self.inbox_label.setText(f"Inbox: {n} new" if n else "Inbox: —")
        s = shield_state()
        txt = {"gray": "Guardian is not running on this machine",
               "green": "Guardian running, no events in 24 h",
               "yellow": "Guardian noticed deviations in the last 24 h",
               "red": "Guardian needs attention"}[s["level"]]
        self.shield.setText(f"<span style='color:{SHIELD[s['level']]}'>⛨</span>")
        self.shield.setToolTip(txt)

    # ------------------------------------------------------------ spokesperson (C1)
    def _first_run(self):
        if spk.first_run_needed():
            self._choose_spokesperson(first=True)
        elif self.voice.has_voice():
            self.chat.narrate(f"<p style='color:#888'>Spokesperson: <b>{esc(self.voice.backend.name)}</b>.</p>")
        else:
            self.chat.narrate("<p style='color:#a97'>No spokesperson chosen — measurements only. "
                              "Choose one anytime via <b>Cartographer → Choose spokesperson…</b></p>")

    def _choose_spokesperson(self, first=False):
        dlg = SpokespersonDialog(self, self.voice.backend)
        dlg.exec()
        if dlg.backend is not None:
            self.voice.backend = dlg.backend
            spk.save(dlg.backend, remember_key=dlg.remember)
            self.chat.narrate(f"<p style='color:#888'>Spokesperson set to <b>{esc(dlg.backend.name)}</b>.</p>")
        elif first:
            spk.save(None, dismissed=True)
            self.chat.narrate("<p style='color:#a97'>No spokesperson chosen — measurements only. "
                              "Choose one anytime via <b>Cartographer → Choose spokesperson…</b> "
                              "for the conversation and the reports.</p>")

    # ------------------------------------------------------------ help
    def _doctor(self):
        txt = doctor.render(self._machine, self.model_tab.path)
        b = QTextBrowser()
        b.setPlainText(txt)
        b.setStyleSheet("font-family: monospace")
        b.resize(760, 520)
        b.setWindowTitle("Machine report")
        b.show()
        self._doc_win = b

    def _show_doc(self, fname):
        base = Path(__file__).resolve().parents[2]
        for cand in (base / fname, base.parent / fname, base / "docs" / fname):
            if cand.is_file():
                b = QTextBrowser()
                b.setMarkdown(cand.read_text(encoding="utf-8"))
                b.resize(820, 700)
                b.setWindowTitle(fname)
                b.show()
                self._doc_win = b
                return
        QMessageBox.information(self, "Guide", f"{fname} was not found next to the application.")

    def _about(self):
        QMessageBox.about(self, "PSE Cartographer",
                          f"<h3>PSE Cartographer {__version__}</h3>"
                          f"<p>Live J-space cartography of a local GGUF model, on the PSE-A engine "
                          f"{ENGINE_VERSION} (numpy runtime, no PyTorch).</p>"
                          "<p>© 2026 Andrej Ristic / Zerava Technologies Inc.<br>"
                          "Free software under the GNU GPL version 3 or later, with the attribution "
                          "term in NOTICE. No warranty.</p>"
                          f"<p>Data folder: {esc(self.data_root)}</p>")
