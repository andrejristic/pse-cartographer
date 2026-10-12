"""Shared GUI pieces: background jobs, matplotlib canvas, small helpers.

The GUI holds no state of its own ([PSE-A] 10.1): every panel is a render of something that
already sits in SHARED_LIB. Heavy work (missions) runs in a worker thread, one at a time —
the patient's weights are large and two missions would compete for the same RAM.
"""
from __future__ import annotations

import json
import time
import traceback

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox, QSizePolicy,
                               QTextBrowser, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pse_a.shell.prognostics import run_kind_label


class _Job(QObject):
    progress = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(lambda stage, i=0, n=0: self.progress.emit(str(stage), int(i), int(n))))
        except Exception as e:                                # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class JobRunner(QObject):
    """Runs one heavy job at a time in a QThread; the window shows its progress.

    Results come back through slots of THIS object, which lives in the GUI thread, so the
    callbacks always run in the GUI thread (a plain closure connected to a worker signal
    would run in the worker thread and touch widgets from there).
    """
    busy_changed = Signal(bool)
    progress = Signal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._job = None
        self._cb = (None, None, None)
        self.label = ""

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def start(self, label: str, fn, on_done=None, on_fail=None, parent_widget=None) -> bool:
        if self.busy:
            QMessageBox.information(parent_widget, "Busy",
                                    f"“{self.label}” is still running. One analysis at a time — "
                                    f"the model's weights need the memory.")
            return False
        self.label = label
        self._cb = (on_done, on_fail, parent_widget)
        t = QThread()
        job = _Job(fn)
        job.moveToThread(t)
        t.started.connect(job.run)
        job.progress.connect(self.progress)
        job.done.connect(self._ok)
        job.failed.connect(self._bad)
        self._thread, self._job = t, job
        self.busy_changed.emit(True)
        t.start()
        return True

    def _finish(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = self._job = None
        self.busy_changed.emit(False)

    def _ok(self, res):
        on_done, _, _ = self._cb
        self._finish()
        if on_done:
            on_done(res)

    def _bad(self, msg):
        _, on_fail, w = self._cb
        self._finish()
        if on_fail:
            on_fail(msg)
        else:
            QMessageBox.warning(w, self.label or "Error", msg)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, w=7, h=4):
        self.fig = Figure(figsize=(w, h), tight_layout=True)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.fig.clear()
        return self.fig


def when(t) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "—"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def jdump(o) -> str:
    return json.dumps(o, indent=1, default=float, ensure_ascii=False)


def html_view() -> QTextBrowser:
    v = QTextBrowser()
    v.setOpenExternalLinks(True)
    return v


class RunPicker(QWidget):
    """Choose one sealed run of given kind(s) for the current model."""
    changed = Signal(str)

    def __init__(self, kinds: tuple, label: str = "Run:"):
        super().__init__()
        self.kinds = kinds
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label))
        self.box = QComboBox()
        self.box.setMinimumWidth(360)
        lay.addWidget(self.box, 1)
        self.box.currentIndexChanged.connect(lambda _i: self.changed.emit(self.box.currentData() or ""))

    def fill(self, lib, model_hash: str | None):
        self.box.blockSignals(True)
        self.box.clear()
        if model_hash:
            for r in lib.dossier(model_hash)[::-1]:
                if r["kind"] in self.kinds:
                    self.box.addItem(f"{run_kind_label(r['kind'])} · {when(r['sealed'])} · {r['run_id'][:8]}",
                                     r["run_id"])
        self.box.blockSignals(False)
        self.changed.emit(self.box.currentData() or "")

    def current(self) -> str:
        return self.box.currentData() or ""
