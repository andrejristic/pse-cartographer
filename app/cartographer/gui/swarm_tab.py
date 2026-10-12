"""Swarm Monitor + Inbox ([SPEC] 10.3, orchestration).

Units as cards — mission, remaining-budget ring, a small theta trace — for the work running
now; the recent history of runs (sealed, or died with the reason in plain words); and the
human-review inbox: every decision the system may not take alone arrives here as mail, not as
an interrupting pop-up. AZ-8: no game form is ever named; only prognostic domain language
(shell/prognostics.py is the only source of sentences about units).
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from pse_a.shell import prognostics as pg

from .common import Canvas, esc, html_view, when


class SwarmTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        sp = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.addWidget(QLabel("<b>Units at work</b> (mission · budget used · engagement θ)"))
        self.canvas = Canvas(8, 2.6)
        tl.addWidget(self.canvas)
        sp.addWidget(top)
        mid = QSplitter(Qt.Horizontal)
        self.history = html_view()
        mid.addWidget(self.history)
        inbox = QWidget()
        il = QVBoxLayout(inbox)
        il.addWidget(QLabel("<b>Inbox</b> — decisions and prognoses for you"))
        self.inbox = QListWidget()
        self.inbox.setWordWrap(True)
        il.addWidget(self.inbox)
        row = QHBoxLayout()
        seen = QPushButton("Mark read")
        seen.clicked.connect(self._seen)
        row.addStretch(1)
        row.addWidget(seen)
        il.addLayout(row)
        mid.addWidget(inbox)
        sp.addWidget(mid)
        sp.setSizes([260, 400])
        lay.addWidget(sp)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_live)
        self.timer.start(3000)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def refresh(self, _model_hash=None):
        self.refresh_live()
        self._history()
        self._inbox()

    def refresh_live(self):
        if not self.isVisible():
            return
        act = self.main.lib.active_runs()
        f = self.canvas.clear()
        if not act:
            ax = f.add_subplot(111)
            ax.text(0.5, 0.5, "No unit is working right now.", ha="center", va="center")
            ax.axis("off")
        n = min(len(act), 4)
        for k, a in enumerate(act[:4]):
            ax = f.add_subplot(1, n * 2, 2 * k + 1)
            frac = a["step"] / max(a["total"], 1)
            ax.pie([frac, 1 - frac], colors=["#4c78a8", "#e8e8e8"], startangle=90,
                   wedgeprops={"width": 0.3})
            ax.set_title(f"{pg.run_kind_label(a['kind'])}\n{(a['model'] or '')[:22]}", fontsize=7)
            ax2 = f.add_subplot(1, n * 2, 2 * k + 2)
            ax2.plot(a["theta"], color="#e45756")
            ax2.axhspan(25, 35, color="#2a7", alpha=0.08)
            ax2.set_ylim(20, 40)
            ax2.set_title(f"θ {a['theta'][-1]:.1f} · {a['label'][:20]}", fontsize=7)
            ax2.tick_params(labelsize=6)
        self.canvas.draw()

    def _history(self):
        lib = self.main.lib
        rows = lib.db.execute(
            "SELECT r.run_id, r.kind, r.state, r.started, r.sealed, m.display_name FROM runs r "
            "LEFT JOIN models m ON m.model_hash=r.model_hash ORDER BY r.started DESC LIMIT 40").fetchall()
        parts = ["<h3>Recent runs</h3>"]
        for run_id, kind, state, st, se, name in rows:
            if state == "sealed":
                badge = "<span style='color:#2a7'>sealed</span>"
                extra = f" · {lib.registry_count(run_id)} finding(s) entered knowledge"
            elif state == "died":
                why = lib.death_reason(run_id) or ""
                reasons = pg.gate_reasons(why)
                badge = "<span style='color:#c33'>not accepted</span>"
                extra = " · " + esc(pg.refusal(reasons) if reasons else why)
            else:
                badge = "<span style='color:#888'>working / interrupted</span>"
                extra = ""
            parts.append(f"<p>{badge} · <b>{esc(pg.run_kind_label(kind))}</b> · {esc(name or '')} · "
                         f"{when(st)}{extra}</p>")
        self.history.setHtml("".join(parts))

    def _inbox(self):
        self.inbox.clear()
        for it in self.main.lib.inbox():
            w = QListWidgetItem(("● " if not it["seen"] else "   ") +
                                f"{time.strftime('%m-%d %H:%M', time.localtime(it['t']))} · "
                                f"{pg.run_kind_label(it['run_kind'])}: {it['text']}")
            w.setData(Qt.UserRole, it["item_id"])
            self.inbox.addItem(w)

    def _seen(self):
        for it in self.inbox.selectedItems() or [self.inbox.item(i) for i in range(self.inbox.count())]:
            self.main.lib.mark_seen(it.data(Qt.UserRole))
        self._inbox()
        self.main.update_badges()
