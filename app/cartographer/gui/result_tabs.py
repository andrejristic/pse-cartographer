"""Result panels, one per algorithm layer ([SPEC] 10.3), each a render of a sealed artifact.

Spectral Map (L1) · Precision (qdiff) · Thought Waterfall (L2) · Features (L3) ·
J-space Map (L4) · Perplexity ladder. Every panel shows the formal register (numbers and
charts) next to the human register (the engine's deterministic story, with artifact tags).
"""
from __future__ import annotations

import json
import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from pse_a.shell import narrative
from pse_a.shell.prognostics import regime_label

from .common import Canvas, RunPicker, esc, html_view

ROLES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def _story_html(lib, run_id: str, kind: str) -> str:
    arts = dict(lib.db.execute("SELECT name, content_hash FROM artifacts WHERE run_id=?",
                               (run_id,)).fetchall())
    s = narrative.story(kind, arts, lambda h: json.loads(lib.get_blob(h)))
    return "".join(f"<p>{esc(x)}</p>" for x in s)


class _Panel(QWidget):
    KINDS: tuple = ()
    ART = ""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.lay = QVBoxLayout(self)
        self.picker = RunPicker(self.KINDS)
        self.lay.addWidget(self.picker)
        self.picker.changed.connect(self._load)
        self.empty = QLabel("No sealed result of this kind for the selected model yet — start one "
                            "from the Model Card.")
        self.empty.setWordWrap(True)
        self.lay.addWidget(self.empty)
        self.body = QWidget()
        self.lay.addWidget(self.body, 1)
        self.build(QVBoxLayout(self.body))

    def refresh(self, model_hash):
        self.picker.fill(self.main.lib, model_hash)

    def _load(self, run_id):
        self.empty.setVisible(not run_id)
        self.body.setVisible(bool(run_id))
        if run_id:
            rep, h = self.main.lib.artifact_json(run_id, self.ART)
            if rep is not None:
                self.render(run_id, rep)

    def build(self, lay):
        raise NotImplementedError

    def render(self, run_id, rep):
        raise NotImplementedError


# ------------------------------------------------------------------ L1
class SpectralTab(_Panel):
    KINDS = ("l1_sweep",)
    ART = "l1_report.json"

    def build(self, lay):
        sp = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        self.canvas = Canvas(7, 4)
        self.canvas.mpl_connect("button_press_event", self._click)
        top.addWidget(self.canvas)
        self.story = html_view()
        top.addWidget(self.story)
        top.setSizes([700, 300])
        sp.addWidget(top)
        bottom = QSplitter(Qt.Horizontal)
        self.table = QTableWidget()
        self.table.itemSelectionChanged.connect(self._row)
        bottom.addWidget(self.table)
        self.esd = Canvas(5, 3)
        bottom.addWidget(self.esd)
        sp.addWidget(bottom)
        lay.addWidget(sp)

    def render(self, run_id, rep):
        self.run_id, self.rep = run_id, rep
        self.hist, _ = self.main.lib.artifact_json(run_id, "esd_histograms.json")
        M = rep["matrices"]
        layers = sorted({m["layer"] for m in M if m["layer"] is not None})
        grid = np.full((len(ROLES), len(layers)), np.nan)
        self._cells = {}
        for m in M:
            if m["layer"] is None or m["role"] not in ROLES:
                continue
            i, j = ROLES.index(m["role"]), layers.index(m["layer"])
            a = m.get("alpha")
            grid[i, j] = a if a is not None and math.isfinite(a) else 8.0
            self._cells[(i, j)] = m["name"]
        f = self.canvas.clear()
        ax = f.add_subplot(111)
        im = ax.imshow(np.clip(grid, 1.5, 8), aspect="auto", cmap="viridis_r", vmin=1.5, vmax=8)
        ax.set_yticks(range(len(ROLES)), ROLES)
        ax.set_xticks(range(0, len(layers), max(1, len(layers) // 12)),
                      [layers[k] for k in range(0, len(layers), max(1, len(layers) // 12))])
        ax.set_xlabel("layer")
        ax.set_title("Heavy-tail exponent α per matrix (2–4 trained · >6 random-like) — click a cell")
        f.colorbar(im, ax=ax, fraction=0.03)
        self.canvas.draw()
        self._layers = layers
        cols = ["matrix", "role", "layer", "type", "regime", "α", "tail %", "stable rank", "MP fit"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(M))
        for r, m in enumerate(M):
            vals = [m["name"], m["role"], m["layer"], m["ggml_type"], regime_label(m["regime"]),
                    "—" if m.get("alpha") is None else f"{m['alpha']:.2f}",
                    f"{100 * m['tail_frac']:.1f}", f"{m['stable_rank']:.1f}",
                    "yes" if m.get("mp_fit_ok") else "no"]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()
        self.story.setHtml(f"<h3>{esc(rep['model'])}</h3>" + _story_html(self.main.lib, run_id, "l1_sweep"))
        if M:
            self._show_esd(M[0]["name"])

    def _click(self, ev):
        if ev.xdata is None or ev.ydata is None:
            return
        nm = self._cells.get((int(round(ev.ydata)), int(round(ev.xdata))))
        if nm:
            self._show_esd(nm)

    def _row(self):
        it = self.table.selectedItems()
        if it:
            self._show_esd(self.table.item(it[0].row(), 0).text())

    def _show_esd(self, name):
        h = (self.hist or {}).get(name)
        m = next((x for x in self.rep["matrices"] if x["name"] == name), None)
        if not h or not m:
            return
        f = self.esd.clear()
        ax = f.add_subplot(111)
        e = np.asarray(h["edges"])
        ax.bar(e[:-1], h["counts"], width=np.diff(e), align="edge", color="#4c78a8")
        ax.axvline(m["edge"], color="#e45756", ls="--", label="Marchenko–Pastur edge")
        ax.set_yscale("log")
        ax.set_title(f"{name}: {regime_label(m['regime'])}"
                     + (f", α={m['alpha']:.2f}" if m.get("alpha") else ""), fontsize=9)
        ax.set_xlabel("eigenvalue of WᵀW / n")
        ax.legend(fontsize=7)
        self.esd.draw()


# ------------------------------------------------------------------ qdiff
class PrecisionTab(_Panel):
    KINDS = ("qdiff",)
    ART = "qdiff_report.json"

    def build(self, lay):
        sp = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        self.canvas = Canvas(6, 4)
        ll.addWidget(self.canvas)
        self.table = QTableWidget()
        ll.addWidget(self.table)
        sp.addWidget(left)
        self.story = html_view()
        sp.addWidget(self.story)
        sp.setSizes([700, 300])
        lay.addWidget(sp)

    def render(self, run_id, rep):
        S = rep["strata"]
        keys = list(S)
        f = self.canvas.clear()
        ax = f.add_subplot(111)
        ax.barh(range(len(keys)), [1 - S[k]["kept_quarter"] for k in keys], color="#4c78a8")
        ax.set_yticks(range(len(keys)), [k.replace("|", "  ").replace("->", "→") for k in keys], fontsize=7)
        ax.set_xlabel("energy lost in the top quarter of directions")
        ax.set_title(f"{rep['ref']}  →  {rep['test']}", fontsize=9)
        self.canvas.draw()
        cols = ["role | types", "n", "kept (top ¼)", "noise ratio", "energy change", "σ² shift", "α shift"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(keys))
        for r, k in enumerate(keys):
            v = S[k]
            for c, x in enumerate([k, v["n"], f"{v['kept_quarter']:.4f}", f"{v['noise_ratio']:.3f}",
                                   f"{v['energy_change']:.4f}", f"{v['sigma2_shift']:.4f}",
                                   f"{v['alpha_shift']:.3f}"]):
                self.table.setItem(r, c, QTableWidgetItem(str(x)))
        self.table.resizeColumnsToContents()
        self.story.setHtml(_story_html(self.main.lib, run_id, "qdiff"))


# ------------------------------------------------------------------ L2
class WaterfallTab(_Panel):
    KINDS = ("l2_lens",)
    ART = "lens_report.json"

    def build(self, lay):
        row = QHBoxLayout()
        row.addWidget(QLabel("Prompt:"))
        self.prompt = QComboBox()
        self.prompt.currentIndexChanged.connect(self._show)
        row.addWidget(self.prompt, 1)
        lay.addLayout(row)
        sp = QSplitter(Qt.Horizontal)
        self.grid = QTableWidget()
        sp.addWidget(self.grid)
        self.story = html_view()
        sp.addWidget(self.story)
        sp.setSizes([800, 260])
        lay.addWidget(sp)

    def render(self, run_id, rep):
        self.rep = rep
        self.prompt.blockSignals(True)
        self.prompt.clear()
        for i, it in enumerate(rep["items"]):
            self.prompt.addItem(it["prompt"], i)
        self.prompt.blockSignals(False)
        self.story.setHtml(_story_html(self.main.lib, run_id, "l2_lens")
                           + "<p style='color:#888'>Cell = the token the model would say if forced to "
                             "speak at that layer; colour = confidence. Forward-only: it cannot see "
                             "computation that is not yet in words.</p>")
        self._show()

    def _show(self):
        if not getattr(self, "rep", None) or self.prompt.currentIndex() < 0:
            return
        it = self.rep["items"][self.prompt.currentData()]
        toks, layers = it["tokens"], it["layers"]
        self.grid.clear()
        self.grid.setRowCount(len(layers))
        self.grid.setColumnCount(len(toks))
        self.grid.setHorizontalHeaderLabels([repr(t)[1:-1] for t in toks])
        self.grid.setVerticalHeaderLabels([f"L{l}" for l in layers])
        for r in range(len(layers)):
            for c in range(len(toks)):
                tok = it["top"][r][c][0]
                p = it["p"][r][c]
                cell = QTableWidgetItem(repr(tok)[1:-1])
                cell.setBackground(QColor.fromHsvF(0.6, min(1.0, p), 1.0))
                cell.setToolTip(f"{', '.join(map(repr, it['top'][r][c]))}  p={p:.2f}")
                if str(layers[r]) == str(it["commit_layer"]) and c == len(toks) - 1:
                    cell.setForeground(QColor("#c00"))
                self.grid.setItem(r, c, cell)
        self.grid.resizeColumnsToContents()


# ------------------------------------------------------------------ L3
class FeaturesTab(_Panel):
    KINDS = ("l3_sae",)
    ART = "sae_report.json"

    def build(self, lay):
        sp = QSplitter(Qt.Horizontal)
        self.cards = html_view()
        sp.addWidget(self.cards)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.canvas = Canvas(4, 3)
        rl.addWidget(self.canvas)
        self.story = html_view()
        rl.addWidget(self.story)
        sp.addWidget(right)
        sp.setSizes([650, 400])
        lay.addWidget(sp)

    def render(self, run_id, rep):
        f = self.canvas.clear()
        ax = f.add_subplot(111)
        h = rep.get("history", [])
        if h:
            ax.plot([x["step"] for x in h], [x["fvu"] for x in h], label="unexplained variance")
            ax2 = ax.twinx()
            ax2.plot([x["step"] for x in h], [x["l0"] for x in h], color="#e45756", label="active / position")
            ax.set_xlabel("training step")
            ax.set_title("Dictionary training", fontsize=9)
        self.canvas.draw()
        parts = [f"<h3>Layer {rep['layer']} · {rep['n_features']} concepts · corpus: {esc(rep['corpus'])}</h3>"]
        for c in rep["cards"][:64]:
            ex = []
            for e in c["examples"][:4]:
                ctx = "".join(f"<b style='background:#ffe08a'>{esc(t)}</b>" if k == e["at"] else esc(t)
                              for k, t in enumerate(e["ctx"]))
                ex.append(f"<li>{ctx} <span style='color:#888'>({e['act']:.2f})</span></li>")
            parts.append(f"<p><b>feature {c['feature']}</b> · fires on {100 * c['freq']:.2f}% of positions"
                         + (" · <i>stable</i>" if c["stable"] else "") + f"</p><ul>{''.join(ex)}</ul>")
        self.cards.setHtml("".join(parts))
        p = rep["superposition_profile"]
        self.story.setHtml(_story_html(self.main.lib, run_id, "l3_sae")
                           + f"<p style='color:#888'>N_eff {p['n_eff']:.0f} · stable {p['n_stable']} · "
                             f"FVU {rep['fvu']:.3f} · L0 {rep['l0']:.1f} · accepted: {rep['accepted']}</p>")


# ------------------------------------------------------------------ L4
class JSpaceTab(_Panel):
    KINDS = ("l4_jspace",)
    ART = "jspace_map.json"

    def build(self, lay):
        sp = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        self.canvas = Canvas(7, 3.5)
        top.addWidget(self.canvas)
        self.story = html_view()
        top.addWidget(self.story)
        top.setSizes([700, 300])
        sp.addWidget(top)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        row = QHBoxLayout()
        row.addWidget(QLabel("Concepts held, for prompt:"))
        self.prompt = QComboBox()
        self.prompt.currentIndexChanged.connect(self._chips)
        row.addWidget(self.prompt, 1)
        bl.addLayout(row)
        self.table = QTableWidget()
        bl.addWidget(self.table)
        sp.addWidget(bottom)
        lay.addWidget(sp)

    def render(self, run_id, m):
        self.m = m
        L = m["layers"]
        pl = m["per_layer"]
        vf = [pl[str(x)]["var_frac"] for x in L]
        en = [pl[str(x)]["entropy"] for x in L]
        f = self.canvas.clear()
        ax = f.add_subplot(111)
        reg = m["regions"]
        for name, col in (("sensory", "#d0e4f5"), ("workspace", "#fde2b8"), ("motor", "#d6f0d0")):
            for x in reg.get(name, []):
                ax.axvspan(x - 0.5, x + 0.5, color=col, lw=0)
        ax.plot(L, vf, "o-", color="#4c78a8", label="J-space share of the state")
        ax.axhspan(*m.get("expected_var", (0.03, 0.10)), color="#999", alpha=0.12, label="expected band")
        if reg.get("ignition") is not None:
            ax.axvline(reg["ignition"], color="#e45756", ls="--", label=f"ignition @ L{reg['ignition']}")
        ax.set_xlabel("layer")
        ax.set_ylabel("variance share")
        ax2 = ax.twinx()
        ax2.plot(L, en, ":", color="#72b7b2", label="decomposition entropy")
        ax2.set_ylabel("entropy")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("Depth profile (sensory · workspace · motor)", fontsize=9)
        self.canvas.draw()
        self.story.setHtml(_story_html(self.main.lib, run_id, "l4_jspace")
                           + (f"<p style='color:#888'>{esc(reg['note'])}</p>" if reg.get("note") else ""))
        self.prompt.blockSignals(True)
        self.prompt.clear()
        for i, p in enumerate(m["prompts"]):
            self.prompt.addItem(p, i)
        self.prompt.blockSignals(False)
        self._chips()

    def _chips(self):
        if not getattr(self, "m", None) or self.prompt.currentIndex() < 0:
            return
        i = self.prompt.currentData()
        L = self.m["layers"]
        self.table.setRowCount(len(L))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["layer", "concepts held (weight)"])
        for r, x in enumerate(L):
            chips = self.m["chips"][str(x)][i]
            self.table.setItem(r, 0, QTableWidgetItem(f"L{x}"))
            self.table.setItem(r, 1, QTableWidgetItem("  ".join(f"{t!r} {w:.2f}" for t, w in chips)))
        self.table.resizeColumnsToContents()


# ------------------------------------------------------------------ PPL
class PplTab(_Panel):
    KINDS = ("ppl_ladder",)
    ART = "ppl_ladder.json"

    def build(self, lay):
        sp = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        self.canvas = Canvas(6, 3.5)
        ll.addWidget(self.canvas)
        self.table = QTableWidget()
        ll.addWidget(self.table)
        sp.addWidget(left)
        self.story = html_view()
        sp.addWidget(self.story)
        lay.addWidget(sp)

    def render(self, run_id, rep):
        R = rep["rows"]
        f = self.canvas.clear()
        ax = f.add_subplot(111)
        ax.bar(range(len(R)), [100 * r["ppl_rel"] for r in R], color="#4c78a8")
        ax.set_xticks(range(len(R)), [r["file"][-18:] for r in R], rotation=30, fontsize=7, ha="right")
        ax.set_ylabel("perplexity change vs reference (%)")
        self.canvas.draw()
        cols = ["file", "perplexity", "change", "MB"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(R))
        for i, r in enumerate(R):
            for c, v in enumerate([r["file"], f"{r['ppl']:.3f}", f"{100 * r['ppl_rel']:+.2f}%",
                                   f"{r['bytes'] / 1e6:.0f}"]):
                self.table.setItem(i, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.story.setHtml(_story_html(self.main.lib, run_id, "ppl_ladder")
                           + f"<p style='color:#888'>Text: {esc(rep['eval'])}, {rep['n_tokens']} tokens.</p>")
