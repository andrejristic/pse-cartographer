"""Findings & Conclusions — the spokesperson reasons over sealed results.

One model, or two models side by side (A vs B). Optional "Suggested next steps". The report
is sealed as its own run (every [tag] it cites must resolve to a sealed artifact); notes can be
attached; it exports to Markdown, HTML, PDF, or a zip with every underlying result.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPageLayout, QTextDocument
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from .. import findings as F
from .common import RunPicker, esc, html_view, when


class FindingsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.current = None
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Model A:"))
        self.model_a = QComboBox()
        row.addWidget(self.model_a, 1)
        row.addWidget(QLabel("compare with B (optional):"))
        self.model_b = QComboBox()
        row.addWidget(self.model_b, 1)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.next_chk = QCheckBox("Also propose next steps (which run to do next and why)")
        self.next_chk.setChecked(True)
        row2.addWidget(self.next_chk)
        row2.addStretch(1)
        self.gen_btn = QPushButton("Write findings & conclusions")
        self.gen_btn.clicked.connect(self._generate)
        row2.addWidget(self.gen_btn)
        lay.addLayout(row2)
        self.picker = RunPicker(("findings",), "Sealed reports:")
        self.picker.changed.connect(self._load)
        lay.addWidget(self.picker)
        self.view = html_view()
        lay.addWidget(self.view, 1)
        row3 = QHBoxLayout()
        for label, fn in (("Export Markdown…", self._export_md), ("Export HTML…", self._export_html),
                          ("Export PDF…", self._export_pdf),
                          ("Export results bundle (.zip)…", self._export_zip)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row3.addWidget(b)
        lay.addLayout(row3)

    def refresh(self, model_hash):
        models = [m for m in self.main.lib.models() if m["architecture"] != "comparison"]
        for box, blank in ((self.model_a, False), (self.model_b, True)):
            keep = box.currentData()
            box.blockSignals(True)
            box.clear()
            if blank:
                box.addItem("—", None)
            for m in models:
                box.addItem(m["display_name"], m["model_hash"])
            i = box.findData(keep if keep else (model_hash if not blank else None))
            box.setCurrentIndex(max(i, 0))
            box.blockSignals(False)
        self._fill_reports()

    def _owner(self):
        a, b = self.model_a.currentData(), self.model_b.currentData()
        if a and b and a != b:
            import hashlib
            return hashlib.sha256("|".join([a, b]).encode()).hexdigest()
        return a

    def _fill_reports(self):
        self.picker.fill(self.main.lib, self._owner())

    def _generate(self):
        a, b = self.model_a.currentData(), self.model_b.currentData()
        if not a:
            return
        mhs = [a] + ([b] if b and b != a else [])
        nxt = self.next_chk.isChecked()
        voice, lib = self.main.voice, self.main.lib

        def job(prog):
            prog("writing findings & conclusions", 0, 1)
            rep = F.generate(voice, lib, mhs, next_steps=nxt)
            rep["run_id"] = F.seal(lib, rep)
            return rep

        self.main.run_job("Findings & conclusions", job, on_done=self._done)

    def _done(self, rep):
        self._fill_reports()
        i = self.picker.box.findData(rep["run_id"])
        if i >= 0:
            self.picker.box.setCurrentIndex(i)
        self.main.mission_finished("Findings & conclusions", {"sealed": True, "run_id": rep["run_id"]})

    def _load(self, run_id):
        if not run_id:
            self.view.setHtml("<p>No sealed report for this selection yet.</p>")
            self.current = None
            return
        lib = self.main.lib
        h = lib.db.execute("SELECT content_hash FROM artifacts WHERE run_id=? AND name='findings.md'",
                           (run_id,)).fetchone()[0]
        md = lib.get_blob(h).decode("utf-8")
        inputs, _ = lib.artifact_json(run_id, "inputs.json")
        sealed = lib.db.execute("SELECT sealed FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
        self.current = {"run_id": run_id, "md": md, "inputs": inputs, "hash": h}
        doc = QTextDocument()
        doc.setMarkdown(md)
        warn = ""
        if not inputs.get("cited") and inputs.get("author") != "deterministic register":
            warn += ("<p style='color:#c33'><b>This text cites no sealed result.</b> Its claims cannot "
                     "be traced to a measurement — treat it with caution (small local models often "
                     "write reports like this; a larger or online spokesperson does better).</p>")
        if inputs.get("uncited_tags"):
            warn += (f"<p style='color:#c33'>Tags that do not match any sealed result (ignored): "
                    f"{esc(', '.join(inputs['uncited_tags']))}</p>")
        self.view.setHtml(
            f"<p style='color:#888'>🔒 sealed {when(sealed)} · author: {esc(inputs.get('author'))} · "
            f"cites {len(inputs.get('cited', []))} artifact(s) · artifact [{h[:8]}]</p>{warn}"
            + doc.toHtml().split("<body", 1)[-1].split(">", 1)[-1].rsplit("</body>", 1)[0])

    # ------------------------------------------------------------ export
    def _save(self, title, flt, suffix):
        if not self.current:
            return None
        p, _ = QFileDialog.getSaveFileName(self, title, f"findings_{self.current['run_id'][:8]}{suffix}", flt)
        return Path(p) if p else None

    def _export_md(self):
        p = self._save("Export Markdown", "Markdown (*.md)", ".md")
        if p:
            p.write_text(self.current["md"], encoding="utf-8")

    def _doc(self):
        doc = QTextDocument()
        doc.setMarkdown(self.current["md"])
        return doc

    def _export_html(self):
        p = self._save("Export HTML", "HTML (*.html)", ".html")
        if p:
            p.write_text(self._doc().toHtml(), encoding="utf-8")

    def _export_pdf(self):
        p = self._save("Export PDF", "PDF (*.pdf)", ".pdf")
        if p:
            from PySide6.QtPrintSupport import QPrinter
            pr = QPrinter(QPrinter.HighResolution)
            pr.setOutputFormat(QPrinter.PdfFormat)
            pr.setOutputFileName(str(p))
            pr.setPageOrientation(QPageLayout.Portrait)
            self._doc().print_(pr)

    def _export_zip(self):
        p = self._save("Export results bundle", "Zip (*.zip)", ".zip")
        if p:
            F.export_bundle(self.main.lib, self.current["run_id"], p)
