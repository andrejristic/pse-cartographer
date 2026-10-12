"""
Findings & Conclusions (the tab where the spokesperson reasons over sealed results).

Input: every sealed result of one model — or of two models for an A-vs-B comparison —
told in the engine's deterministic human register (narrative.py), each sentence tagged with
the hash of the artifact it comes from. The spokesperson writes Findings (each one citing
its [tag]), Conclusions, and optionally Suggested next steps (which run to do next and why).

The report is sealed as its own run of kind `findings` (SharedLib.seal_document): every cited
tag must resolve to a sealed artifact (layer 1) and the document must hash-verify (layer 2).
It does not enter the priors — it is a reading of findings, not a finding. Like every run it
is immutable; a new reading is a new run, and notes can be attached to it.

Without a spokesperson the report is the deterministic register itself, labelled as such.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

from pse_a.shell import narrative
from pse_a.shell.prognostics import run_kind_label

TAG = re.compile(r"\[([0-9a-f]{8})\]")


def _name(lib, mh: str) -> str:
    row = lib.db.execute("SELECT display_name FROM models WHERE model_hash=?", (mh,)).fetchone()
    return row[0] if row else mh[:12]


def gather(lib, model_hashes: list) -> tuple[str, list]:
    """Facts text (one block per model) + list of (run_id, kind, artifacts)."""
    blocks, runs_out = [], []
    load = lambda h: json.loads(lib.get_blob(h))
    for mh in model_hashes:
        lines = [f"MODEL: {_name(lib, mh)}"]
        for r in lib.dossier(mh)[::-1]:
            if r["kind"] == "findings":
                continue
            story = narrative.story(r["kind"], r["artifacts"], load)
            if not story:
                continue
            lines.append(f"- {run_kind_label(r['kind'])} (run {r['run_id']}, sealed "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['sealed']))}):")
            lines += [f"    {s}" for s in story]
            for n in r["notes"]:
                lines.append(f"    human note: {n['text']}")
            runs_out.append((r["run_id"], r["kind"], r["artifacts"]))
        if len(lines) == 1:
            lines.append("- (no sealed results yet)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), runs_out


PROMPT = """Below are sealed measurement results{scope}. Write a report in Markdown with these \
sections:

## Findings
Numbered findings. Each finding states one thing that was measured, keeps the numbers, and \
ends with the [tag] of the artifact it comes from (the 8-character tags in the results).
{compare}
## Conclusions
What the findings together suggest, with the uncertainty stated in the sentence itself. \
Separate what is established from what is only suggested.
{next_steps}
Rules: use ONLY these results; do not invent numbers or runs; every finding must carry a tag \
that appears in the results; if something important was not measured, say so.

RESULTS:
{facts}
"""

NEXT = """
## Suggested next steps
Two to four concrete next runs in this tool (for example: map J-space with more prompts, \
compare two precisions, train a concept dictionary on a given layer), each with one sentence \
on what it would settle.
"""
COMPARE = """
## A vs B
The differences between the two models that the results support, each with tags from both.
"""


def generate(voice, lib, model_hashes: list, next_steps: bool = True) -> dict:
    facts, runs = gather(lib, model_hashes)
    names = [_name(lib, m) for m in model_hashes]
    if voice is not None and voice.has_voice():
        prompt = PROMPT.format(
            scope=" for " + " and ".join(names),
            compare=COMPARE if len(model_hashes) > 1 else "",
            next_steps=NEXT if next_steps else "", facts=facts)
        text = voice.ask(prompt, system=(
            "You are the reporting voice of an analysis tool for local language models. "
            "You write precise, honest reports from sealed measurements only. Never name "
            "theoretical frameworks or clinical terms; speak about measurements, convergence, "
            "learning and reproducibility."))
        author = voice.backend.name
        t = text.strip()
        if t.startswith("```"):                       # neki modeli umotaju ceo izvestaj
            t = t.split("\n", 1)[1] if "\n" in t else ""
            t = t.rsplit("```", 1)[0] if t.rstrip().endswith("```") else t
        text = t.strip()
    else:
        text = ("## Findings\n\n(No spokesperson is configured, so this is the system's own "
                "deterministic account of the sealed results, without interpretation.)\n\n"
                + facts)
        author = "deterministic register"
    known = {h[:8] for _, _, arts in runs for h in arts.values()}
    cited = sorted(set(TAG.findall(text)))
    bad = [c for c in cited if c not in known]
    return {"text": text, "facts": facts, "cited": [c for c in cited if c in known],
            "uncited_tags": bad, "author": author, "models": model_hashes, "names": names,
            "next_steps": next_steps}


def seal(lib, rep: dict) -> str:
    """Seal the report as a `findings` run. Returns run_id."""
    mhs = rep["models"]
    if len(mhs) == 1:
        owner = mhs[0]
    else:
        owner = hashlib.sha256("|".join(mhs).encode()).hexdigest()
        lib.register_model(owner, " ↔ ".join(rep["names"]), "comparison", {})
    run = lib.open_run(owner, "findings", {"kind": "findings", "models": mhs,
                                           "next_steps": rep["next_steps"],
                                           "author": rep["author"]}, {})
    lib.attach(run, "findings.md", rep["text"].encode(), media="md")
    lib.attach(run, "inputs.json", json.dumps({
        "models": mhs, "names": rep["names"], "facts": rep["facts"],
        "cited": rep["cited"], "uncited_tags": rep["uncited_tags"],
        "author": rep["author"]}).encode())
    lib.seal_document(run, rep["cited"])
    return run.run_id


def export_bundle(lib, run_id: str, dest: Path) -> Path:
    """A zip with the report, its inputs, and every artifact of every cited model run."""
    rep_md, _ = lib.db.execute("SELECT content_hash, 1 FROM artifacts WHERE run_id=? AND "
                               "name='findings.md'", (run_id,)).fetchone()
    inputs, _ = lib.artifact_json(run_id, "inputs.json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("findings.md", lib.get_blob(rep_md))
        z.writestr("inputs.json", json.dumps(inputs, indent=1))
        for mh in inputs["models"]:
            for r in lib.dossier(mh):
                for name, h in r["artifacts"].items():
                    z.writestr(f"results/{r['kind']}_{r['run_id']}/{name}", lib.get_blob(h))
    dest = Path(dest)
    dest.write_bytes(buf.getvalue())
    return dest
