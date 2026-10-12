"""PSE Cartographer engine CLI (headless): the same missions the GUI runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _lib():
    from .shell.shared_lib import SharedLib
    from .shell.doctor import data_root
    return SharedLib(data_root())


def _report(r: dict) -> None:
    from .shell import prognostics as pg
    if r.get("sealed"):
        print(f"\nSealed: run {r['run_id']}, {r.get('registry', 0)} findings in knowledge.")
    else:
        print("\n" + pg.refusal(pg.gate_reasons(r.get("refused", ""))) +
              (f"\n({r['refused']})" if r.get("refused") else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pse-cartographer-cli",
                                 description="PSE Cartographer engine (headless)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="machine capabilities, measured")
    d.add_argument("model", nargs="?", help="GGUF file for the model card + semaphore")
    d.add_argument("--no-bench", action="store_true")
    i = sub.add_parser("info", help="model card and quantization-type inventory")
    i.add_argument("model")
    st = sub.add_parser("selftest", help="check the installation (no network, no models)")
    st.add_argument("--k2", metavar="DIR", help="folder with f16 and quantized GGUFs of one model")
    l1 = sub.add_parser("l1", help="L1 spectral sweep (sealed mission)")
    l1.add_argument("model")
    l1.add_argument("--max", type=int, default=None)
    qd = sub.add_parser("qdiff", help="compare two precisions of the same model")
    qd.add_argument("ref")
    qd.add_argument("test")
    qd.add_argument("--max", type=int, default=None)
    l2 = sub.add_parser("lens", help="L2 logit lens over prompts")
    l2.add_argument("model")
    l2.add_argument("prompts", nargs="*")
    l4 = sub.add_parser("jspace", help="L4 J-space map")
    l4.add_argument("model")
    l4.add_argument("--layers", default="auto", help="e.g. 3,4,5 or auto = every layer")
    l4.add_argument("--probes", type=int, default=48)
    l3 = sub.add_parser("sae", help="L3 harvest + concept dictionary for one layer")
    l3.add_argument("model")
    l3.add_argument("--layer", type=int, required=True)
    l3.add_argument("--corpus")
    l3.add_argument("--profile", default=None, help="dev/toy/workstation/full")
    pp = sub.add_parser("ppl", help="perplexity of every GGUF in a folder")
    pp.add_argument("folder")
    pp.add_argument("--tokens", type=int, default=2048)
    pp.add_argument("--text")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        from .tests import selftest
        return selftest.main(a.k2)
    if a.cmd == "doctor":
        from .shell import doctor
        print(doctor.render(doctor.probe(benchmark=not a.no_bench), a.model))
        return 0
    if a.cmd == "info":
        from .core.source.gguf import open_gguf
        from .core.source.canonical import build_map
        g = open_gguf(a.model)
        print(build_map(g).summary())
        print(f"model_hash {g.model_hash()}\n")
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values())
        print(f"{'type':9} {'tensors':>8} {'% params':>10} {'MB':>9}")
        for k, v in inv.items():
            print(f"{k:9} {v['tensors']:8} {100*v['elements']/tot:9.1f}% {v['bytes']/1e6:9.0f}")
        return 0
    lib = _lib()
    try:
        if a.cmd == "l1":
            from .shell.l1_mission import run_l1
            r = run_l1(a.model, lib, max_matrices=a.max)
        elif a.cmd == "qdiff":
            from .shell.qdiff_mission import run_qdiff
            r = run_qdiff(a.ref, a.test, lib, max_matrices=a.max)
        elif a.cmd == "lens":
            from .core.lens.jlens import load_ambiguous
            from .shell import lens_missions as lm
            r = lm.run_logit_lens(a.model, lib, a.prompts or load_ambiguous())
        elif a.cmd == "jspace":
            from .core.source.gguf import open_gguf
            from .core.source.canonical import build_map
            from .shell import lens_missions as lm
            nl = build_map(open_gguf(a.model)).n_layers
            layers = (list(range(1, nl)) if a.layers == "auto"
                      else [int(x) for x in a.layers.split(",")])
            r = lm.run_jspace(a.model, lib, layers, a.probes)
        elif a.cmd == "sae":
            from .shell.doctor import probe, select_profile
            from .shell.sae_mission import run_sae
            from .profiles import PROFILES
            prof = PROFILES[a.profile] if a.profile else select_profile(probe(benchmark=True))
            text = Path(a.corpus).read_text(encoding="utf-8", errors="ignore") if a.corpus else None
            r = run_sae(a.model, lib, a.layer, prof, text, a.corpus or "built-in corpus")
        elif a.cmd == "ppl":
            from .shell.ppl_mission import run_ppl_ladder
            text = Path(a.text).read_text(encoding="utf-8", errors="ignore") if a.text else None
            r = run_ppl_ladder(a.folder, lib, a.tokens, text, a.text or "built-in corpus")
        else:
            return 1
    except RuntimeError as e:
        print(f"Not started: {e}")
        return 2
    _report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
