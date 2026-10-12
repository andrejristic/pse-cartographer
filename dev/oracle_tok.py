"""Tokenizer oracle: our GGUF-built tokenizer vs the model's original tokenizer.json.
   .dev/venv/bin/python dev/oracle_tok.py <model.gguf> <dir with tokenizer.json>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from tokenizers import Tokenizer
from pse_a.core.source.gguf import open_gguf
from pse_a.core.runtime.tokenizer import GGUFTokenizer

ours = GGUFTokenizer(open_gguf(sys.argv[1]).metadata)
ref = Tokenizer.from_file(str(Path(sys.argv[2]) / "tokenizer.json"))
lic = (Path(__file__).resolve().parents[1] / "app/pse_a/data/corpus_gpl3.txt").read_text(encoding="utf-8")
texts = [l for l in lic.splitlines() if l.strip()][:400] + [
    "The year 2026 had 12345 days?", "  leading spaces and émojis 🙂 中文 тест", "a  b\n\nc\t d",
    "def add(a, b):\n    return a + b  # 100%", "Ćevapi i burek — 2 + 2 = 4!", "x=1;y=22;z=333",
    "I'm can't won't they'll we've", "<|endoftext|> special?", "   ", "\n\n\n", "naïve café résumé"]
bad = 0
for t in texts:
    a = ours.encode(t, add_bos=False)
    b = ref.encode(t, add_special_tokens=False).ids
    if a != b:
        bad += 1
        if bad <= 5:
            print("MISMATCH", repr(t[:60]), "\n  ours", a[:16], "\n  ref ", b[:16])
print(f"{len(texts) - bad}/{len(texts)} texts tokenize identically")
