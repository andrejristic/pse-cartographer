"""Tokenizer iz GGUF metapodataka -- bez tokenizer.json, bez transformers-a.

GGUF nosi vokabular, merge-ove (ili ocene) i id-jeve specijalnih tokena pod
llama.cpp imenima. Odavde se gradi isti tokenizer koji model ocekuje:

  tokenizer.ggml.model == "gpt2"  -> byte-level BPE (Qwen2/3, Llama 3, ...);
                                     pre-tokenizer regex po tokenizer.ggml.pre
  tokenizer.ggml.model == "llama" -> SentencePiece BPE sa byte fallback-om
                                     (Llama 2, Mistral, TinyLlama...); merge-ovi
                                     se rekonstruisu iz ocena kad ih fajl nema

Ako se tokenizer razlikuje od originala makar za jedan token, "cipovi pojmova"
su pogresni. Zato tests/oracle_tokenizer.py poredi token po token sa
transformers-om na vecem skupu tekstova (lokalno, pre svakog izdanja).

Pravila preuzeta iz transformers integrations/gguf (Apache-2.0) -- ponasanje,
ne kod.
"""
from __future__ import annotations

_QWEN2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
                r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_TEKKEN_SPLIT = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
                 r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
                 r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_GPT2_SPLIT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
# vrednost moze biti TORKA regexa: primenjuju se redom (kao u llama.cpp-u), npr.
# "refact" prvo izdvoji svaku cifru, pa ostatak cepa GPT-2 pravilom
_STABLELM2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# kad fajl ne kaze tokenizer.ggml.pre, pravilo originalnog tokenizer.json po arhitekturi
DEFAULT_PRE = {"stablelm": "stablelm2", "qwen2": "qwen2", "qwen3": "qwen2"}
SPLITS = {
    "stablelm2": _STABLELM2_SPLIT,
    # Granite 3.x je oznacen "refact", ali njegov tokenizer.json koristi obican GPT-2
    # regex (izmereno: 411/411 identicno); pravi refact deli cifre pojedinacno
    "refact": _GPT2_SPLIT,
    "starcoder": (r"\p{N}", _GPT2_SPLIT),
    "qwen2": _QWEN2_SPLIT, "deepseek-r1-qwen": _QWEN2_SPLIT, "megrez": _QWEN2_SPLIT,
    "llama3": _LLAMA3_SPLIT, "llama-v3": _LLAMA3_SPLIT, "llama-bpe": _LLAMA3_SPLIT,
    "smaug-bpe": _LLAMA3_SPLIT, "dbrx": _LLAMA3_SPLIT, "falcon3": _LLAMA3_SPLIT,
    "pixtral": _LLAMA3_SPLIT, "tekken": _TEKKEN_SPLIT,
}

CONTROL, USER_DEFINED = 3, 4


def bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def merges_from_scores(tokens: list[str], scores: list[float]) -> list[tuple[str, str]]:
    """SentencePiece BPE bez merge liste: svaki token koji se moze iseci na dva
    postojeca daje merge, sa ocenom tog tokena; bolja ocena = raniji merge."""
    vocab = {t: i for i, t in enumerate(tokens)}
    out = []
    for tok, sc in zip(tokens, scores):
        local = []
        for i in range(1, len(tok)):
            l, r = tok[:i], tok[i:]
            if l in vocab and r in vocab:
                local.append((l, r, sc))
        local.sort(key=lambda x: (vocab[x[0]], vocab[x[1]]))
        out.extend(local)
    out.sort(key=lambda v: (v[2], len(v[0]), len(v[1])), reverse=True)
    return [(a, b) for a, b, _ in out]


class GGUFTokenizer:
    def __init__(self, meta: dict):
        from tokenizers import (AddedToken, Regex, Tokenizer, decoders, models,
                                normalizers, pre_tokenizers)
        self.kind = str(meta.get("tokenizer.ggml.model", "gpt2"))
        toks = list(meta["tokenizer.ggml.tokens"])
        types = list(meta.get("tokenizer.ggml.token_type") or [1] * len(toks))
        self.tokens = toks
        self.bos_id = meta.get("tokenizer.ggml.bos_token_id")
        self.eos_id = meta.get("tokenizer.ggml.eos_token_id")
        default_bos = self.kind == "llama"
        self.add_bos = bool(meta.get("tokenizer.ggml.add_bos_token", default_bos)) \
            and self.bos_id is not None
        self.chat_template = meta.get("tokenizer.chat_template")

        if self.kind == "gpt2":
            spelling = bytes_to_unicode()
            alphabet = set(spelling.values())
            vocab = {}
            for i, t in enumerate(toks):
                if types[i] in (CONTROL, USER_DEFINED) and not set(t) <= alphabet:
                    key = t           # dodat token (npr. niz razmaka kod Phi-2) zadrzava SVOJ id
                else:
                    key = t if set(t) <= alphabet else "".join(spelling[b] for b in t.encode("utf-8"))
                vocab.setdefault(key, i)
            merges = [tuple(m.split(" ", 1)) for m in meta.get("tokenizer.ggml.merges", [])]
            tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False,
                                      byte_fallback=False))
            pre = meta.get("tokenizer.ggml.pre") or DEFAULT_PRE.get(
                str(meta.get("general.architecture", "")), "")
            split = SPLITS.get(str(pre))
            if split:
                regs = split if isinstance(split, tuple) else (split,)
                tk.pre_tokenizer = pre_tokenizers.Sequence(
                    [pre_tokenizers.Split(Regex(r), behavior="isolated", invert=False) for r in regs]
                    + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
            else:
                tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                            use_regex=True)
            tk.decoder = decoders.ByteLevel()
            specials = [AddedToken(t, special=True, normalized=False)
                        for t, ty in zip(toks, types) if ty == CONTROL]
        elif self.kind == "llama":
            arch = str(meta.get("general.architecture", ""))
            gemma = arch in ("gemma", "gemma2", "gemma3")
            if gemma:
                # Gemma pise tokene od vise razmaka kao prave razmake; GemmaTokenizer ih trazi
                # kao "▁▁"; merge-ovi slede REDOSLED u fajlu, ne ocene (transformers gguf)
                toks = ["▁" * len(t) if " " in t and not t.strip() else t for t in toks]
            vocab = {t: i for i, t in enumerate(toks)}
            if "tokenizer.ggml.merges" in meta:
                merges = [tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]
            elif gemma:
                merges = merges_from_scores(toks, [-float(i) for i in range(len(toks))])
            else:
                merges = merges_from_scores(toks, list(meta["tokenizer.ggml.scores"]))
            unk = meta.get("tokenizer.ggml.unknown_token_id")
            tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=True,
                                      byte_fallback=True,
                                      unk_token=toks[unk] if unk is not None else None))
            prefix = bool(meta.get("tokenizer.ggml.add_space_prefix", not gemma))
            seq = [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
            if gemma:
                tk.normalizer = normalizers.Replace(" ", "▁")
                tk.pre_tokenizer = pre_tokenizers.Split(" ", behavior="merged_with_previous",
                                                        invert=False)
            elif arch == "phi3":
                # prefiks razmak dolazi iz normalizatora, ne iz Metaspace-a (transformers gguf)
                tk.normalizer = normalizers.Sequence([normalizers.Prepend("▁"),
                                                      normalizers.Replace(" ", "▁")])
                seq.append(decoders.Strip(content=" ", left=1))
            else:
                tk.pre_tokenizer = pre_tokenizers.Metaspace(
                    replacement="▁", prepend_scheme="first" if prefix else "never", split=False)
                if prefix:
                    seq.append(decoders.Strip(content=" ", left=1))
            tk.decoder = decoders.Sequence(seq)
            specials = [AddedToken(t, special=True, normalized=False)
                        for t, ty in zip(toks, types) if ty == CONTROL]
            self.tokens = toks
        else:
            raise ValueError(f"tokenizer type '{self.kind}' is not supported yet")

        if specials:
            tk.add_special_tokens(specials)
        users = [AddedToken(t, special=False, normalized=False)
                 for t, ty in zip(toks, types) if ty == USER_DEFINED]
        if users:
            tk.add_tokens(users)
        self.tk = tk

    # ------------------------------------------------------------------ api
    def encode(self, text: str, add_bos: bool | None = None) -> list[int]:
        ids = self.tk.encode(text, add_special_tokens=False).ids
        if (self.add_bos if add_bos is None else add_bos) and self.bos_id is not None:
            ids = [int(self.bos_id)] + ids
        return ids

    def decode(self, ids) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=False)

    def token_to_id(self, t: str):
        return self.tk.token_to_id(t)

    def chat_ids(self, system: str, turns: list) -> list[int]:
        """Sablon razgovora iz GGUF-a (jinja), kao apply_chat_template."""
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": r, "content": c} for r, c in turns]
        if self.chat_template:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def raise_exception(msg):
                raise ValueError(msg)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = raise_exception
            env.globals["strftime_now"] = lambda fmt: __import__("time").strftime(fmt)
            bos = self.tokens[self.bos_id] if self.bos_id is not None else ""
            eos = self.tokens[self.eos_id] if self.eos_id is not None else ""
            try:
                text = env.from_string(self.chat_template).render(
                    messages=msgs, add_generation_prompt=True, bos_token=bos,
                    eos_token=eos, enable_thinking=False)
            except Exception:
                text = None
            if text is not None:
                ids = self.encode(text, add_bos=False)
                if self.add_bos and (not ids or ids[0] != self.bos_id):
                    ids = [int(self.bos_id)] + ids
                return ids
        # bez sablona: obican tekst
        text = "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant:"
        return self.encode(text)

    def stop_ids(self) -> tuple:
        out = set()
        if self.eos_id is not None:
            out.add(int(self.eos_id))
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "</s>",
                  "<end_of_turn>"):
            i = self.tk.token_to_id(t)
            if i is not None:
                out.add(int(i))
        return tuple(out)
