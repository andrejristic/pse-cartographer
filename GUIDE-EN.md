# PSE Cartographer — Guide (English)

Version 0.4.0. Ordered from **most important to least important**. Read as far as you need and stop.

---

## 1. What this is, and the heart of it

You open a **local GGUF model (the patient)**, estimate its **J-space basis once**, then
**feed it prompts and watch its J-space move** — which concepts the model holds "in mind",
ready to say, at every layer — and at the end you get a **plain-language report** of what
happened, with concrete numbers.

Two roles (don't mix them):
- **spokesperson (working model)** — talks with you and writes the reports; you pick it on
  first launch (Claude, Gemini, Mistral, Groq and others via an API key; a local server such as
  Ollama; or a local GGUF, offline).
- **patient** — **always a local GGUF**, the model being dissected. Its J-space is what we map.
  It never leaves your computer; an online spokesperson only ever receives summaries of
  sealed results.

Inside the same window is **everything the PSE analyzer does** (spectral anatomy, precision
comparison, thought waterfall, concept dictionary, J-space map, perplexity ladder, dossier,
swarm & inbox), plus **Findings & Conclusions**, where the spokesperson reasons over all of it.

Everything runs on the CPU with the app's own numpy runtime — **no PyTorch, no GPU, nothing to
install afterwards**.

---

## 2. First launch: choose a spokesperson

A picker pops up. **Choose one** — without it you get the numbers and charts, but not the
conversation, the plain-language report, the expert notes, Findings & Conclusions or the active
cartographer. "Defer" works, and you can choose later (**Cartographer → Choose spokesperson…**).

- Enter the key, press **Fetch model list**, pick a model, **Test**, then **Use**.
- The choice is remembered in `~/.config/pse-cartographer/llm.json` (owner-only). Untick
  "Remember the key" if you don't want the key on disk.
- A **local GGUF** spokesperson is free and offline but slow on a CPU; small models (≤1B) write
  weak reports. ~3B or an online model is recommended for reports.

---

## 3. Getting to the heart (mapping J-space)

1. **★ Cartographer → Patient**: give the path to a local GGUF (disk or USB).
2. Optionally write your own prompts (one per line) and set the number of probe words.
3. **Analyze patient**. The chain runs by itself: spectral anatomy (once per model) → J-space
   basis for **every** layer → readout of each prompt → drift → report → seal. Progress is in the
   status bar. When it finishes, the app jumps to **Drift / Diff** — the heart:
   - bars = how much the held concepts move from prompt to prompt (0 = same, 1 = totally different);
   - red line = the ignition layer per prompt (where the model abruptly commits; −1 = none);
   - pick two prompts and **Diff A → B** to see exactly what turned on and off.
4. **Session** = what the model held in mind for each prompt. **Basis** = mapped layers and the
   probe words. **Report** = the plain-language report + a short expert note.
5. Every finished run is in the run picker at the top, so you can reopen it later.

**Experts** (mixture-of-experts models only): the second map next to J-space. At every layer the
router sends each token to a few experts out of many; this tab shows which experts carry a prompt
(heatmap: layer × expert), how much the chosen experts change from prompt to prompt (routing drift,
drawn next to the concept drift), and which concepts the model holds when a given expert is chosen.

**Active cartographer** (checkbox): the spokesperson invents the next prompt from the previous
readout — the working model steers the mapping instead of a fixed list.

---

## 4. How to read the numbers

- **Drift 0..1** — small: the model "thinks" alike about consecutive prompts; large: an abrupt
  change.
- **Ignition layer** — where the decomposition entropy drops sharply, i.e. the model commits.
- **Concept chips** — probe words (single tokens) whose lens directions build the state at that
  layer, by weight. Probe words are the content words of your prompts plus a built-in list of
  common concepts.
- **J-space share** — how much of the internal state is in this "ready to say" form; a few
  percent is normal.
- **Groups** — prompts grouped by which concepts they light up.

> Honestly: whether small models have a clean structure at all is an open question. The numbers
> are real; their interpretation is exploratory, and the reports say so.

---

## 5. Findings & Conclusions

Pick a model (and optionally a second one for **A vs B**), tick "next steps" if you want the
spokesperson to propose the next runs, and press **Write findings & conclusions**. Every finding
must cite the tag of the sealed result it comes from (e.g. `[83edeee8]`); a report that cites
nothing is marked as unreliable. The report is sealed like every other result and exports to
**Markdown, HTML, PDF** or a **zip with every underlying result**.

---

## 6. Everything else in the window

- **Model Card** — the model's anatomy, the capability semaphore for *this* machine and estimated
  times; buttons for every analysis (L1 spectral sweep, L2 thought waterfall, L4 J-space map,
  L3 concept dictionary, precision comparison, perplexity ladder).
- **Spectral Map** — heavy-tail exponent α per matrix (2–4 = trained, > 6 = still random-like);
  click a cell for its spectrum against the Marchenko–Pastur edge.
- **Precision** — what quantization changed, per role. **Perplexity** — the same text through
  several precisions of one model.
- **Thought Waterfall** — what the model would say if forced to speak at each layer.
- **Features** — the concept dictionary (sparse autoencoder) with examples where each concept fires.
- **J-space** — depth profile: sensory / workspace / motor regions and the ignition marker.
- **Dossier** — every sealed run in two registers (plain and mathematical); add notes
  (append-only — results are never edited, a correction is a new run); compare two models.
- **Swarm & Inbox** — the units at work (budget ring, engagement θ), recent runs and why a run
  was not accepted, and the inbox of decisions for you.
- **Conversation** (right) — ask the spokesperson about the sealed results.

---

## 7. Install

- **Linux (.deb)**: `sudo apt install ./pse-cartographer_0.4.0_amd64.deb`, then start
  *PSE Cartographer* from the menu or run `pse-cartographer`.
- **Windows (.exe)**: run `PSECartographer-Setup-0.4.0.exe` (no admin rights needed).
- **From source**: `cd app && ./setup.sh && ./run.sh` (Python 3.10+).

Check an installation at any time: `pse-cartographer --selftest` (no network, no models).

---

## 8. Supported models and limits

- Architectures: **Qwen2 / Qwen2.5, Qwen3, Llama 2 / 3 / 3.x, Mistral, Gemma / Gemma 2 / Gemma 3,
  Phi-2, Phi-3 / 3.5, StableLM 2, OLMo 2, Granite 3** (and other models llama.cpp writes as
  `llama`, e.g. SmolLM2, Falcon 3, TinyLlama), and the mixture-of-experts models **Granite 3 MoE**
  (1B-A400M, 3B-A800M) and **OLMoE**.
  Others are refused when the file is opened, never mid-run.
- Quantizations: F32, F16, BF16, Q8_0, Q4_0/Q4_1, Q5_0/Q5_1, Q2_K–Q6_K, IQ4_NL.
- Machine: a 16 GB computer without a GPU (even without AVX2) maps **0.5–3B** models comfortably
  (minutes for 0.5B, ~10 minutes for 1B, longer for 3B). Bigger models still work when they don't
  fit in RAM — layers are then decoded one at a time, slower; the Model Card says so up front.

---

## 9. Troubleshooting

- **"L2–L4 unavailable"** on the Model Card → the architecture or quantization type is not
  supported yet; the card names which.
- **No report text** → no spokesperson, or its key/model is missing (Cartographer menu).
- **Slow** → fewer probe words or a smaller model; the J-space basis is computed once and cached.
- Data (dossier, cached bases, results files): `~/.local/share/pse-cartographer`
  (File → Open data folder).

---

## 10. Licence

Free software under the **GNU GPL v3 or later**; see `LICENSE` and the attribution term in `NOTICE`.
© 2026 Andrej Ristic / Zerava Technologies Inc.
