# PSE Cartographer — Guide (English)

Ordered from **most important to least important**. Read as far as you need and stop.

---

## 1. What this is, and the heart of it

You enter a **local GGUF model (the patient)**, estimate its **J-space basis once**, then
**feed it prompts and watch its J-space move** — and at the end you get a **plain-language
report** explaining what happened, with concrete numbers.

Two roles (don't mix them):
- **spokesperson (working model)** — drives the conversation and writes the report; you pick
  it on launch (Claude / Gemini / Mistral / Groq via key, or a local GGUF via path).
- **patient** — **always a local GGUF**, the thing we dissect; its J-space is what we map.

Cartographer runs on the same engine as the existing PSE analyzer, so **inside it you can also
do everything that app did** (spectral map, ESD, waterfall, dossier, chat) plus the new
cartography flow.

---

## 2. First launch: choose a spokesperson

On first launch a picker pops up. **Choose one** — without it you only get raw numbers and
charts, and you lose: the conversation, the plain-language report, the short expert note on
each result, and the active cartographer. You may "Defer" and set it later (**Cartographer →
Choose spokesperson…**), but the app nudges you to choose, because the text is what makes the
results readable.

---

## 3. Getting to the heart (mapping J-space)

1. Open the **★ Cartographer** tab → **Patient** sub-tab.
2. **Give the path to a local GGUF** (disk or USB) — the *Browse…* button.
3. If torch is missing, the tab says so and offers an **"Install dissection support"** button
   (or copy-paste commands, see §7). Mapping needs torch; L0/L1 do not.
4. Click **Analyze patient**. When it finishes, the app jumps to **Drift / Diff** — the heart:
   - bars = how much the "held concepts" move from prompt to prompt (0 = same, 1 = totally different);
   - red line = the ignition layer per prompt (where the model abruptly commits);
   - pick two prompts and **Diff A → B** to see exactly what turned on/off.
5. The **Report** tab = the plain-language "what happened" + a short expert note.
6. The **Session** tab = per-prompt readouts (what the model "holds in mind"); **Basis** = which
   layers were mapped.

---

## 4. How to read the numbers

- **Drift 0..1** — small: the model "thinks" alike about similar prompts; large: an abrupt
  change. Spikes mark where the J-space shifts sharply.
- **Ignition layer** — where entropy drops sharply, i.e. the model "decides." Different prompts
  may ignite at different layers.
- **Concept chips** — vocabulary tokens the model holds as verbalizable at that layer.
- **Clusters** — prompts grouped by J-space signature (similar "modes of thinking").

> Honestly: whether small models even have a clean structure is an open question. You get real
> numbers, but interpretation is exploratory — the report is phrased with that uncertainty.

---

## 5. Active cartographer (optional)

On the Patient tab, check **"Active cartographer"**: the spokesperson invents the next prompt
from the previous readout — the working model steers the mapping instead of a fixed battery.

---

## 6. Everything the old app did is still here

Inside Cartographer are all the existing tabs: Model Card + capability semaphore, Spectral Map
(L1), ESD, Waterfall (L2), Features (L3), J-space (L4), Dossier, Swarm, chat. L0/L1 need neither
a key nor torch — they run offline.

---

## 7. Install (on the machine where you test)

```bash
cd "~/Desktop/PSE/PSE Cartographer/app"
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # GUI + L0/L1 + cloud spokespersons
python main.py
```
To **dissect a patient (J-space)** you need torch (CPU build — no GPU here):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece protobuf
```
The Patient tab's button runs exactly this. (The `pse_a` engine is imported from the sibling
`PSE J-space/app`; if you move it, set `PSE_ENGINE` to that path.)

---

## 8. Building the `.deb`

```bash
cd "~/Desktop/PSE/PSE Cartographer/packaging"
./build_deb.sh                       # bundles both Cartographer and the engine
sudo apt install ./pse-cartographer_0.1.0_all.deb
```
The `.deb` carries the engine (`pse_a`) plus its own `/opt` venv; the launcher sets
`PSE_ENGINE` automatically.

---

## 9. The one limitation (unchanged)

Full J-space on a 16 GB **no-GPU** box is realistic for **small patients (~0.5–3B)** in fp32;
7–8B wants a GPU or quantized loading. L0/L1 work on 7–8B too. The semaphore says so on the
Patient tab up front.

---

## 10. Troubleshooting

- **"Analysis unavailable"** → no torch/transformers → install (§7) or use the button.
- **Spokesperson silent / no report** → you didn't choose one, or the key/path is missing.
- **Model won't open / odd tensor names** → the family may not be in the engine's name map
  (`pse_a/core/naming.py`) or transformers won't load it from GGUF; try the safetensors version.
- **Slow** → smaller model; the basis is computed once (minutes), readouts are fast.
