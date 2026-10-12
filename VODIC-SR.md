# PSE Cartographer — Vodič (srpski)

Verzija 0.4.0. Poređano od **najvažnijeg ka manje važnom**. Čitaj koliko ti treba pa stani.

---

## 1. Šta je ovo i šta je srce

Otvoriš **lokalni GGUF model (pacijent)**, procijeniš mu **J-space bazu jednom**, pa ga
**hraniš promptovima i gledaš kako mu se J-space mijenja** — koje pojmove model "drži u glavi",
spremne za izgovor, na svakom sloju — a na kraju dobiješ **narodski raport** šta se desilo, s
konkretnim brojevima.

Dvije uloge (ne miješaj):
- **glasnogovornik (radni model)** — razgovara s tobom i piše raporte; biraš ga pri prvom
  pokretanju (Claude, Gemini, Mistral, Groq i drugi preko ključa; lokalni server kao Ollama; ili
  lokalni GGUF, bez interneta).
- **pacijent** — **uvijek lokalni GGUF** koji seciramo; njemu mapiramo J-space. Ne napušta tvoj
  računar; online glasnogovornik dobija samo sažetke zapečaćenih rezultata.

U istom prozoru je **sve što radi PSE analizator** (spektralna anatomija, poređenje preciznosti,
waterfall misli, rječnik pojmova, mapa J-prostora, perplexity lestvica, dosije, swarm i inbox),
plus **Findings & Conclusions**, gdje glasnogovornik razmišlja nad svim tim.

Sve radi na procesoru, s vlastitim numpy motorom — **bez PyTorch-a, bez GPU-a, ništa se ne
doinstalira**.

---

## 2. Prvi ulazak: izbor glasnogovornika

Iskoči izbor. **Izaberi ga** — bez njega vidiš brojeve i grafove, ali nema razgovora, narodskog
raporta, stručnih napomena, Findings & Conclusions ni aktivnog kartografa. Može i "Defer", pa
kasnije (**Cartographer → Choose spokesperson…**).

- Unesi ključ, **Fetch model list**, izaberi model, **Test**, pa **Use**.
- Izbor se pamti u `~/.config/pse-cartographer/llm.json` (samo vlasnik). Skini kvačicu "Remember
  the key" ako ne želiš ključ na disku.
- **Lokalni GGUF** glasnogovornik je besplatan i offline, ali spor na procesoru; mali modeli (≤1B)
  pišu slabe raporte. Za raporte se preporučuje ~3B ili online model.

---

## 3. Kako doći do srca (mapiranje J-space)

1. **★ Cartographer → Patient**: daj putanju do lokalnog GGUF-a (disk ili USB).
2. Po želji upiši svoje promptove (jedan po redu) i broj riječi-sondi.
3. **Analyze patient**. Lanac ide sam: spektralna anatomija (jednom po modelu) → J-space baza za
   **svaki** sloj → očitavanje svakog prompta → drift → raport → pečat. Napredak je u statusnoj
   traci. Kad završi, app skoči na **Drift / Diff** — srce:
   - stubići = koliko se držani pojmovi pomjere od prompta do prompta (0 = isto, 1 = skroz drugo);
   - crvena linija = ignicioni sloj po promptu (gdje se model naglo odluči; −1 = nema);
   - izaberi dva prompta i **Diff A → B** da vidiš šta se tačno upalilo i ugasilo.
4. **Session** = šta je model držao u glavi za svaki prompt. **Basis** = mapirani slojevi i
   riječi-sonde. **Report** = narodski raport + kratka stručna napomena.
5. Svaki završen run je u biraču na vrhu, pa ga možeš kasnije ponovo otvoriti.

**Experts** (samo za mixture-of-experts modele): druga mapa pored J-prostora. Na svakom sloju
router šalje svaki token kod par eksperata od mnogih; ovaj tab pokazuje koji eksperti nose prompt
(mapa: sloj × ekspert), koliko se izabrani eksperti mijenjaju od prompta do prompta (drift
rutiranja, nacrtan pored drifta pojmova) i koje pojmove model drži kad je izabran neki ekspert.

**Aktivni kartograf** (kvačica): glasnogovornik smišlja sljedeći prompt iz prethodnog očitanja —
radni model vodi mapiranje umjesto fiksne liste.

---

## 4. Kako čitati brojeve

- **Drift 0..1** — malo: model o uzastopnim promptovima "misli" slično; veliko: nagla promjena.
- **Ignicioni sloj** — gdje entropija dekompozicije naglo padne, tj. model se odluči.
- **Čipovi pojmova** — riječi-sonde (jedan token) čiji lens-pravci grade stanje na tom sloju, po
  težini. Sonde su sadržajne riječi tvojih promptova plus ugrađena lista čestih pojmova.
- **Udio J-prostora** — koliko je unutrašnjeg stanja u obliku "spremno za izgovor"; par posto je
  normalno.
- **Grupe** — promptovi grupisani po tome koje pojmove pale.

> Iskreno: da li mali modeli uopšte imaju čistu strukturu je otvoreno pitanje. Brojevi su pravi;
> tumačenje je istraživačko, i raporti to kažu.

---

## 5. Findings & Conclusions

Izaberi model (i po želji drugi za **A vs B**), štikliraj "next steps" ako želiš da
glasnogovornik predloži sljedeće runove, i klikni **Write findings & conclusions**. Svaki nalaz
mora navesti oznaku zapečaćenog rezultata iz kog dolazi (npr. `[83edeee8]`); raport koji ništa
ne citira označen je kao nepouzdan. Raport se pečati kao i svaki drugi rezultat i izvozi u
**Markdown, HTML, PDF** ili **zip sa svim rezultatima ispod njega**.

---

## 6. Sve ostalo u prozoru

- **Model Card** — anatomija modela, semafor sposobnosti za *ovu* mašinu i procjene vremena;
  dugmad za svaku analizu (L1, L2, L4, L3, poređenje preciznosti, perplexity lestvica).
- **Spectral Map** — eksponent α po matrici (2–4 = istrenirano, > 6 = još kao šum); klik na
  ćeliju daje spektar naspram Marchenko–Pastur ivice.
- **Precision** — šta je kvantizacija promijenila, po ulozi. **Perplexity** — isti tekst kroz
  više preciznosti istog modela.
- **Thought Waterfall** — šta bi model rekao da mora progovoriti na svakom sloju.
- **Features** — rječnik pojmova (sparse autoencoder) s primjerima gdje se koji pojam pali.
- **J-space** — profil po dubini: čulni dio / radni prostor / izlaz i ignicioni marker.
- **Dossier** — svaki zapečaćen run u dva registra (narodski i matematički); bilješke (samo se
  dodaju — rezultati se nikad ne mijenjaju, ispravka je nov run); poređenje dva modela.
- **Swarm & Inbox** — jedinice na poslu (prsten budžeta, angažman θ), skorašnji runovi i zašto
  neki nije primljen, i inbox s odlukama za tebe.
- **Conversation** (desno) — pitaj glasnogovornika o zapečaćenim rezultatima.

---

## 7. Instalacija

- **Linux (.deb)**: `sudo apt install ./pse-cartographer_0.4.0_amd64.deb`, pa *PSE Cartographer*
  iz menija ili `pse-cartographer`.
- **Windows (.exe)**: pokreni `PSECartographer-Setup-0.4.0.exe` (ne treba admin).
- **Iz izvornog koda**: `cd app && ./setup.sh && ./run.sh` (Python 3.10+).

Provjera instalacije bilo kad: `pse-cartographer --selftest` (bez interneta, bez modela).

---

## 8. Podržani modeli i ograničenja

- Arhitekture: **Qwen2 / Qwen2.5, Qwen3, Llama 2 / 3 / 3.x, Mistral, Gemma / Gemma 2 / Gemma 3,
  Phi-2, Phi-3 / 3.5, StableLM 2, OLMo 2, Granite 3** (i ostali koje llama.cpp piše kao `llama`,
  npr. SmolLM2, Falcon 3, TinyLlama), i mixture-of-experts modeli **Granite 3 MoE** (1B-A400M,
  3B-A800M) i **OLMoE**. Ostali se odbiju pri otvaranju fajla, nikad usred rada.
- Kvantizacije: F32, F16, BF16, Q8_0, Q4_0/Q4_1, Q5_0/Q5_1, Q2_K–Q6_K, IQ4_NL.
- Mašina: računar od 16 GB bez GPU-a (čak i bez AVX2) udobno mapira modele od **0,5–3B**
  (minute za 0,5B, ~10 minuta za 1B, duže za 3B). Veći modeli rade i kad ne stanu u RAM —
  slojevi se tada dekodiraju jedan po jedan, sporije; Model Card to kaže unaprijed.

---

## 9. Kad nešto ne radi

- **"L2–L4 unavailable"** na Model Card-u → arhitektura ili tip kvantizacije još nije podržan;
  kartica kaže koji.
- **Nema teksta raporta** → nema glasnogovornika ili mu fali ključ/model (meni Cartographer).
- **Sporo** → manje riječi-sondi ili manji model; J-space baza se računa jednom i kešira.
- Podaci (dosije, keširane baze, fajlovi rezultata): `~/.local/share/pse-cartographer`
  (File → Open data folder).

---

## 10. Licenca

Slobodan softver pod **GNU GPL v3 ili novijom**; vidi `LICENSE` i uslov o autorstvu u `NOTICE`.
© 2026 Andrej Ristic / Zerava Technologies Inc.
