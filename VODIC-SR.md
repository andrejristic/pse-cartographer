# PSE Cartographer — Vodič (srpski)

Poređano od **najvažnijeg ka manje važnom**. Čitaj koliko ti treba pa stani.

---

## 1. Šta je ovo i šta je srce

Uđeš u **lokalni GGUF model (pacijent)**, procijeniš mu **J-space bazu jednom**, pa ga
**hraniš promptovima i gledaš kako mu se J-space mijenja** — a na kraju dobiješ **narodski
raport** koji objašnjava šta se desilo, s konkretnim brojevima.

Dvije uloge (ne miješaj):
- **glasnogovornik (radni model)** — vodi razgovor i piše raport; biraš ga na startu
  (Claude / Gemini / Mistral / Groq preko ključa, ili lokalni GGUF preko putanje).
- **pacijent** — **uvijek lokalni GGUF** koji seciramo; njemu mapiramo J-space.

Cartographer koristi isti motor kao postojeći PSE analizator, pa **unutra možeš i sve što
je ta app umjela** (spektar, ESD, waterfall, dossier, chat) + novi kartografski tok.

---

## 2. Prvi ulazak: izbor glasnogovornika

Kad prvi put uđeš, iskoči izbor glasnogovornika. **Izaberi ga** — bez njega vidiš samo sirove
brojeve i grafove, a **nema**: razgovora, narodskog raporta, kratkih stručnih objašnjenja, ni
aktivnog kartografa. Možeš i "Odloži" pa podesiti kasnije (**Cartographer → Choose
spokesperson…**), ali app te gura da izabereš jer tekst je ono što rezultate čini čitljivim.

---

## 3. Kako doći do srca (mapiranje J-space)

1. Otvori tab **★ Cartographer** → pod-tab **Patient**.
2. **Daj putanju do lokalnog GGUF-a** (disk ili USB) — dugme *Browse…*.
3. Ako fali torch, tab to kaže i ponudi **dugme "Install dissection support"** (ili copy-paste
   komande, vidi §7). Bez torch-a mapiranje ne radi (L0/L1 rade).
4. Klikni **Analyze patient**. Kad završi, app skoči na **Drift / Diff** — srce:
   - stubići = koliko se "držani koncepti" pomjere od prompta do prompta (0 = isto, 1 = skroz drugo);
   - crvena linija = ignicioni sloj po promptu (gdje model naglo odluči);
   - izaberi dva prompta i **Diff A → B** da vidiš šta se tačno upalilo/ugasilo.
5. Tab **Report** = narodski raport "šta se desilo" + kratko stručno objašnjenje.
6. Tab **Session** = po-prompt očitanja (šta model "drži u glavi"). **Basis** = koji su slojevi
   mapirani.

---

## 4. Kako čitati brojeve

- **Drift 0..1** — mala vrijednost: model o sličnim promptovima "misli" slično; velika: naglo
  drugačije. Skokovi pokazuju gdje se J-space naglo mijenja.
- **Ignicioni sloj** — gdje entropija naglo padne, tj. model se "odluči". Različiti promptovi
  mogu paliti na različitim slojevima.
- **Koncept-čipovi** — vokabularni tokeni koje model drži kao izgovorljive na tom sloju.
- **Klasteri** — promptovi grupisani po J-space potpisu (slični "načini razmišljanja").

> Iskreno: da li mali modeli uopšte imaju čistu strukturu je otvoreno pitanje. Dobijaš prave
> brojeve, ali interpretacija je istraživačka — raport i piše s nesigurnošću.

---

## 5. Aktivni kartograf (opciono)

Na Patient tabu čekiraj **"Active cartographer"**: glasnogovornik sam smišlja sljedeći prompt
na osnovu prethodnog očitanja — radni model vodi mapiranje pacijenta umjesto fiksne baterije.

---

## 6. Sve što je stara app umjela — i dalje je tu

Unutar Cartographer-a su svi postojeći tabovi: Model Card + capability semafor, Spectral Map
(L1), ESD, Waterfall (L2), Features (L3), J-space (L4), Dossier, Swarm, chat. Za L0/L1 ne treba
ni ključ ni torch — radi offline.

---

## 7. Instalacija (na mašini gdje testiraš)

```bash
cd "~/Desktop/PSE/PSE Cartographer/app"
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # GUI + L0/L1 + cloud glasnogovornici
python main.py
```
Za **seciranje pacijenta (J-space)** treba torch (CPU verzija, jer nema grafičke):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece protobuf
```
Isto ovo radi i dugme na Patient tabu. (Motor `pse_a` se uvozi iz susjednog foldera
`PSE J-space/app`; ako ga premjestiš, postavi `PSE_ENGINE` na tu putanju.)

---

## 8. Pravljenje `.deb`

```bash
cd "~/Desktop/PSE/PSE Cartographer/packaging"
./build_deb.sh                       # spakuje i Cartographer i motor
sudo apt install ./pse-cartographer_0.1.0_all.deb
```
`.deb` nosi i motor (`pse_a`) i svoj `/opt` venv; launcher sam postavi `PSE_ENGINE`.

---

## 9. Ograničenje (nepromijenjeno)

Pun J-space na 16GB **bez grafičke** realan je za **male pacijente (~0.5–3B)** u fp32; 7–8B
traži GPU ili kvantovano učitavanje. L0/L1 rade i na 7–8B. Semafor to javi na Patient tabu.

---

## 10. Ako zapne

- **"Analysis unavailable"** → nema torch/transformers → instaliraj (§7) ili dugmetom.
- **Glasnogovornik ćuti / nema raporta** → nisi izabrao glasnogovornika ili fali ključ/putanja.
- **Model se ne otvara / čudna imena tenzora** → familija možda nije u mapi imena motora
  (`pse_a/core/naming.py`) ili je transformers ne učitava iz GGUF-a; probaj safetensors verziju.
- **Sporo** → mali model; baza se računa jednom (minute), očitavanja su brza.
