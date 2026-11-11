# NetworkPancake: plan revizuit (un singur AI local în față, mai multe noduri în spate)

Data: 2026-11-10. Se bazează pe rapoartele `Plan NetworkPancake inferență distribuită.md` și `NetworkPancake adoptare tehnologie existentă.md`, plus discuția despre gândire paralelă, orchestrator-lucrători, straturi de orchestratori și portal.
Convenții: **[M]** măsurat de o sursă citată, **[D]** derivat din numere măsurate, **[E]** estimare sau prag provizoriu, **[U]** neverificat. Niciun număr de rețea pentru Strata nu a fost măsurat.

## Principiul

Utilizatorul vede un singur asistent și un singur endpoint compatibil OpenAI. În spate, un nod puternic ține modelul mare, celelalte noduri execută în paralel părți întregi de lucru, iar rezultatele sunt verificate. Nu tăiem un model între mașini la nivel de layer sau de expert (latența per layer depășește bugetul pe orice rețea obișnuită [D]).

**Regula de siguranță:** sistemul nu trebuie să fie niciodată mai prost decât cel mai bun nod singur. Routerul delegă doar sarcinile care se pot împărți și au câștig probabil; altfel răspunde direct nodul puternic.

## Decizii

- Bază: **Mesh-LLM** (rețea, descoperire, API, consolă), cu Strata ca plugin `openai_http`. Rezervă: **LocalAI** (backend gRPC, MIT). Licența Mesh e inconsistentă (Apache-2.0 în `LICENSE`, „MIT OR Apache-2.0" în Cargo): tratată ca Apache-2.0 până la lămurire.
- Nu forkăm Mesh-LLM, nu copiem cod din Cake (licență FAIR), nu facem Strata să vorbească llama.cpp RPC, nu construim expert-parallel peste rețea.
- Hardware vechi (iGPU Vega/Intel, AMD 400/500, Pascal): noduri llama.cpp (Vulkan/CPU). Telefoane: doar clienți. Strata nu are backend Vulkan/OpenCL și nici cod Android/iOS.
- Mesh privat, doar peer-uri de încredere (`--trust-policy require-owned`).

## Faze

| Fază | Livrabil | Criteriu de acceptare | Oprire |
|---|---|---|---|
| **0** (1-2 zile) | Calibrare: `ping` p50/p99, `iperf3`, `llama-bench` pe fiecare nod. A/B Strata față de llama.cpp ≥ v0.7 (`--moe-cache-mib`, baleiaj `--n-cpu-moe`) pe 1660 Ti și pe o placă de 12 GB, același GGUF, ≥ 8 runde alternate, cu RAM sub presiune ca variantă. Test Mesh cu două mașini (rutarea unui model găzduit de plugin) | Strata trece dacă limita inferioară a intervalului de 95% e cu ≥ 20% peste cel mai bun braț llama.cpp [E] | Dacă nu trece pe 6 GB, Strata se restrânge la 12-16 GB. Dacă nu trece nici pe 12 GB, integrarea Strata se oprește (planul rămâne pe llama.cpp) |
| **1** (~2 săptămâni [E]) | Endpoint unic OpenAI în LAN. Nodul puternic ține modelul mare, celelalte ține modele mici întregi. Router, consola Mesh, telefoane ca clienți. Plugin Strata (doar dacă faza 0 trece) | Un client OpenAI obișnuit funcționează. Supraviețuiește căderii unui nod. Viteza pe nod ≥ 90% din cea locală [E] | Dacă `GET /v1/models` și SSE din serverul Python al lui Strata nu merg cu sonda Mesh, shim în ~500-1000 de linii [E] |
| **1.5** | **Portal MVP**: joburi asincrone (trimitere, număr de job, poziție în coadă, stare, rezultat păstrat), coadă cu priorități și cote per utilizator, rezultat cu scor de încredere și dovezi. 3-5 utilizatori de încredere în LAN. Două tipuri de joburi: analiză de documente, revizuire de cod. Chei API, fără conturi publice | Metrici: joburi pe oră, timp de finalizare p50/p95, corectitudine, rata de eșec, echitate între utilizatori. Joburile reluate automat la căderea unui nod | Dacă rămâne doar coadă peste un singur nod puternic, se păstrează ca produs (cozi, cote, rezultate), dar nu se prezintă ca „creier distribuit" |
| **2** | **Orchestrator-lucrători**: nodul puternic planifică, lucrătorii execută subsarcini în paralel, nodul puternic verifică (teste, validare de schemă, două lucrători pe aceeași subsarcină pentru comparare) și îmbină. Sarcini: cod pe mai multe fișiere cu teste, extragere din ~50 de documente | ≥ 1,5× mai rapid decât modelul mare singur, la aceeași corectitudine (teste trecute) [E] | Sub 1,5×: oprim delegarea, păstrăm doar decodarea speculativă (dacă faza 5 o justifică) |
| **2b** | **Strat de orchestratori**: mai multe noduri puternice, fiecare cu lucrători proprii, deasupra rutarea Mesh și un planificator simplu. Contract uniform (sarcină → rezultat + încredere + dovezi), adâncime maximă 2, buget de tokeni, protecție la bucle, rutare lipicioasă a sesiunilor | Experiment de diversitate: două modele mari diferite pe ~100 de probleme verificabile; rata ambelor greșite comparată cu independența. Merită dacă e aproape de independență | Nu se construiește înainte ca faza 2 să treacă. Necesită ≥ 2-3 noduri puternice (azi există una de test) |
| **3** | **Gândire paralelă**: 10-200 de lanțuri independente cu modele mici, vot cu oprire prin cvorum, scor de încredere per lanț, nod puternic ca verificator. Probleme verificabile (matematică, cod cu teste) | Colectivul bate clar un singur model mai mare, la timp comparabil | Dacă nu, rămânem la debit agregat, fără pretenție de calitate |
| **4** | **Memorie comună**: cache de conversații al lui Strata cu rutare lipicioasă, regăsire din documente, rutare cu hashing rar comparată cu un router simplu | Timp până la primul token mai mic la a doua tură a aceleiași sesiuni | Dacă cache-ul local nu aduce câștig față de llama.cpp, rămâne doar regăsirea |
| **5** | **Internet**, doar după rezultate bune în LAN: Tailscale sau Headscale cu peer-uri de încredere; iroh doar dacă e nevoie. Decodare speculativă (ciorne pe lucrători, verificare pe nodul puternic), doar dacă faza 3 arată câștig | Prag provizoriu ≥ 3 tok/s pe flux la 2 etape [E], de recalibrat după faza 0 | Pe MoE, verificarea mai multor tokeni atinge mai mulți experți; câștigul poate fi nul în LAN |

## Ce așteptăm realist

- Debit agregat: aproximativ ×N cu N noduri puternice pentru cereri independente [D]. Dacă sistemul ajunge la ~100 tok/s agregat [E], asta înseamnă ~360.000 de tokeni pe oră, adică ~120 de joburi pe oră la ~3.000 de tokeni per job [D].
- Viteza unui singur răspuns = viteza unui singur nod. Un model de ~500B cu 100 tok/s pe un flux nu e realist: 100 tok/s înseamnă ~0,17 ms per layer la 60 de layere, iar rețelele obișnuite costă 0,2-0,5 ms (cablat) până la 20-100 ms (internet) per rundă [D].
- Calitatea e plafonată de cel mai bun model din sistem și de cât de bine verifică.

## Ce nu facem

- Nu promitem telefoane ca workeri (nu există niciun worker iOS/Android întreținut activ, conform căutărilor).
- Nu publicăm avantajul Strata pe 1660 Ti înainte de A/B-ul din faza 0 (rezultatul actual: 34,3 ± 14,1 tok/s, 3 rulări, fără comparație cu llama.cpp pe acea mașină).
- Nu deschidem portalul public sau rețeaua publică Mesh.

## Riscuri

- Confidențialitate: peer-urile văd activările și promptul. Date sensibile rămân pe nodurile proprii. Date ale terților intră sub regulile de protecție a datelor.
- Licență Mesh neclară; licență Cake nu se aplică (nu o folosim).
- Mesh: proiect de ~9 luni, ritm mare de schimbări, ABI v0, 3-6 oameni dominanți, fără audit de securitate [U pentru unele detalii].
- Rutarea prin plugin a fost testată live doar pe un singur nod (de verificat în faza 0).
- Nu s-a verificat dacă llama.cpp-ul fixat de Mesh conține deja PR #29887.
- Efortul de integrare e estimare, nu măsurat.
- Multe cifre pentru plăci de 4-6 GB vin din surse secundare, pe Q4_K_M, niciuna pe 1660 Ti.

## De făcut imediat, în ordine

1. Pasul 0 în paralel, fără cod Strata: testul Mesh cu două mașini.
2. A/B-ul same-machine (poarta go/no-go).
3. Doar dacă A/B trece: plugin-ul Strata.
4. Portal MVP.
