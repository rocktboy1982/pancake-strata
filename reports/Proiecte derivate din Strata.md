# Proiecte derivate din Strata: ce a portat deja pancake și ce mai are de luat

Ecosistemul Strata a ajuns, în noiembrie 2026, la **42.317 de stele, 4.627 de fork-uri și 1.216 de issue-uri deschise**, cu un ritm de release pe care puține proiecte îl au: v0.2.9 (31.10), v0.2.10 (02.11), v0.2.11 (05.11), v0.2.12 (07.11) și **v0.2.13 (09.11, engine 0.1.59)**. Baza fork-ului nostru `pancake-strata` este `146df3c4` (v0.2.12, engine 0.1.58), deci upstream ne-a depășit cu un release în chiar ziua în care am scris raportul. Am verificat în jur de **30 de proiecte derivate cu muncă reală în ele**: 14 studiate în profunzime și deja exploatate de pancake (nivelul 1), plus aproximativ 20 de fork-uri și derivate descoperite sau reconfirmate în trecerea live de pe 09.11, din care nu am portat nimic încă (nivelul 2). La ele se adaugă ecosistemul de pachete de cuantizare de pe HuggingFace, un motor concurent închis (Infernix) și acoperirea din bloguri și Reddit, inclusiv o reacție de respingere a hype-ului și o dezmățire a unui scam crypto de către autor. Pe terenul lui propriu, **pancake se compară bine**: pe un singur RX 7900 XT măsurăm ~112 tok/s la decode după porturile de kernele Strata-rdna3, mai repede decât Strata-rdna3 însuși (109-119), decât Avalon (95-100) și decât Strata stock (75-77) în aceleași suite. Ce ne lipsește e lățimea: pipeline-uri dual-GPU, scalare multi-utilizator, backend-uri în afara CUDA/HIP și o serie de patch-uri upstream de kernele și de pornire care au ajuns sau s-au maturat după baza noastră.

Câteva convenții pentru tot raportul. „Fork” înseamnă orice proiect derivat din `Niko1221/Strata`, chiar dacă istoria a fost rescrisă. „Portat” trimite la ledger-ul din [MOD_PLAN.md][mp] (fazele 2-14, cu registrul „Not adopted” la liniile 1073-1286), iar starea ramurilor a fost verificată cu git pe 09.11: `port/faststrata-poly`, `exp/expert-prefetch`, `feat/grammar` și `feat/pattern-cache` sunt comasate în linia de lucru `port/rdna3-decode` (care nu e încă comasată în `main`); necompletate rămân `feat/disco` (+1), `feat/bfla-hip` (+7), `feat/bitdecoding-hip` (+2) și `feat/contextpilot` (+2). Cifrele măsurate de autori pe alt hardware sunt marcate ca atare; nimic din ce descrie nu a fost măsurat pe mașina noastră, dacă nu se spune explicit.

## 1. Rezumat executiv: ecosistemul și cele trei goluri mari

Din cele 4.627 de fork-uri, marea majoritate sunt clone pasive. Cele cu muncă reală se împart în trei niveluri, iar comparația cu pancake se rezumă la trei goluri:

1. **Un bug de corectitudine care ne atinge codul livrat.** PR-ul upstream [#5655][pr5655] arată că `route_prior` lasă să scape un ID de expert sentinelă și poate scrie în afara limitelor pe logits degenerate. Noi livrăm `STRATA_ROUTE_PRIOR` (preseturile `turbo`/`turbo-hip`) și `STRATA_ROUTE_PRIOR_DEPTH`, deci e și bug-ul nostru până la dovada contrarie. E singurul item din raport cu steagul „oprește-te și verifică întâi”.
2. **Pornirea și elasticitatea de memorie.** Repornirile noastre plătesc **28-65 s doar pentru citirea arenei de experți de 33 GiB** de pe ntfs3. Trei soluții independente există în ecosistem și nu am portat niciuna: [#3509][pr3509] (`STRATA_ARENA_REUSE=1`, atașarea unei arene partajate în locul re-copierii), citirile directe paralele de experți din Project Maya (O_DIRECT, 2-16 fire de citire, **+26% decode pe un PC sărac în RAM**) și sleep-in-place din basalt (commit `64167ba73`). Din aceeași familie e și skip-ul de miss după pondere (`MAYA_GLM_MISS_SKIP_W`, **+7-13% decode la KL 0,037-0,052** pe GLM).
3. **Dual-GPU și servire multi-utilizator.** Hardin22/Strata-DualGPU a demonstrat **29 → 161 tok/s** pe două GPU-uri cu 32 GB RAM, iar issue-ul upstream [#1352][i1352] arată că comasarea upstream e incompletă (upstream e încă de ~2x mai lent decât fork-ul pe un 4070 Ti Super + 5060 Ti). Cele trei PR-uri deschise ale lui cobra91 (#4522, #4534, #4549) sunt restul. havok2-htwo/StrataQ38FN arată 43 de patch-uri care duc o cutie de 4x3090 la **~578 tok/s cu 16 utilizatori (6,6x)** și taie re-citirile la comutarea de agenți de la **25 s la 1,5 s**. Totul stă în spatele fazei 12 (al doilea GPU), dar PR-urile nu ne așteaptă.

## 2. Nivelul 1: fork-urile mari deja studiate și portate în pancake

Pentru fiecare: URL, licență, ce am luat și ce a rămas neportat.

### 2.1 Jarson1408/FastStrata și simplexai-labs/PolyStrata (MIT)

- URL-uri: [Jarson1408/FastStrata][faststrata] (commit `df91582`) și [simplexai-labs/PolyStrata][polystrata] (commit `a82999fd`). Licență MIT. PolyStrata e părintele lui FastStrata; ambele duc motorul secundar `src/poly`.
- **Portat și comasat**: tot motorul `src/poly`, portat ca `strata-poly` pe ramura `port/faststrata-poly` (acum comasată), cu `--attn-split`, `--graph-experts`, `--elastic-prefill`, `--lend-staging`, `--stage-next`, `--suffix-drafts`, `--cpu-multi`, `--kv-q8` și `--draft-vocab` de la poly. Măsurat pe 3060: chat scurt **100,7 tok/s**, editare de cod **151,1 tok/s**, prompt de 41,8K în **35,8 s (71,7 tok/s)**, față de strata stock la 64,6 / 40,1 / 42,1 s.
- **Respins**: adâncimea adaptivă de MTP (regula de cost `Guesses` din poly o acoperă deja); kernelul fuzionat de FA pentru prompt în motorul principal (atenția e 6,7% dintr-un prompt de 42K; parcat până la profilul de prefill pe 7900 XT).
- **Neportat**: `--conversation-disk` (sesiuni automate pe disc) există doar în strata-poly; echivalentul din motorul principal (itemul 39 din MOD_PLAN, restaurare în ~0,5-1 s) nu e construit. Atenția de decode pe tensor-core (`--attn-split`) e doar CUDA; nu există cale WMMA pentru HIP/gfx1100.

### 2.2 nicejunjie/Strata-rdna3 (MIT)

- URL: [nicejunjie/Strata-rdna3][rdna3] (remote local `rdna3`).
- **Portat și comasat** (ramura `port/rdna3-decode`): familia de citiri hyper-connection pentru RDNA (+Q8, A8, INJ_UP, FUSED), GEMV-urile multi-rând wave32 (`RDNA_Q6K`, `MMVQ_WAVE`), `COMBINE_CPU`, `WAIT_FUSE`, `ROUTE_PUBLISH`. Decode-ul a urcat 87-101 → 104-108 (kernele) → **~112 tok/s** (fuziuni).
- **Respins**: top-k pentru context lung (= upstream `SELECT_LONG`), `PF_GROUP_RES` (= upstream `group_resident`), `PA_KLDS` (upstream l-a rescris ca `STRATA_PA_RPF`), `RDNA_PAIR` (neutru), profilul de experți și setările adapt ale rdna3 (ale noastre au măsurat mai bine: 121,2 față de 116,4), vocabularul de draft zh60k (92,5, mai rău) și megakernel-ul GDN de decode (0,0 ms pe un singur GPU; fereastra așteaptă rândurile CPU).
- **Neportat, totul parcat pe faza 12 (al doilea GPU)**: decode tensor-parallel (`src/core/tp.cpp`, `--tensor-parallel 2`; rdna3 a măsurat **129-170 tok/s pe 2x 7900 XTX**), transfer de experți între GPU-uri / RCCL (`peer_experts.cpp`, `peer_transport.inc`), parcare kfdq la idle (putere 98+91 W → 5+13 W), megakernel-ul GDN revizuit sub TP și re-măsurătoarea gfx12 (RDNA4) a porturilor tune pentru gfx11 (itemul 62).

### 2.3 jesdga95/basalt (MIT)

- URL: [jesdga95/basalt][basalt] (commit `a9468e5c`; fork cu istoria rescrisă din Strata `822a5c783`; remote local `basalt`).
- **Portat și comasat**: (a) motorul de gramatică, gata pe 08.11 pe `feat/grammar`: constructorii GBNF / JSON-schema / tool-call (`d64768699`), garda per cerere (`02e1006fe`), `Verifier::resample_row` (`c7813a6c3`), `--grammar on` în motor plus cheia `grammar=` per cerere, cu redesenare mascată în fereastra de verify ca MTP să continue să schițeze, și `serve/grammar.py`. Cost: 0,1-0,2% per fereastră; livrat în preseturile IQ2_XS. (b) Kernelele de head (cerere de owner, gata): `q5k_head.cu` portat pentru head-uri Q5_K (Q2_0/IQ3_XXS) și aceleași designuri scrise pentru IQ4_XS: GEMV exact de head plus alegere de draft în doi pași pe 2 biți.
- **Respins**: ecranul de draft-head pe 2 biți și K/V-append-beside-draft-query (pasul de draft e 2,5-3,2 ms dintr-o fereastră de 46-130 ms, maxim 1-2%); formatul de pachet `.basalt` (cere Blackwell sm_120).
- **Neportat**: `--idle-sleep` / sleep-in-place (commit `64167ba73`, ~815 de linii: renunță și reface înregistrarea copiei din RAM, reumple la trezire). Parcat, dar rămâne singurul item care închide golul `--vram-elastic`-refuză-cu-complement-pinned.

### 2.4 mw00/project-maya (MIT)

- URL: [mw00/project-maya][maya]. Licență MIT. Fork Strata blocat la engine 0.1.27, specializat pe Qwen.
- **Portat și comasat** (partea de server, secțiunea „Agent robustness from Project Maya” din MOD_PLAN): (a) **loop guard-ul**, `serve/loop_guard.py`, activ implicit: detecția buclelor de tokeni cu drop și penalizare de repetiție 1,15, închiderea gândirii la a 3-a buclă, oprirea pe răspuns, note și blocare pentru tool-uri repetate după 3 note, prăbușirea de prompt, nudges de tip keep-going, contoare `/metrics` plus Prometheus, teste în `serve/test_loop_guard.py`. (b) **Bugetul moale de gândire**, `reasoning_budget_grace` (închidere la sfârșit de frază sau rând; Maya #613). (c) **Tool-call-uri tolerante**: un tag `<parameter=` neînchis e malformat dar recuperabil, iar un apel malformat nu ridică niciodată excepție (#613/#706; teste în `serve/test_tolerant_tool_calls.py`).
- **Respins**: rularea lui Maya însuși (motorul lui Qwen e un snapshot 0.1.27 învechit, de 1,4-1,7x mai lent decât Strata curent, care deja pierde în fața pancake); reluarea gândirii după tokenul de end-of-turn (ambele tokenuri de end-of-turn Qwen sunt EOS pentru motor; cazul răspunsului gol e acoperit de #1053); DFlash2/DSpark/EXL3 (licență sau arhitectură); multi-GPU tensor-parallel; vision lending; dashboard-ul și contorul de economii.
- **Neportat**: (a) **citirile directe paralele de experți** pentru experții care nu încape în RAM: O_DIRECT cu 2-16 fire de citire sondate de drive (`MAYA_EXPERT_DIRECT` / `MAYA_EXPERT_STAGERS`); Maya a măsurat **+26% decode** pe un PC sărac în RAM. Itemul nostru înrudit 38 (loader direct de arenă pe Linux, estimare 28-65 s → 10-14 s la pornire) e planificat, dar neconstruit. (b) **Skip-ul de miss după pondere** (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode la KL 0,037-0,052 pe GLM); noi avem doar skip pe rang. (c) Itemuri mai mici: un sweep de pornire în stil `--calibrate`, auto-păstrarea drafter-ului, presetul Turbo ca profil de launcher, un memo de prompt (~6% din TTFT-ul de follow-up) și batching multi-cerere pe un singur GPU.

### 2.5 DevDock-AI/Speedy (MIT)

- URL: [DevDock-AI/Speedy][speedy]. Licență MIT.
- **Portat și comasat**: preseturile sm_86 ale lui Speedy ca configurație de start (itemul 8.1); opt-in-urile upstream pe care Speedy le-a măsurat: `STRATA_PF_GROUP_SCRATCH`/`_HC_REGION` (#4039), `STRATA_PREFILL_ROUTED_STREAM` (#4283), `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_PREFETCH` (itemul 8.2, făcut sau măsurat).
- **În lucru**: `STRATA_VERIFY_DEFER_CPU` (#2689/#2690, itemul 8.3) e codat, cea mai mare pârghie cu pierderi pe care o cunoaștem (până la **1,45x** per KTransformers), dar produce **logits non-finite pe HIP** cu kernelele noastre portate. Un bisect de 6 rulări e în coadă (PANCAKE_STATUS: Dev 100%, Test 10% după fix-ul de bisect ROUTE_PUBLISH). Baseline-ul `STRATA_PSI_SHED` e în coada grupei de stage 1.
- **Respins**: varianta INT8-IMMA de atenție a lui Speedy (atenția e 6,7% din fereastra motorului principal).
- **Neportat**: garda de rang pentru deferral, `STRATA_VERIFY_DEFER_CPU_TOP` (itemul 28), codată (Dev 100%) dar testată doar 5%; fix-ul HIP de non-finite le condiționează pe ambele.

### 2.6 JustVugg/colibri (Apache-2.0)

- URL: [JustVugg/colibri][colibri]. Licență Apache-2.0.
- **Portat (opt-in)**: rutarea conștientă de cache, `CACHE_ROUTE` → `STRATA_ROUTE_PRIOR` al nostru (itemul 8.9). Măsurat: lambda 0,5 dă **+25% decode la 2,89% mismatch**; livrat ca presetul opt-in `turbo`/`turbo-hip`. Ideea PILOT din colibri a devenit lookahead-ul de rutare, acum activ implicit pe GPU (`STRATA_LOOKAHEAD_GPU`, **+10% decode** pe experți mmap-uiți, aceleași alegeri).
- **Respins**: lambda 2,0 (9,90% mismatch, 8 evenimente cu marjă mare); predictorul de lookahead pe fire CPU (-13% decode, înlocuit de varianta GPU).
- **Neportat**: nimic la nivel de cod. Decizia de poartă de calitate pentru turbo-hip (P2a) e deschisă: presetul există, forțarea de tokeni a fost INCONCLUSIVĂ (+0,4 puncte), Test 30%.

### 2.7 eddoursul/StrataFast (MIT)

- URL: [eddoursul/StrataFast][stratafast]. Licență MIT.
- **Neportat**: `--prompt-split` (prompturi în două etape; itemul 8.7; **1,24-1,55x pe prompturi lungi** acolo). Blocat: cere un al doilea GPU, deci e parcat în faza 12. Nimic altceva din acest fork nu apare în notele noastre.

### 2.8 SensAI-PT/aviary-hy3 (Apache-2.0)

- URL: [SensAI-PT/aviary-hy3][aviary]. Licență Apache-2.0.
- **Portat**: nimic atribuit acestei surse nu apare în ledger ca item separat; ideile ei relevante au ajuns prin opt-in-urile upstream deja acoperite la Speedy (familiile `STRATA_PF_GROUP_*` și routed-stream).
- **Neportat**: nicio funcționalitate distinctivă rămasă neexaminată. Dacă apare un release nou, verifică directorul de kernele față de registrul „Not adopted” înainte să presupui suprapunerea.

### 2.9 JakeATX/llamAmpere (MIT)

- URL: [JakeATX/llamAmpere][llamampere]. Licență MIT.
- **Respins / închis**: adâncimea adaptivă de MTP (regula de cost din poly o acoperă); corpul de KV pe 8 biți în stil SJ-KVaRN și mb-MTP (VRAM-ul nu cumpără decode aici; draft-ul e 2-5% dintr-o fereastră).
- **Neportat**: nimic altceva atribuit acestei surse.

### 2.10 freriadoc/Avalon

- URL: [freriadoc/Avalon][avalon]. Motor pe bază de llama.cpp (nu un fork Strata) care urmărește upstream llama.cpp și concurează cu Strata pe aceleași modele; l-am studiat ca derivat al aceluiași ecosistem și țintă de benchmark.
- **Portat / adoptat**: tabela de tuning hipBLASLt gfx1100 `100401` în preseturile noastre 7900 XT (gata; **+38% prompt** provizoriu, Test 80%); hosted-MoE „finish together” (itemul 10.4) s-a dovedit deja prezent în Strata ca PcieAuto; flag-ul `--lookup-max-t 12` există (A/B în coadă, +1% zgomot până acum). Am fast-forward-at sursa Avalon de la `c24c03d` la `88313b6` (48 de commit-uri: arenă de RAM înregistrată, limita hosted-MoE din link rate, IQ3_KS MMA MMQ, fix-ul de tile pentru sparse-FA) și am construit-o, cu rollback binar la `Avalon/bin-c24c03d-rollback`; rebuild-ul și bench-ul așteaptă pentru că modelul lipsește (~20 GB insuficienți pe D).
- **Respins** (momente; lista completă e în „Avalon upgrades from upstream projects.md”): linia upstream de expert-cache #29887/#30637/#30966/#31888/#31286/#30594/#27861 (tier-ul nostru plus `--pcie-frac` fac deja asta; #30594 strică MTP verify); #31497 (contrazice cache-ul MTP partajat); KVA approximate prefill (decode -11%); PLE-in-RAM (57 GB de RAM nu țin modelul + tabela de 28,8 GB + rezerva); bias-ul de rezidență al router-ului ca implicit (bucle în sesiunile de agent); switch-urile gfx1100 ale Strata (atenție int8 WMMA, TSUM, layout-uri de experți: neutru sau negativ pe 7900); scorarea ClassLRU de memorie lungă (sub 0,5 puncte la replay); HIP graphs, Vulkan, turbo4/TurboQuant KV și restul, testate fără câștig.
- **Neportat**: A/B-ul `--lookup-max-t 16` și A/B-ul de confirmare hipBLASLt `100401` (ambele în coadă, ambele cer o mașină liniștită). Și un item pe care raportul Avalon l-a clasat „Adaugă acum” și pentru care nu există omolog pancake consemnat nicăieri: **llama.cpp #31751**, fix-ul de un singur pragma pentru register spills MMQ J=32 pe RDNA3, **+36% la batch-uri de 17-32 de tokeni pe gfx1100**, exact dimensiunea ferestrelor noastre de verify. Vezi secțiunea 4.4.

### 2.11 ewhacc/tandem.cpp și itlab-cl/tandem.cpp-amd

- URL-uri: [ewhacc/tandem.cpp][tandem] și [itlab-cl/tandem.cpp-amd][tandemamd] (varianta AMD).
- **Portat**: `STRATA_POOL_PREREAD` (workerii de pool inactivi pre-citesc experții CPU ai stratului următor). A/B-ul e încă în așteptare. `MTP_DEFER` era deja prezent.
- **Respins**: requant-ul Q4_K al MTP-head-ului (head-ul nostru e ~2 biți, 787 MB; requant-ul nu cumpără nimic).
- **Neportat**: nimic altceva identificat.

### 2.12 cutlerbenjamin1-cmd/qwen38-megakernel (AGPL)

- URL: [cutlerbenjamin1-cmd/qwen38-megakernel][q38mk]. Licență AGPL. Doar idei; nu copiem cod din cauza licenței.
- **Luat ca idei**: experimentele de vocabular de draft (full/en/zh60k) au măsurat toate plate sau mai rele; ideea de GDN replay era deja prezentă.
- **Neportat**: nimic. Licența AGPL face portarea de cod un non-obiectiv.

### 2.13 Surse mai mici audiate alături de nivelul 1

- **gpu-unlock**: măsurat, fără adopție. MMA cu sume FP32 a atins 49,7 TFLOPS, dar decode-ul neschimbat; un prompt de 5,3K a scăzut 7,6 → 6,0 s. Închis în scopuri de decode.
- **SGLang**: eșantionarea speculativă portată în motor (`STRATA_SPEC_PROB`, măsurată; calea implicită e la fel de rapidă sau mai rapidă); masca de tokeni GPU pentru gramatică e gata (faza 6); arborii de draft respinși (copieri de stare GDN plus treziri de experți; urmărim upstream #30931).
- **ContextPilot** (`feat/contextpilot`, necompletat, +2): codat (dedup de blocuri cu pierderi, opt-in, admitere exactă conștientă de cache), apoi respins: **1,6% economie** pe 40 de sesiuni reale opencode, iar reutilizarea non-prefix rupe starea GDN. Rămâne oprit.
- **BFLA (lucrarea CityU)** (`feat/bfla-hip`, necompletat, +7, doar HIP): atenție de prompt block-sparse, măsurată cap-coadă pe Qwen3.6 UD-Q4_K_XL: un prompt de 128K a urcat 386,9 → **1.219,8 tok/s sparse (3,15x)** → 771,2 tok/s dens; recall 48/48, KL 0,0014/0,0045. Decizia de comasare în așteptare (Test 10%).
- **BitDecoding** (`feat/bitdecoding-hip`, necompletat, +2): `STRATA_ATTN_WMMA`, atenție de decode pe matrix cores HIP gfx11, măsurată cu modelul plus forțarea de tokeni; parcată ca non-implicit (atenția e ~3% din decode; ~2-3% confirmat).
- **DISCO** (`feat/disco`, necompletat, +1): regulă de stop învățată pentru lanțul de draft MTP (opt-in). Funcționalitatea SpecFit de experți ratate (itemul 36) e atribuită acestei ramuri, neîncepută.
- **Seria de lucrări universitare** (rapoartele Stanford/MIT/Harvard/SUA/Europa/Asia/Canada/Pacific): plafonul de gândire per nivel e gata (`5dd620fb9`); măștile de sparsitate (Berkeley gate/TEAL/CATS/CHESS/SpAx) respinse offline (experții FF-640 din Flash-Next sunt denși: gate cos 0,844 la 70%); XAttention înlocuită de BFLA; itemul predictor-recall închis (acoperire 19-21%; promoțiile nu plătesc). În așteptare: oprirea învățată a gândirii (itemul 21), default-urile de efort (12), contoarele de loop-guard (33), semnalul de ruminare (34), testul de echivalență la restaurare (35), GDN chunk 64 (18), kernelul AVX2 IQ1_M (24, 0 pe Q2_0), A/B-ul variantei Coder (25) și constrângerea semantică oldString (26).

## 3. Nivelul 2: fork-uri noi și active, încă neportate în pancake

Găsite în trecerea live de pe 09.11 prin referințe încrucișate în issue-uri și PR-uri, sondaje pe profilurile autorilor, căutări de cod după simbolurile distinctive ale Strata (`STRATA_`, `native_mmvq`, `qsa_scores`, `combine_cpu`) și căutări web. Numărul de stele e din acea zi.

### 3.1 gputier/StrataGP

- URL: [gputier/StrataGP][stratagp] (9 stele).
- **Ce are**: 16 patch-uri de performanță/precizie împachetate sub `docs/perf/`, măsurate față de upstream `f1b1d96` pe un RTX 5090: **+5-15% decode și +42-49% prompt**.
- **De ce contează**: cel mai mare pachet unic de patch-uri de kernele gata făcute pe care nu l-am examinat. Câștigurile pe partea de prompt sunt partea interesantă, fiindcă calea noastră de prompt e cea care se degradează după contexte lungi.
- **Acțiune**: diff între cele 16 patch-uri și baza noastră; așteaptă-te la suprapunere cu clusterul bahooo22 (#5595/#5467/#5468) și cu lista noastră de respinse (atenția int8 WMMA). Portează ce supraviețuiește.

### 3.2 havok2-htwo/StrataQ38FN

- URL: [havok2-htwo/StrataQ38FN][strataq38fn] (1 stea).
- **Ce are**: o serie de **43 de patch-uri gata pentru upstream** pe v0.1.43, plus o fațadă compatibilă vLLM și un dashboard. Măsurat de autor: 4x3090 cu 16 utilizatori la **~578 tok/s (6,6x)**, RTX 5090 **+16% decode** single-user, iar re-citirile la comutarea de agenți tăiate de la **25 s la 1,5 s**.
- **De ce contează**: numărul de re-citiri la comutarea de agenți e exact durerea noastră (datele noastre opencode arată ~18% din timpul turelor în re-citiri). Partea multi-utilizator e irelevantă pentru serverul nostru cu un singur slot de azi, dar e cea mai bine documentată referință de scalare a servirii din ecosistem.
- **Acțiune**: citește întâi patch-ul de re-citiri; se poate combina cu checkpoint-urile noastre la granița de mesaj (itemul 11) și cu #5643 de mai jos.

### 3.3 bnkaiteam/Juud_engine

- URL: [bnkaiteam/Juud_engine][juud] (2 stele).
- **Ce are**: două ajustări de decode pe CPU: scurtarea spin-ului workerilor după lotul de experți și sărirea cuantizării de activări pentru tokenii rezidenți în GPU. Măsurat de autor **+9,6-15,6% decode** pe un RTX 4090 (post pe HF Forums).
- **De ce contează**: pool-ul nostru rulează deja la ~80% din lățimea de bandă DDR4 măsurată, deci scurtarea spin-ului e exact tipul de tăiere de overhead care mai poate apărea pe ferestrele de editare, iar skip-ul de activări pentru tokenii rezidenți e gratuit ori de câte ori tier-ul nostru ține experți rezidenți.
- **Acțiune**: portează ambele ca flag-uri opt-in; A/B față de default-urile din sweep-ul nostru de pool.

### 3.4 Hardin22/Strata-DualGPU și cobra91/Strata-DualGPU

- URL-uri: [Hardin22/Strata-DualGPU][dualgpu] (84 de stele) și fork-ul de continuare al lui cobra91.
- **Ce au**: pipeline-ul pe două GPU-uri pentru mașini cu 32 GB RAM: Swift IQ2_XS a urcat de la 29 la **161 tok/s** pe cod. Comasat parțial upstream: issue-ul [#1352][i1352] arată că upstream e încă de ~2x mai lent decât fork-ul pe un 4070 Ti Super + 5060 Ti, deci comasarea e incompletă. Piesele care au ajuns upstream: #848 (mod RAM rezident la un layer split plus `--vram-reserve-later-mib`, livrat în 0.1.40), #859 (ferestre de verify pipelinate, **+16% mediu, +30% copy-heavy**), #876/#905 (tier asincron). Efect combinat pe un 4060 Ti + 5080 cu 32 GB RAM: 29 → ~84+ tok/s.
- **Restul necompletat** (PR-urile deschise ale lui cobra91 spre upstream): [#4522][pr4522] (subțierea greutăților dense la `--layer-split auto`), [#4534][pr4534] (prețul unui split pipelinat după etapa lui cea mai lentă), [#4549][pr4549] (spațiul de draft-head ținut în afara cache-ului ultimei plăci).
- **De ce contează**: acesta e manualul de fază 12, deja scris și măsurat. VRAM-ul nostru de 20 GB ține rezidente doar 44% din perechile de experți; un al doilea GPU e singura pârghie care schimbă asta.
- **Acțiune**: când al doilea GPU ajunge, pornește din #4522/#4534/#4549 și din ferestrele de verify pipelinate #859, nu dintr-un design nou.

### 3.5 MistVVK/XeStrata și NeneNyann/Strata-intel-sycl-windows

- URL-uri: [MistVVK/XeStrata][xestrata] (334 de stele), [NeneNyann/Strata-intel-sycl-windows][nenenyann] (10 stele).
- **Ce au**: XeStrata e build-ul multi-vendor fără CUDA: SYCL/Level Zero cu căi XMX; `xe0.2.2-1` bate sycl 0.1.45.3 upstream pe un Arc Pro B50 (issue [#4506][i4506]). NeneNyann împachetează build-uri Windows SYCL cu runtime-uri oneAPI incluse și vision pe GPU.
- **De ce contează**: prioritate mică. Suntem HIP/CUDA. Relevant doar dacă vom ținti vreodată GPU-uri Intel, sau ca referință pentru idei de kernele SYCL care există și în formă HIP.
- **Acțiune**: doar urmărire.

### 3.6 lighttransport/Strata

- URL: [lighttransport/Strata][lighttransport] (20 de stele), ramura `glm53f`.
- **Ce are**: un backend nativ GGUF pentru GLM-5.3-Flash în interiorul Strata, ~29 tok/s pe 2x V100 și ~7,5 tok/s pe un Threadripper (CPU). A motivat cererea de funcționalitate #2067 și acum PR-ul upstream [#5545][pr5545] (calea de motor `strata-glm`, pipeline de cuantizare, aducerea multimodal).
- **De ce contează**: rulăm noi înșine GLM-5.3-Flash (sesiunile acestui raport rulează pe el). O cale de motor nativă ar înlocui fallback-ul folosit azi.
- **Acțiune**: urmărim #5545; când se comasează, rebazăm fork-ul pe el și re-măsurăm calea noastră de servire GLM.

### 3.7 JChen-318/gaudi2-single-card-strata

- URL: [JChen-318/gaudi2-single-card-strata][gaudi2] (10 stele).
- **Ce are**: un backend nativ Intel Gaudi2 (`hpu/`) cu kernele TPC proprii; Qwen3.8-Flash-Next IQ4_XS integral rezident; măsurat de autor **126,8 tok/s decode chat la 262K** de context (issue [#4322][i4322]).
- **De ce contează**: direct, deloc (nu avem hardware Gaudi). Pentru ecosistem contează ca dovadă că motorul Strata se portă dincolo de vendorii de GPU.
- **Acțiune**: doar urmărire.

### 3.8 CC-David-CC/Private_Context_COW

- URL: [CC-David-CC/Private_Context_COW][cow] (1 stea).
- **Ce are**: context privat copy-on-write, discutat în issue-ul upstream [#3865][i3865], autodescris ca „destul de stabil pentru testare”.
- **De ce contează**: conversațiile noastre parcate trăiesc doar în RAM și o repornire le pierde; contextul COW e una dintre ideile din familia de memorie (alături de prefixele host KV imuabile partajate din #5623) care ar putea tăia din cei 5,4-6,1 GiB cheltuiți pe 4 conversații parcate.
- **Acțiune**: citește thread-ul; compară cu itemul nostru 3 (store multi-stare) înainte de a construi ceva.

### 3.9 alesha-pro/Strata

- URL: [alesha-pro/Strata][alesha] (3 stele).
- **Ce are**: portul patch-ului de cuantizare trellis „S” pe 2 biți din Mirai la Strata 0.2.0 (PR-ul upstream [#3384][pr3384]), plus pachete abliterated pe HuggingFace.
- **De ce contează**: head-ul nostru e deja ~2 biți (787 MB), dar cuantizarea trellis pe corp e neexaminată. Dacă micșorează experții rezidenți în CPU la același KL, înmulțește tot ce face pool-ul legat de DDR4.
- **Acțiune**: evaluează cuantizarea offline (perplexitate/KL pe modelul nostru) înainte de orice muncă de kernele.

### 3.10 Unelte, jurnale de tuning și concurentul

- **Vop3p/dual-6900xt-flash-next-tuning**: [repo][vop3p] (3 stele). Jurnal de tuning de sine stătător pentru llama.cpp → Strata pe 2x RX 6900 XT. Același autor ca PR-urile #4005 și #5475 de mai jos, deci repo-ul e un carnețel util de teren pentru comportamentul HIP pe RDNA2/3. Urmărire.
- **xyzzing/kernelbench-hip** și **xyzzing/cupola**: [KernelBench portat pe HIP/ROCm][kernelbench] și [un supervizor de inferență auditabil pentru Radeon][cupola]. Același autor a deschis #5535 (atenție de prompt int8-WMMA pe gfx1100). Ideea de supervizor se suprapune cu metricile noastre de loop-guard; KernelBench-HIP e un harness la îndemână dacă începem să scriem kernele proprii. Urmărire.
- **lwYeo/llamaPeek**: [monitor de tavă Windows][lamapeek] pentru servere compatibile llama.cpp. Rulăm Linux; cosmetic. Urmărire.
- **Xolotl** (fostul Strata-DSV4.1): numit în sumarul ecosistemului de la stringers.io; nu verificat individual. Probabil o specializare DeepSeek V4.1. Urmărire.
- **Infernix** (2026.10.09.2): **nu e fork, e un motor concurent**. Artefact NVFP4 recipe-C plus un volum n-gram mmap-uit: măsurat de autor **+23% decode la 262K și +9% la 512K** față de Strata 0.1.41 pe un singur RTX 5090 (149,0 față de 120,6 tok/s), vision funcțional, pornire la rece ~28 s (prismix.dev, r/LocalLLaMA). Numărul de bătut dacă pretindem conducere la decode pe context lung. Urmărim notele de release.

### 3.11 Numite în acoperirea comunitară, nu verificate individual

PolyStrata (863 de stele conform sumarului; adaugă un al doilea motor pentru GLM/DeepSeek/Gemma), strata-nvfp4, strata-mlx (Mac), Strata-V100, eddoursul/Lamina, momoe, talus-public (Strix Halo), FastStrata. Tot în acoperire: squatter-ul SEO `stratallm.org` și dezmățirea scam-ului crypto de către autor pe X (19.10). Niciunul nu are o funcționalitate verificată care ne lipsește; verifică înainte de a cita.

### 3.12 Derivate externe (în afara GitHub)

- **Pachete de cuantizare HuggingFace**: [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata][hfmax] (issue [#4063][i4063], RTX PRO 5000 48GB); pachetele DavidAU Turbo-Brilliance GSQ-RCO (issue [#2841][i2841]); pachetele abliterated/trellis ale lui alesha-pro; ISTA-DASLab GSQ-RCO (Q2_0 → IQ3_S plus o variantă Coder cu 256/512 de experți, mmproj BF16 de 0,91 GB); Unsloth Dynamic v3.1 (re-încărcat pe 04.11; issue-ul [#5297][i5297] cere varianta UD-IQ2_XS cu ~9 GB mai mică; UD-IQ1_S nu se încarcă din cauza formatelor de experți amestecate într-un strat); Gyro-S/Gyro-M de la agentionai (issue-urile [#5176][i5176], [#5584][i5584]; Gyro-M TQ2_0 pe un RX 7900 XTX; [#3784][i3784] Gyro-S OOM pe Mac).
- **Comunitate/blog**: un quickstart dev.to pentru 64 GB (rosgluk, 07.11); o piesă de reacție la hype pe traictory.com; un ghid explainx.ai; postarea HF Forums despre Juud_engine.
- **Reddit**: postarea de origine de pe r/LocalLLaMA de la KnownAd4832 (65 tok/s pe 12 GB VRAM), thread-ul de backlash „stop the Strata spam” și datele de teren WDDM dual-5090 în comentariile la issue-ul #642.

## 4. Nivelul 3: upstream Niko1221/Strata (v0.2.12 → v0.2.13) și PR-urile valoroase

### 4.1 Ce a schimbat v0.2.13 (și ce înseamnă pentru baza noastră)

v0.2.13 a ieșit pe 09.11 (engine 0.1.59), la două zile după baza noastră v0.2.12 (`146df3c4`). Commit-urile de pe `main` din ultimele 48 de ore rulează de la `f33c8c305` la `8a9feaedd`:

- **Arhitectura shape-engine pentru Qwen4** (`f-0213-shape`): un model Qwen4 de formă nouă se livrează ca propria componentă de motor `strata-q4-<shape>`, lângă motorul Strata, selectată de model. `strata --geometry` tipărește geometria de kernele compilată (`flash_next` e implicit), iar verificarea de geometrie la instalare rulează față de un manifest semnat. E o schimbare de arhitectură: modelele viitoare pot înceta să fie GGUF-uri drop-in pentru motoarele vechi, ceea ce afectează modul în care facem rebase.
- **Default-urile DFlash2 per dispozitiv** (`e3bd5cfde`): Strix Halo și Blackwell cc 12.0 cu 48 GB sau mai mult primesc fereastră de draft 6 → 8, iar conf 0,55 pentru IQ3_S pe clasa aceea Blackwell. Măsurat de autor: Strix UD-Q4_K_XL x1,052, l620 IQ3_S x1,028. Noi am măsurat DFlash2 și l-am închis (115,9 tok/s pe chat scurt dar 84,3 după un prompt lung; un crash upstream pe QFN), deci contează doar dacă îl redeschidem.
- **Fix-ul Strix Halo G16** (`46ea3947e`): fără plaja Windows de rezervă VRAM de 2560 MiB pe dispozitivele HIP integrate (UMA); setup-ul scrie `--vram-reserve-mib 700` pe gfx1151 single-card. Start cu configurație veche 77 s, 20,5-24,3 tok/s. Nu e hardware-ul nostru, dar logica UMA poate atinge cod de setup partajat.
- Mai mici: unificarea etichetelor de fit („fits” / „may run slower”), un fix pentru folderul de date pe drive extern, NVFP4-QAD dimensionat după pachetul lui de 76,8 GB și o poartă de instalare feed-model (`tools/feed_model_install_gate.py`).

### 4.2 Registrul de PR-uri deschise, pe clustere (toate necompletate la 09.11)

**Kernele și MoE:**

| PR | Autor | Mecanism |
|---|---|---|
| [#5595][pr5595] | bahooo22 | q36: kernel de down-expert grupat pentru Q4_K (fmt 12) |
| [#5467][pr5467] | bahooo22 | q36: flash attention pe mma.m16n8k16 pentru prompt (K/V fp16) |
| [#5468][pr5468] | bahooo22 | q36: prefill layer-major care etajează experții unui strat o dată per N chunk-uri („super-chunk”) |
| [#5466][pr5466] | bahooo22 | q36: raportare need/free MiB la OOM de expert-buffer/staging |
| [#5535][pr5535] | xyzzing | atenție de prompt int8-WMMA pe RDNA3/gfx1100 (RX 7900 XTX), cu rețetă, validare și rezultate |
| [#4005][pr4005] | Vop3p | prefill: evenimente de copiere sparse opt-in + dedup de așteptări de slot; prompt Windows HIP **+36% @4K** (închis, necompletat) |
| [#5475][pr5475] | Vop3p | `STRATA_HC_PERSIST_Q8`: copieri int8 pentru citirea hyper-connection persistentă |
| [#4876][pr4876] | xrip | `STRATA_GDN_TC=2`: recurența GDN pe tensor-core pe sm_75, de 2,2x mai rapid |
| [#3277][pr3277] | c3c | extinde dispatch-ul de prefill native-IQ la SM75 (Turing) |
| [#4854][pr4854] | Keoian | rânduri native Q2_0 pe AVX1 (CPU-uri pre-AVX2) |
| [#4886][pr4886] | Spinoza0 | Metal: gate/up de experți IQ1_M în pachetele GSQ (instanțieri type-29 lipsă, #2537) |
| [#5518][pr5518] | rwkeyes | HIP: tablourile de staging int4 `alignas(16)` nu compilează (două site-uri din #778) |
| [#2392][pr2392] | skuridin | tabelă de tuning hipBLASLt pentru gfx1101 (RX 7800 XT, hipBLASLt 1.2.0); deschis din 14.10 |
| [#5540][pr5540] | Captain-Tripps | sycl: numără cache-urile helper/peer înainte de tier-ul adaptiv; decode în mod helper 32 → 61 tok/s |
| [#5562][pr5562] / [#5473][pr5473] | lwYeo / YamineRL | micșorarea layout-ului fuzionat întreabă calea fiecărui strat; stub-uri de link pentru `fused::want_dm_f16` |
| [#3925][pr3925] | schmittcn | q36: încărcarea GGUF-urilor cu `ssm_alpha`/`ssm_beta` K-quant sau IQ4_XS |

**Bug-ul de corectitudine (vezi P0 mai jos):**

- [#5655][pr5655] (WilliamStone): `route_prior` lasă să scape un ID de expert sentinelă și poate scrie în afara limitelor pe logits degenerate.

**Clusterul DeepSeek V4.1-Flash (R-Okauchi, toate deschise):**

- [#5471][pr5471]: GEMV-uri de decode Q4_K/Q5_K mai rapide, plus experți light la aceiași biți.
- [#5470][pr5470]: `DS4_PROF_WIN`, profil de decode al ferestrelor de verify ale drafter-ului.
- [#5472][pr5472]: DGX Spark `STRATA_DS4_DISK_SKIP_RANK`/`PROMPT_SKIP_RANK`.
- [#5453][pr5453]: DGX Spark `DS4_UMA_TIER=1`, un singur tier de experți pe care GPU-ul îl citește în loc.
- [#5566][pr5566] (yeyeyeason): cold store opt-in pentru arena de experți (jumătatea de citire-sursă din #5391).
- [#5596][pr5596] (AlexGabbia): `STRATA_DS4_CACHE_ROUTE`, bonus de selecție pentru experții deja în VRAM/RAM.

**Server și comportare de servire:**

- [#5516][pr5516] (rwkeyes): retune în loc (TUNE) plus rutarea cache-ului de conversație după forma traficului.
- [#5643][pr5643] (kelvincht): restaurarea reasoning-ului pe care clientul l-a scurs, ca următoarea tură să reutilizeze prefixul de prompt din cache.
- [#5623][pr5623] (shoucandanghehe): partajarea prefixelor host KV imuabile între sesiuni de streaming.
- [#3509][pr3509] (Jerry-Lee661): `STRATA_ARENA_REUSE=1`, atașarea unei arene partajate în locul re-copierii.
- [#4625][pr4625] (GoodSlopBadSlop): `STRATA_POOL_SPIN_US=0` înseamnă într-adevăr zero.
- [#5512][pr5512] (rwkeyes): sărirea blocului de gândire pe prompturi scurte cu răspuns direct (opt-in).
- [#3971][pr3971] (macintog): păstrarea tokenilor de gândire generați peste o tăiere de buget.
- [#5628][pr5628] (gener-vscode): streaming-ul argumentelor de tool la granițele JSON de tokeni.
- [#4777][pr4777] (code-martin): `route_prior` dinamic prin API/per cerere.
- [#5561][pr5561] (njy1991): read-ahead secvențial pe file-tier Linux (`STRATA_EXPERTS_SEQ_READAHEAD`).
- [#5593][pr5593] (krispgece): `STRATA_PROMPT_DIRECT=1` fără buget de RAM.
- [#4972][pr4972] (01xjw): default-urile file-tier Linux off când cache-urile GPU țin toți experții.
- [#5637][pr5637] / [#5437][pr5437]: inhibarea sleep-ului în timpul inferenței (Linux / cross-platform).
- [#5588][pr5588] (noahark): `--ple-gguf`, o tabelă PLE încărcată lateral câștigă rezoluția după nume.

**Familii de modele și platforme noi:**

- [#5517][pr5517] (archadrian): familia qwen3next, trunchi derivat din geometrie plus suport GDN grouped-head (lane-ul Groot identic bit cu bit).
- [#5545][pr5545] (zhuxm-ai): calea de motor GLM-5.3-Flash (`strata-glm`), pipeline de cuantizare, aducerea multimodal.
- [#5638][pr5638] (xiezhipeng-git): fine-tune Qwen3.8-Flash-Next pentru +0,40 GiB VRAM (antrenare de ordin zero).
- [#5592][pr5592] (DobricLilujun): Jetson AGX Thor (sm_110), cod mașină aarch64 CUDA.
- [#3607][pr3607] (demetree): suport nativ Windows Intel Arc SYCL (deschis din 21.10).
- [#5647][pr5647] / [#5644][pr5644] (kelvincht): arenă de experți elastică pe Metal (urmărește memoria liberă) plus pool-uri de K/V rezervate în locul alocării din start.
- [#4522][pr4522] / [#4534][pr4534] / [#4549][pr4549] (cobra91): piesele Strata-DualGPU necompletate (vezi 3.4).

**Issue-uri Qwen3.6 încă deschise upstream** (relevante fiindcă Qwen3.6-35B-A3B e unul din modelele noastre zilnice):

- [#5069][i5069]: Qwen3.6-35B-A3B nu are `--kv int8`/`--kv-resident` și o cale de decode pe context lung (Flash-Next le are).
- [#2464][i2464]: decode-ul Qwen3.6 cade cu contextul (V100: 109 → 14,6 tok/s la 114K).
- [#5609][i5609] (închis pe 08.11): blocul de MTP q36 se încarcă dar nu schițează niciodată. Fix-at upstream; verifică dacă baza noastră are fix-ul.
- [#5551][i5551]: q36 pe forma qwen35moe 19B-A3B, trei ziduri localizate.
- [#1402][i1402]: Qwen3.6 plus Ornith-1.5-35B-A3B ca linie de modele mici pentru plăci de 8-12 GB.

### 4.3 Ce a comasat deja upstream (valul de sfârșit de septembrie, pentru orientare)

#43 kernele de rând i-quant multi-token pe AVX2; #109 kernele de verify-window pe loturi (+~10% decode, identic bit cu bit); #21 cache KV Q4_0 cu rotația Hadamard FWHT-256; #54 experți tăiați GSQ-RCO-Coder (256/512); #8 cache de conversație; #62 workeri de pool hibrid-CPU (P-cores plus jumătate din E-cores); #67 împachetare BF16 OrcaRouter; #89 arenă de experți cu citire în vrac (split-ul ifstream de 4095 bytes MSVC); #81 accept runtime sm_80+; #122 endpoint-uri de descoperire llama.cpp; #118 urme de rutare; #107 raportarea tokenilor din promptul cache-uit; #126 Task Scheduler de 24x mai lent la încărcarea experților; #42 privilegiul MEM_LARGE_PAGES; #44 o sondă reală de bandă H2D pentru `pcie_frac`; #64 hard-fail pentru unealta de paritate. Majoritatea preced baza noastră sau sunt deja în ea; lista contează ca hartă a ceea ce proiectul comasează.

### 4.4 Singurul item llama.cpp

**llama.cpp #31751** ([PR][pr31751]): un fix de un singur pragma pentru register spills MMQ pe RDNA3, **+36% la batch-uri de verify de 17-32 de tokeni pe gfx1100**. Ferestrele noastre de verify au exact această dimensiune. Raportul Avalon l-a clasat „Adaugă acum” și niciun omolog pancake nu e consemnat nicăieri. Nimeni nu a verificat dacă kernelele MMQ ale Strata sau ale noastre au tiparul de spill. E cel mai ieftin item de kernele neexaminat din tot raportul.

## 5. Analiza prioritizată a golurilor: ce lipsește din pancake și ce luăm în continuare

Clasat după (impact x dovezi) în fiecare clasă de prioritate. Itemurile marcate „al nostru” trimit la numerele de itemi din MOD_PLAN.

### P0: Fix-uri de bug-uri și câștiguri fără risc (de făcut primele, în ordinea asta)

1. **Verifică #5655 în arborele nostru** (`route_prior`: scurgerea ID-ului de expert sentinelă + scriere OOB pe logits degenerate). Noi livrăm `STRATA_ROUTE_PRIOR` (turbo/turbo-hip) și `STRATA_ROUTE_PRIOR_DEPTH`. Dacă baza noastră `146df3c4` precede fix-ul, avem bug-ul. Citește PR-ul, reprodus pe logits degenerate, portează fix-ul indiferent de starea upstream.
2. **Verifică că fix-ul #5609 pentru MTP q36 e în baza noastră** (blocul de MTP se încarcă dar nu schițează; fix-at upstream pe 08.11). Un diff, cinci minute.
3. **Verificarea tiparului de spill din llama.cpp #31751**: caută în kernelele noastre MMQ tiparul J=32 de register spill; dacă e prezent, un pragma, **+36% la dimensiunile noastre de batch de verify pe gfx1100**. Niciodată evaluat.
4. **#4625** (`STRATA_POOL_SPIN_US=0` chiar zero): trivial, iar noi tune-ăm spin-ul de pool.
5. **#5466** (raportare need/free MiB la OOM de expert-buffer/staging): build-urile noastre de arenă declanșează systemd-oomd; mesaje de OOM mai bune ar fi economisit ore.
6. **#5518** (fix-ul de compilare HIP pentru staging int4 `alignas(16)`): două site-uri; ține build-urile HIP viitoare cinstite.

### P1: Câștiguri de decode și prefill pe un singur GPU, RDNA3/CUDA

1. **#4005** (Vop3p): evenimente de copiere sparse pentru prefill + dedup de așteptări de slot, **+36% prompt Windows HIP @4K**. Închis necompletat, deci codul trebuie resuscitat. Calea noastră de prompt e cea care se degradează după contexte lungi.
2. **Clusterul bahooo22 #5595/#5467/#5468**: kernel de down-expert grupat Q4_K, FA de prompt pe mma.m16n8k16, prefill layer-major super-chunk. Toate q36, toate pe modelul nostru zilnic.
3. **Cele 16 patch-uri StrataGP** (gputier): +5-15% decode, +42-49% prompt pe RTX 5090. Diff față de baza noastră; așteaptă-te la suprapunere cu clusterul bahooo22 și cu atenția int8-WMMA respinsă de noi.
4. **Cele două ajustări CPU Juud_engine**: spin scurtat după lotul de experți, skip de cuantizare a activărilor pentru tokenii rezidenți în GPU, +9,6-15,6% decode pe un 4090. Pool-ul nostru e legat de bandă, deci tăierile de overhead au forma potrivită.
5. **#5475** (`STRATA_HC_PERSIST_Q8`): copieri int8 pentru citirea hyper-connection persistentă. Verifică suprapunerea întâi: portul nostru rdna3 are deja citirea hyper-connection cu Q8; varianta persistentă poate fi nouă.
6. **#5535** (xyzzing): atenție de prompt int8-WMMA pe gfx1100, cu rețetă și validare. Am respins atenția int8-IMMA pentru motorul principal (atenția: 6,7% din fereastră), dar asta e pe partea de prompt, pe GPU-ul nostru exact, cu o rețetă de validare; ieftin de măsurat, așteptări mici.
7. **#2392** (skuridin): tabela de tuning hipBLASLt gfx1101. Avem deja tabela gfx1100 `100401`; asta extinde acoperirea la plăci de clasa 7800 XT.
8. **#5471** (R-Okauchi): GEMV-uri de decode Q4_K/Q5_K mai rapide. Calea noastră DeepSeek V4.1 beneficiază direct dacă formele de kernel se potrivesc.
9. **#3925** (schmittcn): încărcarea GGUF-urilor q36 cu `ssm_alpha`/`ssm_beta` K-quant sau IQ4_XS. Înlătură o clasă de pachete care nu se încarcă azi.

### P2: Pornire, memorie și câștiguri de server de agent

1. **#3509** (`STRATA_ARENA_REUSE=1`): atașarea unei arene de experți partajate între reporniri în locul re-copierii. Repornirile noastre plătesc 28-65 s pentru citirea arenei de 33 GiB pe ntfs3; asta atacă exact asta și se combină cu itemul 38 (loader direct de arenă pe Linux, est. 10-14 s) ca fie/fie.
2. **Citirile directe paralele de experți din Maya** (`MAYA_EXPERT_DIRECT`/`MAYA_EXPERT_STAGERS`, +26% decode pe un PC sărac în RAM) plus itemul nostru 38: decizia de pornire din 08.11 (oomd vs `STRATA_RAM_KEEP_GIB=20`) trebuie luată împreună cu asta.
3. **#5643** (kelvincht): restaurarea reasoning-ului scurs de client, ca următoarea tură să reutilizeze prefixul de prompt din cache. Datele noastre opencode arată ~18% din timpul turelor în re-citiri; e unul dintre puținele patch-uri de server care le atacă direct. Poartă-l pe testul nostru de echivalență la restaurare (itemul 35).
4. **#5516** (rwkeyes): TUNE în loc plus rutarea cache-ului de conversație după forma traficului. Am închis `/v1/tune` upstream ca de valoare mică, dar jumătatea de rutare de cache se potrivește cu munca noastră `STRATA_ROUTE_PRIOR`.
5. **Skip-ul de miss după pondere din Maya** (`MAYA_GLM_MISS_SKIP_W`, +7-13% decode la KL 0,037-0,052 pe GLM): noi avem doar skip pe rang. Poartă-l de calitate ca turbo-hip.
6. **#5566** (yeyeyeason): cold store opt-in pentru arena de experți. Elasticitate de memorie pentru arenă; se potrivește cu ideea de arenă elastică din #5647/#5644 (Metal, dar designul se portă).
7. **#5623** (shoucandanghehe): prefixe host KV imuabile partajate între sesiuni. Cele 4 conversații parcate ne costă 5,4-6,1 GiB; prefixele partajate sunt cea mai ieftină compresie disponibilă.
8. **`--conversation-disk` din FastStrata/poly pe motorul principal** (itemul 39): restaurare în ~0,5-1 s după repornire; FastStrata a măsurat o conversație de 70K reluată în 7,2 s în loc de 83 s.
9. **Sleep-in-place din basalt** (`64167ba73`): parcat, dar rămâne singurul item care închide golul `--vram-elastic`-refuză-cu-complement-pinned.
10. **#5561** (`STRATA_EXPERTS_SEQ_READAHEAD`): parcat în ledger-ul nostru (doar dacă experții fac fault de pe NVMe; arena elimină fault-urile). Revenire doar dacă itemul 38 alunecă.
11. **Itemuri de server mai mici**: #5593 (`STRATA_PROMPT_DIRECT=1` fără buget de RAM), #4972 (file-tier off implicit când cache-urile GPU țin toți experții), #5637/#5437 (inhibarea sleep-ului în inferență), #5588 (`--ple-gguf`), #4777 (`route_prior` dinamic per cerere), #3971 (tokeni de gândire păstrați peste tăierea de buget), #5628 (streaming-ul argumentelor de tool la granițele JSON), #5512 (skip de gândire pe prompturi scurte cu răspuns direct; l-am numit euristic, upstream e tot opt-in).

### P3: Funcționalități neportate din fork-urile cunoscute (cu porți de calitate sau parcate)

1. **`STRATA_VERIFY_DEFER_CPU` pe HIP** (portul Speedy, itemul 8.3): codat, până la 1,45x per KTransformers, blocat de logits non-finite; bisect-ul de 6 rulări e în coadă. Plus garda de rang `STRATA_VERIFY_DEFER_CPU_TOP` (itemul 28, 5% testat). Cea mai mare pârghie cu pierderi pe care o deținem.
2. **Decizia de poartă de calitate turbo-hip** (portul colibri, itemul 7): presetul există, forțarea de tokeni INCONCLUSIVĂ, Test 30%. Decizie de owner.
3. **Deciziile de comasare a ramurilor**: `feat/bfla-hip` (3,15x la prompt de 128K, recall 48/48), `feat/bitdecoding-hip` (~2-3%, parcat ca non-implicit), `feat/disco` (+SpecFit itemul 36, neînceput) și comasarea `port/rdna3-decode` → `main`. Toate decizii de owner, toate în așteptare.
4. **Cuantizarea trellis „S” din Mirai, portată de alesha-pro** (#3384): evaluează offline întâi (secțiunea 3.9).
5. **#4876** (`STRATA_GDN_TC=2`, recurența GDN pe tensor-core pe sm_75, 2,2x): doar CUDA; relevant pentru calea noastră 3060.
6. **#3277** (prefill native-IQ pe SM75) și **#4854** (rânduri Q2_0 pe AVX1): lățire de acoperire hardware; prioritate mică pentru noi.
7. **#5545 / #5517**: calea de motor GLM-5.3-Flash și trunchiul qwen3next derivat din geometrie. Urmărim ambele; schimbă modul în care servim GLM și familiile Qwen viitoare.
8. **Restul clusterului DeepSeek V4.1**: #5470 (profilerul `DS4_PROF_WIN`), #5596 (`DS4_CACHE_ROUTE`), #5453/#5472 (tier UMA și skip-rank pentru DGX Spark; ideea de skip-rank se poate generaliza la file-tier-ul nostru).
9. **Resturile din Maya**: sweep de pornire în stil `--calibrate`, auto-păstrarea drafter-ului, memo de prompt (~6% din TTFT-ul de follow-up), batching multi-cerere pe un singur GPU.
10. **Itemurile în așteptare din seria universitară** (ale noastre): oprirea învățată a gândirii (21), default-urile de efort (12), contoarele de loop-guard (33), semnalul de ruminare (34), testul de echivalență la restaurare (35), GDN chunk 64 (18), kernelul AVX2 IQ1_M (24), A/B-ul variantei Coder (25), constrângerea semantică oldString (26).

### P4: Dual-GPU și câștigurile fazei 12 (blocate de hardware, nu de cercetare)

1. **Decode-ul tensor-parallel din Strata-rdna3** (`src/core/tp.cpp`, `--tensor-parallel 2`; 129-170 tok/s pe 2x 7900 XTX), transferul de experți între GPU-uri / RCCL, parcarea kfdq la idle (98+91 W → 5+13 W) și megakernel-ul GDN revizuit sub TP.
2. **Piesele rămase din Strata-DualGPU**: #4522/#4534/#4549 ale lui cobra91, plus ferestrele de verify pipelinate din #859 (+16% mediu, +30% copy-heavy) dacă baza noastră nu le are. Issue-ul #1352 e avertismentul: comasările parțiale lasă upstream de ~2x mai lent decât fork-ul.
3. **`--prompt-split` din StrataFast** (1,24-1,55x pe prompturi lungi).
4. **Re-măsurătoarea gfx12 (RDNA4)** a porturilor tune pentru gfx11 (itemul 62) și DeepSeek V4.1 pe gfx1100.
5. **Urmărire**: numerele de context lung ale lui Infernix (+23% la 262K pe un 5090) ca reper pentru orice pretenție viitoare; XeStrata și #3607 dacă GPU-urile Intel contează vreodată; strata-mlx dacă apare un Mac.

### Ce NU luăm (deja respins, consemnat ca să nu re-judecăm)

Registrul complet de respinse e în MOD_PLAN.md, liniile 1073-1286. Momente: linia upstream de PR-uri de expert-cache (#29887 și rudele), KVA approximate prefill, PLE-in-RAM, DFlash2 pentru hardware-ul nostru, atenția int8-WMMA de decode ca implicit, ContextPilot (1,6% economie, rupe starea GDN), arborii de draft, formatul de pachet `.basalt`, NVFP4 (doar Blackwell), offload-ul de modele dense și toate itemurile multi-utilizator/batching până când serverul capătă un al doilea slot. Re-verificarea lor duplicează muncă; registrul e înregistrarea canonică.

## Anexă: surse și note de verificare

- **Auditul local**: MOD_PLAN.md (fazele 2-14, registrul „Not adopted” la liniile 1073-1286, secțiunea de robustețe Maya la 1287-1311, tabela 2-GPU la 1409-1424), PANCAKE_STATUS.md, cele cinci note de cercetare din `research_notes/Avalon upgrades from upstream projects/` și raportul `reports/Avalon upgrades from upstream projects.md`. Starea de comasare a ramurilor verificată cu git pe 09.11.
- **Trecerea live GitHub**: 09.11, prin API-ul de repo (fork-uri sortate după stele și după noutate), căutări de PR-uri și issue-uri, sondaje pe profilurile autorilor (cobra91, Vop3p, xyzzing, lwYeo, kelvincht, rwkeyes, bahooo22, R-Okauchi), căutări de cod după simbolurile distinctive ale Strata și căutări web (sumarul stringers.io, prismix.dev, r/LocalLLaMA, HF Forums, dev.to).
- **Avertismente de verificare**: (a) numărul de stele și starea PR-urilor sunt din 09.11 și vor aluneca; (b) numerele măsurate de autori (StrataGP, StrataQ38FN, Juud_engine, DualGPU, Infernix, Gaudi2, XeStrata) sunt pretenții de pe hardware pe care nu îl deținem, luate din README-uri, issue-uri și postări, nu reproduse de noi; (c) lista de fork-uri nu poate fi exhaustivă la 4.627 de fork-uri; căutările după simboluri au scos la iveală 10+ derivate dincolo de arborele de fork-uri, iar acoperirea comunitară numește repo-uri pe care nu le-am verificat individual (secțiunea 3.11); (d) sesiunea de inspecție a fork-urilor s-a încheiat la mijloc fără sumar final; descoperirile ei sunt acoperite de trecerea live rezumată aici, iar rezultatele intermediare confirmate (4.516 de fork-uri la ora 13:27, ținta downstream `rocktboy1982/pancake-strata`, 10+ descoperiri prin căutări de simboluri) sunt consistente cu numerele de mai sus.
- **Document însoțitor**: versiunea în engleză a acestui raport e la `pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md`. Ledger-ul de porturi și starea trăiesc în MOD_PLAN.md și PANCAKE_STATUS.md.

[mp]: file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md
[faststrata]: https://github.com/Jarson1408/FastStrata
[polystrata]: https://github.com/simplexai-labs/PolyStrata
[rdna3]: https://github.com/nicejunjie/Strata-rdna3
[basalt]: https://github.com/jesdga95/basalt
[maya]: https://github.com/mw00/project-maya
[speedy]: https://github.com/DevDock-AI/Speedy
[colibri]: https://github.com/JustVugg/colibri
[stratafast]: https://github.com/eddoursul/StrataFast
[aviary]: https://github.com/SensAI-PT/aviary-hy3
[llamampere]: https://github.com/JakeATX/llamAmpere
[avalon]: https://github.com/freriadoc/Avalon
[tandem]: https://github.com/ewhacc/tandem.cpp
[tandemamd]: https://github.com/itlab-cl/tandem.cpp-amd
[q38mk]: https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel
[stratagp]: https://github.com/gputier/StrataGP
[strataq38fn]: https://github.com/havok2-htwo/StrataQ38FN
[juud]: https://github.com/bnkaiteam/Juud_engine
[dualgpu]: https://github.com/Hardin22/Strata-DualGPU
[xestrata]: https://github.com/MistVVK/XeStrata
[nenenyann]: https://github.com/NeneNyann/Strata-intel-sycl-windows
[lighttransport]: https://github.com/lighttransport/Strata
[gaudi2]: https://github.com/JChen-318/gaudi2-single-card-strata
[cow]: https://github.com/CC-David-CC/Private_Context_COW
[alesha]: https://github.com/alesha-pro/Strata
[vop3p]: https://github.com/Vop3p/dual-6900xt-flash-next-tuning
[kernelbench]: https://github.com/xyzzing/kernelbench-hip
[cupola]: https://github.com/xyzzing/cupola
[lamapeek]: https://github.com/lwYeo/llamaPeek
[hfmax]: https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata
[i1352]: https://github.com/Niko1221/Strata/issues/1352
[i4506]: https://github.com/Niko1221/Strata/issues/4506
[i4322]: https://github.com/Niko1221/Strata/issues/4322
[i3865]: https://github.com/Niko1221/Strata/issues/3865
[i4063]: https://github.com/Niko1221/Strata/issues/4063
[i2841]: https://github.com/Niko1221/Strata/issues/2841
[i5297]: https://github.com/Niko1221/Strata/issues/5297
[i5176]: https://github.com/Niko1221/Strata/issues/5176
[i5584]: https://github.com/Niko1221/Strata/issues/5584
[i3784]: https://github.com/Niko1221/Strata/issues/3784
[i5069]: https://github.com/Niko1221/Strata/issues/5069
[i2464]: https://github.com/Niko1221/Strata/issues/2464
[i5609]: https://github.com/Niko1221/Strata/issues/5609
[i5551]: https://github.com/Niko1221/Strata/issues/5551
[i1402]: https://github.com/Niko1221/Strata/issues/1402
[pr5655]: https://github.com/Niko1221/Strata/pull/5655
[pr5535]: https://github.com/Niko1221/Strata/pull/5535
[pr4005]: https://github.com/Niko1221/Strata/pull/4005
[pr5475]: https://github.com/Niko1221/Strata/pull/5475
[pr5595]: https://github.com/Niko1221/Strata/pull/5595
[pr5467]: https://github.com/Niko1221/Strata/pull/5467
[pr5468]: https://github.com/Niko1221/Strata/pull/5468
[pr5466]: https://github.com/Niko1221/Strata/pull/5466
[pr3509]: https://github.com/Niko1221/Strata/pull/3509
[pr5643]: https://github.com/Niko1221/Strata/pull/5643
[pr5516]: https://github.com/Niko1221/Strata/pull/5516
[pr4625]: https://github.com/Niko1221/Strata/pull/4625
[pr5471]: https://github.com/Niko1221/Strata/pull/5471
[pr5470]: https://github.com/Niko1221/Strata/pull/5470
[pr5472]: https://github.com/Niko1221/Strata/pull/5472
[pr5453]: https://github.com/Niko1221/Strata/pull/5453
[pr5566]: https://github.com/Niko1221/Strata/pull/5566
[pr5596]: https://github.com/Niko1221/Strata/pull/5596
[pr5623]: https://github.com/Niko1221/Strata/pull/5623
[pr5512]: https://github.com/Niko1221/Strata/pull/5512
[pr3971]: https://github.com/Niko1221/Strata/pull/3971
[pr5628]: https://github.com/Niko1221/Strata/pull/5628
[pr4777]: https://github.com/Niko1221/Strata/pull/4777
[pr5561]: https://github.com/Niko1221/Strata/pull/5561
[pr5593]: https://github.com/Niko1221/Strata/pull/5593
[pr4972]: https://github.com/Niko1221/Strata/pull/4972
[pr5637]: https://github.com/Niko1221/Strata/pull/5637
[pr5437]: https://github.com/Niko1221/Strata/pull/5437
[pr5588]: https://github.com/Niko1221/Strata/pull/5588
[pr5517]: https://github.com/Niko1221/Strata/pull/5517
[pr5545]: https://github.com/Niko1221/Strata/pull/5545
[pr5638]: https://github.com/Niko1221/Strata/pull/5638
[pr5592]: https://github.com/Niko1221/Strata/pull/5592
[pr3607]: https://github.com/Niko1221/Strata/pull/3607
[pr5647]: https://github.com/Niko1221/Strata/pull/5647
[pr5644]: https://github.com/Niko1221/Strata/pull/5644
[pr4522]: https://github.com/Niko1221/Strata/pull/4522
[pr4534]: https://github.com/Niko1221/Strata/pull/4534
[pr4549]: https://github.com/Niko1221/Strata/pull/4549
[pr4876]: https://github.com/Niko1221/Strata/pull/4876
[pr3277]: https://github.com/Niko1221/Strata/pull/3277
[pr4854]: https://github.com/Niko1221/Strata/pull/4854
[pr4886]: https://github.com/Niko1221/Strata/pull/4886
[pr5518]: https://github.com/Niko1221/Strata/pull/5518
[pr2392]: https://github.com/Niko1221/Strata/pull/2392
[pr5540]: https://github.com/Niko1221/Strata/pull/5540
[pr5562]: https://github.com/Niko1221/Strata/pull/5562
[pr5473]: https://github.com/Niko1221/Strata/pull/5473
[pr3925]: https://github.com/Niko1221/Strata/pull/3925
[pr3384]: https://github.com/Niko1221/Strata/pull/3384
[pr31751]: https://github.com/ggml-org/llama.cpp/pull/31751
