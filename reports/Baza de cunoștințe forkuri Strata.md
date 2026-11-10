# Baza de cunoștințe forkuri Strata

Data: 2026-11-10. Instrument: `tools/forks/fork_kb.py`, bază de date în `/home/dan/clone-of-d/strata-forks-kb/forks.sqlite`
(în afara repo-ului). Cum se rulează și se actualizează: README, secțiunea 3A.

## 1. Ce este

O bază de cunoștințe care se actualizează incremental: conține toate forkurile directe ale `Niko1221/Strata`, commit-urile
lor proprii (cele care nu sunt în upstream) și README-urile lor. Fiecare rulare nouă interoghează GitHub doar pentru
forkurile noi sau cele la care s-a făcut push între timp, și scrie un CSV numai cu commit-urile apărute de la rularea
precedentă. Pentru fiecare commit se știe la ce rulare a apărut prima dată (`first_seen_run`) și la ce rulare a fost văzut
ultima dată (`last_seen_run`), așa că se văd și commit-urile șterse prin force-push.

Scriptul pornește de la exportul CSV primit de la utilizator, cu aceste schimbări:

- o singură bază SQLite care se completează de la o rulare la alta, în loc de CSV-uri suprascrise;
- se sar forkurile cu același `pushed_at` (o actualizare durează sub un minut în loc de 70 de minute);
- README-urile se salvează o singură dată pentru fiecare conținut diferit (167 de README-uri distincte la 4.550 de forkuri);
- un fork fără README nu mai e eroare; nu se mai face o cerere în plus la sfârșitul fiecărei liste;
- un fork cu eroare nu e marcat ca scanat și se reîncearcă la rularea următoare;
- forkurile fără strămoș comun cu upstream-ul sunt tratate separat (secțiunea 2).

## 2. Problema găsită: istoricul upstream-ului a fost rescris

La prima rulare, 995 de forkuri au dat 404 la `compare`. Motivul, confirmat manual: GitHub răspunde „No common ancestor”.
Upstream-ul și-a rescris istoricul pe la începutul lui octombrie (nici tag-urile vechi, nici branch-urile `rc/0.1.30` și
`ci/checks-green` nu mai au istoric comun cu forkurile vechi), așa că 974 de forkuri create înainte nu mai pot fi comparate.

Soluția: pentru ele se iau commit-urile de pe branch-ul forkului datate după crearea forkului. Printre ele sunt și
commit-uri vechi din upstream pe care forkul le-a sincronizat; acestea sunt marcate `upstream_history = 1` dacă sunt
scrise de proprietarul upstream-ului sau apar în cel puțin două astfel de forkuri. Din 13.006 commit-uri aduse așa,
1.672 sunt muncă proprie a 66 de forkuri; restul sunt istoric vechi al upstream-ului. Au rămas 21 de forkuri cu 404 simplu
(„Not Found”, probabil repo-uri goale sau blocate), reîncercate la fiecare rulare.

Limită: un commit vechi din upstream sincronizat de un singur fork ar fi numărat greșit ca muncă proprie. E rar, pentru că
commit-urile din upstream apar de obicei în multe forkuri.

## 3. Cifre

| Măsură | Valoare |
| --- | ---: |
| Forkuri directe | 4.550 |
| Forkuri cu commit-uri proprii | 407 (340 divergente, 66 fără strămoș comun, 1 doar înainte) |
| Commit-uri proprii în total | 43.685 |
| Forkuri active din 2026-11-01 | 140 (2.957 commit-uri) |
| Forkuri active din 2026-11-08 | 60 (666 commit-uri) |
| Forkuri identice cu upstream-ul sau doar în urmă | 3.216 |

Activitatea pe săptămâni (commit-uri proprii / forkuri active): 1.682 / 29 (săpt. 38), 20.882 / 71 (39), 6.964 / 124 (40),
4.767 / 132 (41), 3.794 / 131 (42), 3.025 / 115 (43), 2.216 / 110 (44), 355 / 46 (45, parțial). Vârful a fost în
săptămâna de după lansare; de atunci numărul de forkuri active e stabil (110-130), dar fiecare face mai puține commit-uri.

Cele mai active forkuri din 2026-11-01 (commit-uri proprii, ultimul commit): Zutfen-LLC/Strata (220, 11-07),
zapabob/Strata (219, 11-10), rocktboy1982/pancake-strata (176, 11-09; forkul nostru), bsorensen110/Strata (173, 11-10),
DamirLukina88/D.L.-AI---Strata (112), eddoursul/Lamina (112), phoenixclyde/Strata (112), qwertie/Strataq (92),
svgop/Strata (92), serboox/strata (85), tyloryang/Strata (84), this-cute-world/Strata (78),
TheHubUser4860938/Strata-Mnemosyne (76), RandoTeam/4gbVramStrata (75), aktalascloud/strata-legacy-gpu (68),
lighttransport/Strata (65).

## 4. Teme din 2026-11-01 (după mesajele commit-urilor)

| Temă | Commit-uri | Forkuri |
| --- | ---: | ---: |
| Server, API, agenți | 953 | 75 |
| Cache de experți, offload, routing | 548 | 71 |
| Cuantizare | 372 | 55 |
| CUDA / NVIDIA | 284 | 54 |
| KV, atenție, context | 278 | 57 |
| GUI / aplicație | 255 | 36 |
| HIP / ROCm / RDNA | 238 | 40 |
| Decodare speculativă, MTP | 214 | 51 |
| Metal / Mac | 97 | 30 |
| Docker / pachete | 95 | 35 |
| DeepSeek | 89 | 29 |
| SYCL / Intel | 73 | 25 |
| Multi-GPU | 16 | 15 |
| Vulkan | 14 | 3 |

Clasificarea e după cuvinte cheie, deci aproximativă: un commit poate intra la mai multe teme.

## 5. Relevant pentru pancake (HIP / AMD, din 2026-11-01)

- **this-cute-world/Strata** (34 de commit-uri HIP): `STRATA_FUSE_SHGATE`, rândul de gate al expertului comun mutat în
  GEMV-ul router-ului, cu sigmoid și scalare fuzionate. Candidat de citit pentru calea de decodare.
- **dardeaw/strata-mxfp4-rdna**: kernele QSA wave32 exacte pentru decodare și reutilizare de fragmente MMVQ pe gfx12.
- **storm-ace/Strata**: cache K/V FP16 opțional pentru Qwen3.6 (`STRATA_Q36_KV16=1`).
- **rpcribari/Strata**, **Zutfen-LLC/Strata**: lucru pe gfx906 (prefetch pentru down, ocupanță, W4A4).
- **MyEcoria/Strata**: `strata-flint`, un motor nou pentru familie (Linux, HIP).
- **bsorensen110/Strata** (52): în mare parte merge-uri din upstream (0.2.13.x), de urmărit pentru ce aduce upstream-ul.

Nimic nu se adoptă fără A/B măsurat și, pentru ce e cu pierderi, fără poarta de calitate; orice preluare intră la
credite (README §9 + RESEARCH_AND_FINDINGS §4) în același commit.

## 6. Pașii următori

1. Rulare de actualizare periodică (de exemplu săptămânal): `GITHUB_TOKEN=$(gh auth token) python3 tools/forks/fork_kb.py`.
2. CSV-ul de commit-uri noi al fiecărei rulări devine intrarea pentru următorul lot de note despre forkuri.
3. De citit în cod: `STRATA_FUSE_SHGATE` (this-cute-world) și kernelele QSA wave32 (dardeaw), după terminarea
   benchmark-urilor din faza de test.
