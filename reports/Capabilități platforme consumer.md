# O calibrare comună bate zece reglaje pe platformă

Pe platformele consumer, pancake se sprijină pe două numere: **cât costă un expert calculat pe CPU** și **cât costă un expert urcat pe GPU**. Raportul dintre ele variază de aproape zece ori de la o platformă la alta. Pe cutia de azi (5700G, PCIe 3.0, DDR4) e **42 µs față de 115 µs, adică 1:2,7**, și e singurul punct măsurat. Pe AM4 cu un Zen 3 fără iGPU pe B550/X570 ar fi ~1:1,2–1,8. Pe LGA1700/1851 cu DDR5 ar fi ~1:2–4. Pe un AM5 cu placă Gen5 ar fi ~1:1 sau chiar invers. Pe un Xeon W-3400 cu 8 canale ar fi ~1:4–7. Pe Strix Halo nu există promovare deloc. Toate acestea sunt estimări aritmetice, nimic nu a fost măsurat pe alt hardware. Concluzia este că **nicio valoare fixă de `--pcie-frac` sau de prag de reutilizare nu poate fi corectă peste tot**. Pancake are deja exemplul negativ în propriile date: PcieAuto calculează din raportul per expert o cotă de 0,20–0,25, dar pe 7900 XT cea mai bună valoare a fost 0 (+6%). Pe R9700, upstream a găsit 0,15 în loc de 0,55. De aceea munca comună vine prima: calibrarea la pornire pe timpul ferestrei, nu doar pe raportul per expert (item 55), valorile implicite după mărimea VRAM (61), loaderul direct (38), suprapunerea CPU/GPU, cache-ul per tură și porțile de calitate. Toate ajută pe orice platformă și se pot construi și verifica pe cutia de acum. Munca specifică vine după și cere aproape toată hardware pe care nu îl avem: afinitate P/E pe Intel hibrid (56), pool conștient de CCD pe Zen 4/5 cu două CCD-uri (58), AVX-512 pe ferestrele de verificare (59), presetul unificat Strix Halo (60), re-măsurarea pe gfx12 (62) și golurile de dispatch ISA (57). Pentru cutia proprie, faptele sunt simple. Un Zen 3 fără iGPU pe B550/X570 dublează legătura PCIe fără să schimbe ISA sau DRAM. AM5 aduce DDR5 și AVX-512, dar un CCD singur e plafonat la ~64 GB/s. Un GPU de 24/32 GB contează pe acest host mai mult decât generația PCIe. Rămâne o verificare deschisă: chipsetul plăcii nu e cunoscut.

Convenții. „Item N” trimite la lista din faza 11 a [MOD_PLAN][mp]. Itemii 55–62 au fost adăugați pe 2026-11-09 din această rundă. „Est.” marchează o estimare aritmetică. **Toate cifrele de timp de pe alte platforme decât 5700G + RX 7900 XT sunt estimări.** Singurele măsurători locale sunt 42 µs pentru calculul pe CPU (cifra planului), 53–55 µs pentru timpul pool-ului per expert ratat (PcieAuto), 115 µs pentru o promovare și 13,4 GB/s pentru proba PCIe. „Cere hardware” înseamnă că itemul nu poate fi măsurat pe cutia de acum. Notele de cercetare sunt în [notele rundei][notes].

## Matricea arată două axe: legătura PCIe și DRAM-ul văzut de CPU

### Platformele AMD

| Platformă | PCIe la GPU / NVMe de pe CPU | Memorie, canale, BW măsurat | L3 | ISA pentru pancake | Nuclee | RAM max | iGPU/NPU util? |
|---|---|---|---|---|---|---|---|
| **AM4 5700G (Zen 3 APU), azi** | **3.0 x16 / 3.0 x4**; chiar și lanes de chipset X570 coboară la Gen3 ([TPU](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)) | DDR4 2ch; ~33 GB/s văzut de pancake ([MOD_PLAN][mp]) | 16 MB (neverificat) | **AVX2 + FMA, fără AVX-512/AVX-VNNI** (`/proc/cpuinfo`) | 8C/16T, un CCX | 128 GB (neverificat) | Vega gfx90c doar Vulkan; nu |
| AM4 Zen 3 fără iGPU (5800X3D/5950X) pe B550/X570 | **4.0 x16 / 4.0 x4** ([AnandTech](https://www.anandtech.com/show/15850)) | DDR4 2ch; AIDA 50,7–54,7 GB/s ([hwcooling](https://www.hwcooling.net/wp-content/uploads/2022/04/amd-ryzen-7-5700x-g302.html)) | 32 MB/CCD; 96 MB la X3D | AVX2 | până la 16C/32T, 2 CCD | 128 GB | — |
| AM4 pe B450/X470 | 3.0 x16 / 3.0 x4 | ca mai sus | ca mai sus | AVX2 | ca mai sus | 128 GB | — |
| AM5 Zen 4 (7000, 7x00X3D) | 5.0 x16 pe X670E/B650E/X870(E), altfel 4.0 ([Puget](https://pugetsystems.com/labs/articles/amd-x870e-vs-x870-vs-x670e-vs-x670-vs-b650e-vs-b650)); NVMe 5.0 pe plăci E/B850 | DDR5 2ch; **~64 GB/s citire per CCD** (legătura IF de 32 B/ciclu) ([C&C](https://old.chipsandcheese.com/?p=11867)) | 32 MB/CCD; 96 MB X3D | **AVX-512** F/BW/VL/DQ/VNNI/VBMI/BF16, double-pumped pe 256 biți ([numberworld](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)) | până la 16C/32T, 2 CCD | 192–256 GB (4×64 GB, ~6000 MT/s) ([TweakTown](https://www.tweaktown.com/news/105825/msi-releases-amd-agesa-1-2-0-3e-bios-officially-supports-4-64gb-sticks-up-to-256gb-ram/index.html)) | — |
| AM5 8700G/8600G (Phoenix 1) | **4.0 x8** / 4.0 x4+x4 ([AnandTech](https://www.anandtech.com/show/21242)) | DDR5 2ch | n/g | AVX-512 (Zen 4, presupus) | 8C | ca AM5 | 780M gfx1103: absent din matricea ROCm |
| AM5 8500G/8300G (Phoenix 2) | **4.0 x4** / 4.0 x4+x2 ([TPU](https://techpowerup.com/317997/amds-phoenix-1-and-phoenix-2-apus-differ-in-pcie-lane-count-affects-nvme-drive-performance-and-gpu-pcie-lane-count)) | DDR5 | — | AVX-512 (presupus) | Zen 4 + 4c | ca AM5 | nu |
| AM5 Zen 5 (9000, 9x00X3D) | după chipset | DDR5 2ch; ~60 GB/s AIDA la DDR5-5000 pe 16C; tot 64 GB/s/CCD ([numberworld](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)) | 96 MB 9800X3D; 128 MB 9950X3D (V-Cache pe un CCD); X3D2 pe ambele ([StorageReview](https://www.storagereview.com/review/amd-ryzen-9-9950x3d2-dual-edition-review-3d-v-cache-on-both-ccds)) | **AVX-512 pe 512 biți reali + AVX-VNNI** ([Tom's](https://www.tomshardware.com/pc-components/cpus/amd-zen-5-support-arrives-with-five-new-instructions-gnu-compiler-collection-posted)) | până la 16C/32T, 2 CCD | 256 GB | — |
| Strix Point (HX 370) | 4.0 (număr de lanes negăsit) | LPDDR5X-7500; GPU 96 GB/s ([C&C](https://old.chipsandcheese.com/2024/08/24/amds-radeon-890m-strix-points-bigger-igpu/)) | 24 MB | AVX-512 pe hardware de 256 biți | 4 Zen 5 + 8 Zen 5c | până la 256 GB | gfx1150 în ROCm 7.11 preview |
| **Strix Halo (AI Max+ 395)** | 4.0, 16 lanes (PRO) | **LPDDR5X-8000 256-bit, 256 GB/s teoretic, ~103 GB/s dintr-un CCD** ([C&C](https://old.chipsandcheese.com/?p=37437)) | 64 MB | AVX-512 (Zen 5; lățimea neconfirmată) | 16C/32T | **128 GB unificați** | **gfx1151 suportat în ROCm** ([ROCm](https://rocm.docs.amd.com/en/docs-7.9.0/install/compatibility-matrix.html)); NPU 50 TOPS nefolosit |

### Platformele Intel

| Platformă | PCIe la GPU / NVMe de pe CPU / DMI | Memorie, BW teoretic/măsurat | L2 / L3 | ISA pentru pancake | Nuclee | RAM max |
|---|---|---|---|---|---|---|
| LGA1151 (8/9 gen) | 3.0 x16 / **niciunul, NVMe prin DMI 3.0 x4 (~3,9 GB/s)** | 2× DDR4-2666; 42,7 / ? | 256 KB / ~16 MB | AVX2 | omogene, HT | 128 GB |
| LGA1200 10 gen | 3.0 x16 / niciunul / DMI 3.0 x4 ([TPU](https://techpowerup.com/review/intel-core-i9-10900k/3.html)) | 2× DDR4-2933; 46,9 / ? | 256 KB / 20 MB | AVX2 | omogene, HT | 128 GB |
| LGA1200 11 gen (Rocket Lake) | **4.0 x16 / 4.0 x4** / DMI x8 ([AnandTech](https://www.anandtech.com/show/16523)) | 2× DDR4-3200 | ? | **AVX2 + AVX-512 cu VBMI, deci trece `cpu_avx512_ok`** | omogene, HT | 128 GB |
| LGA1700 (12–14 gen) | 5.0 x16 / 4.0 x4 / DMI 4.0 x8 | DDR5-5600 ~82 GB/s, DDR4 ~50 GB/s ([CapFrameX](https://www.capframex.com/tests/Alder%20Lake-S%2012900K%20is%20the%20new%20Gaming%20King)) | 2 MB/P / 36 MB | AVX2 + **AVX-VNNI**, fără AVX-512 (dezactivat din fabrică) ([HotHardware](https://hothardware.com/news/intel-fusing-avx-512-alder-lake-silicon)) | 8P(HT) + până la 16E | 192 GB |
| LGA1851 (Core Ultra 200S) | 5.0 x16 / 5.0 x4 (latență mai mare) + 4.0 x4 / DMI 4.0 x8 | DDR5: 76 GB/s la 5600, ~90 la 6000, ~118 la 7800 ([Igor's Lab](https://www.igorslab.de/en/arrow-lake-ddr5-detail-memory-clock-uncore-timings-core-ultra-7-265k-two-platforms/)) | 3 MB/P / 36 MB ([C&C](https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the)) | AVX2 + AVX-VNNI | **8P + 16E, fără SMT** | 192 GB (?) |
| Xeon W-3400 (W790) | 5.0, 112 lanes | **8× DDR5-4800 RDIMM, 307 GB/s teoretic** ([AnandTech](https://anandtech.com/show/18741)) | 2 MB / 105 MB | AVX-512 complet + AMX | omogene, HT | 4 TB |
| Xeon E5 v3/v4 (folosit) | 3.0 / lanes CPU | 4× DDR4-2400, ~77 teoretic; 119 GB/s STREAM pe 2 socluri ([AnandTech](https://www.anandtech.com/show/10158/the-intel-xeon-e5-v4-review/9)) | 256 KB / până la 55 MB | AVX2 | omogene, HT | 768 GB+ |

Pe Intel, iGPU-urile nu folosesc pancake. UHD 630 a rămas pe compute-runtime „legacy” înghețat ([Fedora](https://www.fedoraproject.org/wiki/Changes/IntelCompute2025)). Xe-LP și cele 4 nuclee Xe din Arrow Lake nu au DPAS int8, iar portul SYCL nu are planificare pentru memorie partajată ([INTEL_ARC.md][arc]). NPU-ul de 13 TOPS nu are cod în repo.

### GPU-urile consumer și acoperirea perechilor de experți

Acoperirea se calculează pentru Qwen3.8-Flash-Next IQ2_XS: 24.576 de perechi, ~1,4 MB fiecare. Punctul de plecare e singurul măsurat, 20 GB → 10.909 perechi (44%), cu ~4,7 GB de overhead non-expert în presetul pancake. Restul rândurilor sunt est. liniare ([AMD_HIP.md][amdhip]; [ROCm specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html); [whitepaper NVIDIA](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf); [Intel ARK B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)).

| Card | Țintă | VRAM | BW GB/s | Lățime card | Instrucțiuni matrice | Cale pancake | Perechi est. |
|---|---|---|---|---|---|---|---|
| RX 6800/6900 XT | gfx1030 | 16 GB | ~512 | 4.0 x16 | niciuna (doar `sdot4`) | HIP generic, pachet netestat | ~8.100 (33%) |
| RX 7600 / 7600 XT | gfx1102 | 8 / 16 GB | ~288 | **4.0 x8** | WMMA FP16/BF16/IU8/IU4 | HIP gfx11, „raportat funcțional” | ~2.400 / ~8.100 |
| RX 7800 XT | gfx1101 | 16 GB | ~624 | 4.0 x16 | WMMA RDNA3 | HIP gfx11, validat de owner | ~8.100 (33%) |
| **RX 7900 XT (azi)** | gfx1100 | **20 GB** | **800** | 4.0 x16, rulează la 3.0 | WMMA RDNA3 | **calea reglată pancake** | **10.909 (44%, măsurat)** |
| RX 7900 XTX | gfx1100 | 24 GB | ~960 | 4.0 x16 | WMMA RDNA3 | același binar și preseturi | ~13.800 (56%) |
| PRO W7900 | gfx1100 | 48 GB | ~864 | 4.0 x16 | WMMA RDNA3 | același binar (netestat) | toate 24.576 |
| RX 9060 XT | gfx1200 | 16 GB | ~320 | **5.0 x16** | WMMA + FP8/BF8 + SWMMAC 4:2 | HIP gfx12 | ~8.100 |
| RX 9070 XT | gfx1201 | 16 GB | 640 | 5.0 x16 ([Guru3D](https://www.guru3d.com/review/review-radeon-rx-9070-xt-reference-edition/)) | ca mai sus | HIP gfx12, testat upstream | ~8.100 |
| AI PRO R9700 | gfx1201 | 32 GB | ~640 | 5.0 x16 | ca mai sus | HIP gfx12, mașina de dezvoltare upstream | ~19.500 (79%) |
| RTX 3060 / 3090 | sm_86 | 12 / 24 GB | 360 / 936 | 4.0 x16 | TC INT8/INT4, 2:4 | CUDA (fostul card) | ~5.200 / ~13.800 |
| RTX 4060 Ti / 4090 | sm_89 | 8–16 / 24 GB | ~288 / 1008 | **4.0 x8** / x16 | + FP8 | CUDA | ~2.400–8.100 / ~13.800 |
| RTX 5060 Ti / 5090 | sm_120 | 16 / 32 GB | ~448 / 1792 | **5.0 x8** / 5.0 x16 ([TechSpot](https://www.techspot.com/review/3004-nvidia-rtx-5060-ti-pcie-benchmark/)) | + FP4/FP6 | CUDA; deblochează strata-poly | ~8.100 / ~19.500 |
| Arc B580 / Pro B60 / Pro B70 | Xe2 | 12 / 24 / 32 GB | 456 / ? / ? | **4.0 x8** (B580) | XMX + DPAS int8 | port SYCL separat | ~5.200 / ~13.800 / ~19.500 |

Coloana care contează pe hostul actual este VRAM-ul. Curba de hit e concavă: la 44% din perechi, Belady ajunge deja la 94,7%, LRU per strat la 89,5%, tier-ul motorului la 79,6% ([MOD_PLAN][mp]). Fiecare GB în plus valorează mult până pe la 80% acoperire și puțin după. Coloana a doua ca importanță e lățimea cardului: **RX 7600, RTX 4060 Ti/5060 Ti și Arc B580 sunt x8**, deci pe un host PCIe 3.0 ajung la ~6,5 GB/s.

## Raportul promovare/CPU variază de la ~1:1 la ~1:7

### Promovarea scalează cu legătura, calculul CPU cu DRAM-ul și numărul de nuclee

Pe cutia de acum, promovarea unui expert de 1,4 MB costă **~115 µs** (Gen3 x16, probă 13,4 GB/s, ~93% din legătură). Calculul lui pe CPU costă **~42 µs**, cu kernelul Q2_0 la ~80% din cei ~33 GB/s DDR4 ([MOD_PLAN][mp]). Pe alte platforme, ambele numere se pot doar estima. Pentru promovare, proba de pornire a motorului a măsurat 28 GB/s pe un RX 7900 GRE pe Gen4 ([bench GRE][gre]). Asta dă **~50 µs la 4.0 x16**. Cu o placă și un host Gen5, ~26–30 µs (est., la 50–55 GB/s nemăsurați). La 3.0 x8, ~215 µs. Gen4 x8 face cam cât Gen3 x16.

Pentru calculul pe CPU, scalarea cu lățimea de bandă dă un interval, nu un număr. Cifrele publicate sunt citiri AIDA, iar cei 33 GB/s locali sunt văzuți de pancake. Varianta optimistă împarte la 33 GB/s, cea conservatoare la ~50 GB/s, cât arată AIDA pe un Zen 3 desktop. Strata are și două măsurători directe, ambele pe calea CPU DeepSeek IQ2, la un token. Pe un **Zen 4 Ryzen 5 7600 cu 6 fire, 19,7–24,2 GB/s de ponderi**, adică mai puțin decât pe 5700G. Pe **Strix Halo cu 16 fire, 44–48 GB/s** ([ds4_gg_cpu.cpp][ds4]). Lecția: un AM5 cu 6 nuclee nu e automat mai rapid per expert decât 5700G, pentru că sub un anumit număr de nuclee kernelul devine limitat de calcul, nu de DRAM.

| Platformă (cu RX 7900 XT, card Gen4) | CPU per expert, est. | Promovare, est. | Raport CPU : promovare |
|---|---|---|---|
| **AM4 5700G, azi** | **42 µs (măsurat)** | **115 µs (măsurat)** | **1 : 2,7** |
| AM4 Zen 3 fără iGPU, B550/X570 | 28–42 µs | ~50 µs | 1 : 1,2–1,8 |
| AM4 Zen 3 fără iGPU, B450/X470 | 28–42 µs | 115 µs | 1 : 2,7–4 |
| LGA1151 / LGA1200 10 gen | ~38–45 µs | 115 µs | ~1 : 2,6–3 |
| LGA1200 Rocket Lake | 28–42 µs | ~50 µs | 1 : 1,2–1,8 |
| LGA1700 DDR5 | 17–26 µs | ~50 µs | 1 : 1,9–2,9 |
| LGA1851 DDR5-6000…7800 | 12–23 µs | ~50 µs (card Gen5: ~26–30) | 1 : 2,2–4,2 (Gen5: 1 : 1,1–2,5) |
| AM5 un CCD (7600…9800X3D) | 22–33 µs; 6 nuclee pot fi mai lente | ~50 µs (card Gen5 pe placă E: ~26–30) | 1 : 1,5–2,3 (Gen5: ~1 : 0,8–1,4) |
| AM5 două CCD (7950X/9950X) | 18–35 µs | ~50 µs | 1 : 1,4–2,8 |
| AM5 8700G (GPU la 4.0 x8) | 22–35 µs | ~105–115 µs | 1 : 3–5 |
| Xeon W-3400 | ≤7 µs, probabil limitat de nuclee | 26–50 µs | 1 : 4–7+ |
| Xeon E5 v4, un soclu | 23–35 µs (AVX2, nuclee vechi) | 115 µs | 1 : 3,3–5 |
| Strix Halo | ~29–32 µs (din 44–48 GB/s) | **nu există** (VRAM = GTT din același RAM) | n/a |

Tabelul spune două lucruri. Pe platformele cu DDR5 și o legătură Gen4, CPU-ul câștigă și mai clar decât azi, deci pragul de reutilizare pentru promovare trebuie **ridicat**. Pe un Zen 3 desktop pe B550/X570, pe Rocket Lake sau pe AM5 cu placă Gen5, cele două căi ajung la paritate, deci promovarea câștigă teren. Și mai important e că raportul per expert nu ajunge ca regulă. **PcieAuto face deja exact acest calcul** (53–55 µs per expert din pool, față de 13,4 GB/s) și alege o cotă de 0,20–0,25 pe toate cele 48 de straturi. Totuși **`--pcie-frac 0` a câștigat pe 7900 XT, cu 111,7 → 118,4 tok/s** ([MOD_PLAN][mp]). Pe R9700, upstream a coborât de la 0,55 la 0,15 și a câștigat +9,9% la 8K și +25,6% la 32K ([AMD_HIP.md][amdhip]). Explicația plauzibilă este contenția: DMA-ul și pool-ul citesc din același DDR, iar promovările concurează cu trasul rândurilor în `combine_cpu`. Raportul static nu vede nimic din asta. De aici forma corectă a itemului 55: calibrarea trebuie să măsoare timpul ferestrei la câteva valori, nu doar cele două costuri unitare.

### DRAM-ul și plafonul de CCD fixează câștigul CPU

Pe Zen 2 și Zen 3, Infinity Fabric merge la frecvența DDR, deci un CCD nu e gâtuit. Pe Zen 4 și Zen 5 desktop, **un CCD citește cel mult 32 B/ciclu, ~64 GB/s la FCLK 2000**, iar cea mai mare lățime de bandă apare doar cu fire pe ambele CCD-uri ([C&C Zen 4](https://old.chipsandcheese.com/?p=11867); [C&C 9950X](https://old.chipsandcheese.com/2024/08/14/amds-ryzen-9950x-zen-5-on-desktop/)). Un 7700X sau un 9800X3D aduce deci experților CPU doar ~1,2–1,3x față de DDR4 la o singură fereastră de un token. Un 7950X/9950X cu pool-ul împărțit pe CCD-uri poate trece de plafon, dar tot sub dublu (est. 1,5–1,8x). Pe Arrow Lake, orice grup de P-cores ajunge la toată lățimea de bandă DRAM ([C&C Arrow Lake](https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the)), deci acolo problema nu e topologia, ci eterogenitatea nucleelor. X3D nu schimbă calculul pentru ponderi. Un L3 de 96 MB ține ~65 de experți, iar prefetch-ul în L3 (`STRATA_ARENA_WARM`) a ieșit mai prost pe cutia de acum, pentru că experții se citesc o dată pe fereastră.

### Căile ISA există, dar cu goluri precise

Dispatch-ul e la rulare, într-un singur binar ([expert_layout.cpp][el]). `cpu_avx512_ok()` cere F+BW+VL+VNNI+VBMI plus starea ZMM salvată de OS (XCR0 & 0xE6). Asta înseamnă că **orice Zen 4/Zen 5 și Rocket Lake / Ice Lake / Sapphire Rapids iau kernelele AVX-512**: `iq_avx512.cpp`, `nvfp4_avx512.cpp`, `tq_avx512.cpp`, plus kernelul Q2_0 de 512 biți din `expert.cpp`. Toate AM4, LGA1151/1200-10 și LGA1700/1851 rămân pe AVX2. Skylake-X și Cascade Lake HEDT pică testul VBMI. Pentru ele există un nivel opt-in, `STRATA_VNNI_TIER=1`, care rotunjește diferit, deci o parte din itemul 57 e deja în cod. Copiile AVX-VNNI ale kernelelor AVX2 (`target("avxvnni")`) sunt calea pentru Alder/Raptor/Arrow Lake. Pe Zen 5 testul AVX-VNNI ar trece și el, dar AVX-512 câștigă dispatch-ul înainte. `cpu_gather_fast()` întoarce adevărat **doar pentru GenuineIntel cu AVX-VNNI**, și nu pe părțile doar cu E-cores, deci gather-ele sunt oprite pe toate AMD și pe părțile afectate de Downfall. Pe hibride, `cpu_gather_fast_here()` le permite doar firelor de pe P-cores (CPUID 0x1A). `STRATA_FORCE_ISA` acceptă doar `avx2`/`avx`/`sse`, deci nu se poate forța un A/B pe AVX-512 sau AVX-VNNI, ci doar oprirea lor (`STRATA_FORCE_AVX2=1`). Folderele vechi Q2_0 fără `native_experts.txt` cer AVX-512, deci pe Intel desktop merg doar pe Rocket Lake ([INTEL_ARC.md][arc]).

Câștigul AVX-512 crește cu numărul de tokeni din fereastră. Pe calea DeepSeek, Strata a măsurat **1,05–1,3x la un token, 2,4–3,5x la 4 tokeni și 2,8–4,8x la 16 tokeni** pe Zen 4/5 ([ds4_gg_cpu.cpp][ds4]). Ferestrele de verificare ale pancake au ~3,05 tokeni, iar kernelele de rânduri AVX-512 sunt deja șabloane pe numărul de tokeni (`row_dot<TY, NT>` în `iq_avx512.cpp`, `row_dot<NT>` în `nvfp4_avx512.cpp`). Un câștig de ~2x pe rândurile CPU pe Zen 4/5 e deci plauzibil (est.). Asta e motivul itemului 59, iar partea de citire a codului se poate face și pe 5700G.

### Nucleele hibride și memoria unificată schimbă regulile pool-ului

Pe Intel hibrid, valoarea implicită a `PoolAffinity` e `All`: un worker pe nucleu fizic, inclusiv E-cores, cu P-cores sortate primele. Fazele pool-ului se termină cu bariere, iar decodarea cu gather rulează doar pe P-cores. Împărțirea egală îi face deci pe P-cores să aștepte E-cores. Testul planificatorului notează deja că pe 8P+12E „12–16 workers bat 19” ([pool.cpp][pool]). `--pool-affinity auto|p-cores` există, dar e opt-in. Pe Arrow Lake, fără SMT, `p-cores` lasă doar 7 workers, deci trebuie măsurat față de `auto` (item 56).

Pe Strix Halo, „promovarea” dispare: VRAM-ul GPU e un pool GTT din același LPDDR5X. Strata are deja calea unificată. `device_free_bytes()` numără MemAvailable minus 6 GiB, iar copierea experților CPU în RAM pe iGPU a supraangajat memoria și a blocat un prompt de 8K. Cutia maintainerilor pornește cu `ttm.pages_limit` pentru 112 GiB GTT și `amd_iommu=off` ([STRIX_HALO.md][halo]). Limita devine lățimea de bandă DRAM împărțită între CPU și iGPU. Latența CPU urcă de la ~140 ns la ~200 ns sub sarcină GPU ([C&C](https://old.chipsandcheese.com/?p=37437)). Referința măsurată este **19–21 tok/s pe cod** ([README.strata.md][readme]), mult sub cele ~126 tok/s ale presetului rapid de pe 7900 XT. Strix Halo e deci o platformă de capacitate, nu de viteză (item 60).

## Terenul comun: șase mecanisme care ajută pe orice platformă

Primul mecanism comun e **calibrarea la pornire (item 55)**. Tabelul de mai sus arată că valoarea corectă a lui `--pcie-frac`, a pragului de reutilizare și a `--adapt-swaps` depinde de platformă cu un factor de până la ~7. Pancake are deja măsurătorile unitare: PcieAuto pentru timpul pool-ului per expert și proba PCIe. Ce lipsește e pasul care le transformă în valori implicite verificate pe timpul ferestrei: 2–3 valori de `--pcie-frac` (0, cota PcieAuto, la mijloc) pe câteva ferestre de încălzire, cu cea mai bună valoare scrisă în log și într-un cache pe mașină. Itemul se construiește și se verifică pe cutia de acum. Punctul măsurat e cunoscut (0 bate 0,20–0,25), deci calibrarea trebuie să-l regăsească singură. Asta e testul ei de acceptare. Variabilitatea de ~20% între rulări este riscul principal: calibrarea are nevoie de mai multe ferestre și de un prag de histerezis.

Al doilea mecanism sunt **valorile implicite după mărimea VRAM (item 61)**. Acoperirea merge de la ~10% pe 8 GB la 100% pe 48 GB. Mărimea tier-ului, cota PCIe și bugetul de swap ar trebui să decurgă din numărul de perechi care încap, nu dintr-un preset scris pentru 20 GB. Regula pentru cardurile x8 pe un host Gen3 (`--pcie-frac 0`) trebuie să vină din proba de legătură, nu dintr-un tabel de carduri. Așa prinde și cazurile în care un card x16 stă într-un slot x8 sau x4 ([MULTI_GPU.md][multi]). O parte se poate verifica local, emulând carduri mai mici prin limitarea sloturilor de cache pe 7900 XT. Pentru cardul de 20 GB, item 61 trebuie să reproducă exact presetul de azi.

Al treilea mecanism e **loaderul direct pe Linux (item 38)**. Pe Linux, arena de 33 GiB se citește azi prin page cache, la 0,49 GiB/s, și concurează cu arena blocată. E mecanismul cel mai plauzibil pentru oomd. Portarea pe `direct_file.cpp` (16 fire, O_DIRECT) ajută orice platformă Linux. Câștigul crește cu NVMe-ul: est. ~10–14 s la pornire pe Gen3 x4, mai puțin pe un M.2 Gen4/Gen5 legat de CPU. Pe LGA1151/1200, unde NVMe trece prin DMI 3.0 x4, e și singura cale de a nu dubla traficul prin page cache.

Al patrulea e **lucrul pe suprapunere și programare** (itemii 42–44, 46, 48–50, 53). Fereastra de editare de 37,4 ms e aproape suma părților ei, iar podeaua cu suprapunere ideală e ~20 ms ([Tehnologii drum de date hardware][dd]). Acest buget există pe orice combinație CPU+dGPU. Proporțiile se schimbă: pe DDR5, partea CPU scade, deci munca GPU de dinainte de sonerie (48) și spinul din `combine_cpu` (46, 49) devin o parte mai mare din fereastră. Itemii aceștia câștigă deci în valoare relativă tocmai pe platformele mai rapide. Simulatorul max-plus (42) e același pe orice platformă. Diferă doar calibrarea lui, iar asta îl leagă de item 55.

Al cincilea e **cache-ul per tură** (35 → 39, 16, 31, 11). Re-citirile la fiecare tură se măsoară în secunde și depind de corectitudinea restaurării, nu de hardware. Testul de echivalență la restaurare (35) e precondiția. Tier-ul pe disc (39) câștigă pe orice NVMe, mai ales pe cele legate de CPU.

Al șaselea sunt **porțile de calitate**. Lanțul e token forcing → BFCL v4 multi-turn + LiveCodeBench → HELMET → Terminal-Bench, cu rulări repetate ca distribuții ([MOD_PLAN][mp]). Orice item lossy trece prin ele, oricare ar fi platforma: deferral-ul 8/28, turbo-ul. Pe platformele cu raport CPU:promovare diferit, același item lossy poate da câștiguri diferite, dar poarta e aceeași. La acestea se adaugă o bucată ieftină din itemul 57: o **linie de platformă în log la pornire** (ISA aleasă, gather da/nu, afinitatea pool-ului, lățimea și generația legăturii, CCD-uri, mărimea VRAM) și valorile `avx512`/`avx-vnni` în `STRATA_FORCE_ISA`. Ambele sunt XS ca efort și fac utile rapoartele venite de pe alte mașini.

## Munca specifică: șapte direcții, aproape toate cer hardware

**Hibride P/E (item 56)** schimbă valoarea implicită în `auto` pe CPU-uri hibride, sau ponderează împărțirea sarcinilor după clasa nucleului (furt dinamic de sarcini sau bucăți mai mici). Codul e mic, iar detecția există deja (PMU hibrid, `cpu_capacity`, `EfficiencyClass` pe Windows). Măsurătoarea cere Raptor Lake sau Arrow Lake. Pe acestea trebuie comparate `auto`, `p-cores` și valoarea implicită a planificatorului. Pe Arrow Lake rezultatul nu e evident, pentru că E-cores Skymont sunt aproape de P-cores, iar fără SMT nu există altă rezervă.

**Pool conștient de CCD (item 58)** împarte workerii pe ambele CCD-uri pe Zen 4/5 cu două CCD-uri, pentru a trece de plafonul de ~64 GB/s per CCD. Regula stă lângă cea din 56, în același cod de afinitate. Pe 9950X3D intră în conflict cu ideea de a ține datele calde pe CCD-ul cu V-Cache. Varianta mai simplă e 9950X sau 9950X3D2, unde CCD-urile sunt simetrice. Cere hardware.

**AVX-512 pe ferestrele de verificare (item 59)**: întâi o verificare de cod, posibilă azi, că rândurile AVX-512 pentru formatele zilnice (Q2_0 și IQ2_XS/IQ3) folosesc tokenii ferestrei ca pe calea DeepSeek. Abia după vine măsurătoarea cu `kq_window_bench` pe Zen 4/5, care cere hardware. Pe 5700G nu se poate măsura nimic AVX-512.

**Golurile ISA (item 57, restul)** sunt trei. Un AVX-512 fără VBMI pentru Skylake-X/Cascade Lake: există ca `STRATA_VNNI_TIER=1`, dar trebuie validat pe hardware real înainte să devină implicit. Folderele Q2_0 vechi pe Intel desktop: fie documentate clar, fie cu o cale AVX2. Și AMX, care a fost respins (vezi mai jos). Prioritatea e mică: platformele vizate sunt HEDT folosit, nu ținta zilnică.

**Presetul unificat Strix Halo (item 60)** pornește de la calea unificată existentă și de la build-ul Docker `rocm-gfx1151`. Fără promovare, presetul trebuie să reechilibreze CPU și iGPU pe aceeași lățime de bandă DRAM. Calculul pe CPU are sens doar când iGPU-ul e saturat, nu pentru a evita transferuri. Cere hardware.

**Re-măsurarea pe gfx12 (item 62)**: preseturile `*-hip` și porturile din Strata-rdna3 (`rdna_gr`, `rdna_mmvq`, fuziunile) sunt reglate pe gfx11. gfx12 are propriile căi WMMA `_gfx12`, propriul tabel de valori implicite și propriul `--pcie-frac` (0,15). Trebuie făcut un sweep, nu presupus. Cere un card RDNA4.

**Cardurile x8 pe PCIe 3.0** sunt cazul-limită: ~6,5 GB/s și ~215 µs per promovare, deci promovarea e practic inutilizabilă. Regula din 61 (`--pcie-frac 0`, tier mai mare) le acoperă. Un test parțial e posibil fără card nou: multe BIOS-uri permit forțarea slotului la Gen2, ceea ce dă cam lățimea de bandă Gen3 x8. Asta e o decizie de setare a ownerului, nu a mea. Tot aici intră și configurațiile 2×x8 de pe LGA1700 sau cu două plăci: Strata a măsurat pe 2× RX 6900 XT la 4.0 x8 o pierdere de 43 ms pe fereastră din cota PCIe a helperului înainte de reparație ([SECOND_GPU.md][second]). Asta ține de faza 12 (al doilea GPU), parcată.

## Planul: etapa comună pe cutia de acum, apoi porturile când apare hardware

| Etapă / ordine | Item | Ce | Efort | Risc | Hardware |
|---|---|---|---|---|---|
| 1.1 | 38 | Loader direct pe Linux peste `direct_file.cpp` (provizoriu `FADV_DONTNEED`) | S–M | mic | aici |
| 1.2 | 57 (parțial) | Linie de platformă în log; `STRATA_FORCE_ISA=avx512\|avx-vnni` | XS | mic | aici (pe 5700G se testează doar căile AVX2/AVX) |
| 1.3 | 49, 50, 44 | Lotul de măsurători GPU (timer `combine_cpu`, spin vs rânduri, cale critică) | S | mic | aici |
| 1.4 | 55 | Calibrare la pornire pe timpul ferestrei: `--pcie-frac`, prag de reutilizare, swap-uri; cache pe mașină | M | mediu (zgomot ~20%) | aici; acceptare = regăsește `--pcie-frac 0` pe 7900 XT |
| 1.5 | 61 | Valori implicite după VRAM și după proba de legătură; regula x8/Gen3 | S–M | mic–mediu (overhead de 4,7 GB dintr-un singur punct) | aici, cu sloturi limitate; restul cere hardware |
| 1.6 | 46, 48, 40, 41, 53 | Exacte și ieftine: un singur poller, fuziuni înainte de sonerie, prefetch Q2_0, număr de fire, buget online de swap | S–M fiecare | mic | aici |
| 1.7 | 42, 43 | Simulator max-plus calibrat; `seed_costs` din trasee | 1–2 zile | mediu | aici |
| 1.8 | 35 → 39, 16, 31, 11 | Restaurare echivalentă, apoi tier pe disc și checkpoint-uri pe tură | M | mediu | aici |
| 1.9 | porți | Token forcing → BFCL v4 + LiveCodeBench → HELMET → Terminal-Bench | M | mic | aici |
| 2.1 | 59 (citire cod) | Verifică dacă rândurile AVX-512 pentru Q2_0/IQ folosesc fereastra multi-token | S | mic | aici (doar citire) |
| 2.2 | 61 (x8) | Validarea regulii x8/Gen3 | XS | mic | slot forțat la Gen2 în BIOS (decizie owner) sau card x8: **cere hardware/setare** |
| 2.3 | 56 | `auto` implicit pe hibride sau împărțire ponderată pe clasa nucleului | S cod + bench | mic–mediu | **cere hardware** (Raptor/Arrow Lake) |
| 2.4 | 58 | Pool împărțit pe CCD-uri pe Zen 4/5 cu două CCD | S–M | mediu (conflict X3D asimetric) | **cere hardware** (7950X/9950X) |
| 2.5 | 59 (măsurare) | `kq_window_bench` AVX-512 la 1/3/4 tokeni | S | mic | **cere hardware** (Zen 4/5) |
| 2.6 | 55 re-sweep | Calibrarea verificată pe Gen4/Gen5 și DDR5 | XS (dacă 55 e gata) | mic | **cere hardware** (orice upgrade de platformă) |
| 2.7 | 62 | Sweep pe gfx12 al preseturilor și al porturilor rdna3 | M | mediu | **cere hardware** (RDNA4) |
| 2.8 | 60 | Preset unificat Strix Halo | M | mediu | **cere hardware** (AI Max+ 395) |
| 2.9 | 57 (rest) | AVX-512 fără VBMI ca implicit; Q2_0 vechi pe AVX2 | M | mediu (rotunjiri diferite) | **cere hardware** (Skylake-X/Cascade Lake), prioritate mică |

Ordinea din etapa 1 urmează două reguli. Întâi vin itemii care nu ating decodarea și au câștig sigur (38, linia de platformă). Urmează măsurătorile care decid ce se construiește (49/50/44), apoi cele două mecanisme care fac pancake portabil (55, 61). Itemii 55 și 61 se construiesc împreună: 61 dă punctul de pornire după VRAM și legătură, iar 55 îl corectează pe mașina reală. În etapa 2, singurii itemi care progresează fără hardware nou sunt citirea de cod din 59 și, cu acordul ownerului, emularea x8 prin BIOS. Restul așteaptă o mașină anume. Dacă 55 e făcut bine, o bună parte din „portarea” pe o platformă nouă se reduce la o pornire și la citirea liniei de platformă.

## Ce ar schimba un upgrade pe cutia proprie

Toate cele de mai jos sunt fapte și estimări din date, nu recomandări de cumpărare. **Chipsetul plăcii nu e identificat**, deci verificarea deschisă e `sudo dmidecode -t baseboard`. Lățimea reală a cardului se vede cu `sudo lspci -vv -s 12:00.0 | grep LnkCap`.

**Un Zen 3 fără iGPU (5800X3D, 5950X) pe aceeași placă.** 5700G forțează PCIe 3.0 pe toată platforma, iar cu el chiar și lanes de chipset X570 merg la Gen3 ([TPU](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)). Pe **B550/X570**, un Zen 3 fără iGPU dă **4.0 x16 la GPU și 4.0 x4 la M.2 de pe CPU** ([AnandTech](https://www.anandtech.com/show/15850)): promovarea ar coborî de la ~115 la ~50 µs (est.), raportul la ~1:1,2–1,8. Pe **B450/X470** rămâne Gen3, deci nu se câștigă nimic pe legătură. ISA rămâne AVX2, iar DRAM rămâne DDR4: calculul CPU per expert se schimbă puțin. AIDA arată 54,7 GB/s pe 5950X față de 50,7 pe 5800X3D, adică ~+8%. Cele 16 nuclee pe două CCD-uri ale unui 5950X pot ajuta dacă pancake e parțial limitat de calcul. X3D nu are dovezi pentru ponderi. Pierderea iGPU-ului mută afișajul pe 7900 XT, care va consuma ceva VRAM (nemăsurat), dar eliberează carve-out-ul iGPU din RAM. MOD_PLAN listează reducerea lui ca decizie a ownerului. Valoarea reală a lui 4.0 x16 depinde de item 55: cu `--pcie-frac 0` câștigător azi, o legătură mai rapidă ajută doar dacă recalibrarea mută cota sau admisia în sus.

**O construcție AM5.** Aduce DDR5 și **AVX-512 automat**, fără flag de build: `cpu_avx512_ok` trece pe orice Zen 4/5. Aduce PCIe 4.0 sau 5.0 după chipset, NVMe Gen5 pe plăcile E/B850 și 192–256 GB RAM. Ultimul punct ar ține tot setul de experți în RAM fără presiune de oomd. Plafonul de ~64 GB/s per CCD limitează însă câștigul pe un singur token la ~1,2–1,3x pe părțile cu un CCD, iar un 6 nuclee poate fi chiar mai lent per expert decât 5700G (7600: 19,7–24,2 GB/s pe calea DeepSeek). Câștigul mare apare pe ferestrele multi-token (59) și pe părțile cu două CCD-uri (58). Ambele trebuie măsurate. **8000G** taie slotul GPU la 4.0 x8 (8700G/8600G) sau x4 (8500G) ([AnandTech](https://www.anandtech.com/show/21242)), deci pentru pancake e un pas înapoi față de legătura de azi sau cel mult la egalitate.

**Un GPU de 24/32 GB.** Pe acest host, capacitatea VRAM bate generația PCIe. **RX 7900 XTX (24 GB)** rulează același binar gfx1100 și aceleași preseturi, cu +~2.900 perechi (56%). **AI PRO R9700 (32 GB)** aduce +~8.600 perechi (79%), dar cere build-ul gfx12 și re-măsurarea (62). Are și o lățime de bandă mai mică (~640 față de 800 GB/s), iar legătura Gen5 coboară la Gen3 pe acest host. **RTX 5090 (32 GB, 1.792 GB/s)** e cel mai puternic card consumer, dar trece pe calea CUDA și deblochează strata-poly. **RX 9070 XT (16 GB, 640 GB/s)** ar fi un pas înapoi pentru experți rezidenți. Toate cardurile x8 (RX 7600/7600 XT, RTX 4060 Ti/5060 Ti, Arc B580) ar merge pe acest host la ~6,5 GB/s.

## Ce nu s-a adoptat în această rundă și de ce

| Tehnică / platformă | De ce nu |
|---|---|
| Kernele de experți pe AMX | Doar Sapphire Rapids W are AMX dintre platformele vizate. Pancake nu are nicio cale AMX în kernelele proprii, iar singura dovadă de câștig e un blog apropiat de vânzător pe llama.cpp, cu metodă nespecificată ([OpenMetal](https://openmetal.io/resources/blog/intel-amx-ai-inference-performance/)). Decodarea la un token rămâne limitată de lățimea de bandă. |
| iGPU și NPU Intel | Arc integrat netestat, fără planificare pentru memorie partajată. Kernelele DPAS cer Xe2, iar Gen9 e pe runtime înghețat ([Debian](https://sources.debian.org/src/intel-compute-runtime/25.44.36015.8-1/LEGACY_PLATFORMS.md)). Nu există cod NPU în repo. |
| APU-uri 8000G ca host | Slotul GPU coboară la 4.0 x8/x4, deci promovarea e cel mult la nivelul Gen3 x16 de azi ([TPU](https://techpowerup.com/317997/amds-phoenix-1-and-phoenix-2-apus-differ-in-pcie-lane-count-affects-nvme-drive-performance-and-gpu-pcie-lane-count)). |
| X3D pentru experți | ~65–90 de experți în L3, față de sute pe fereastră. Prefetch-ul în L3 a fost mai prost aici (măsurat). Pe 9950X3D, V-Cache-ul asimetric complică afinitatea. Rămâne o ipoteză pentru activări și KV, nu pentru ponderi. |
| Vega (gfx90c) și 780M (gfx1103) ca ținte de calcul | Lipsesc din matricile ROCm ([ROCm 7.11](https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html)). Vega e doar Vulkan pe această cutie. |
| FP8/BF8 WMMA și SWMMAC 4:2 pe RDNA4 | Experții pancake sunt coduri întregi decodate în int8, deci contează căile IU8, nu FP8 ([C&C RDNA4](https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/)). Experții Flash-Next au ieșit denși (cos 0,844 la 70% sparsitate, item 13). |
| Suport RDNA2 | Fără WMMA și fără dot iu8, pachet netestat pe card real, absent din lista ROCm ([AMD ROCm Linux](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)). Valorile 7900 XTX costă 8,6–13,6 tok/s pe 6900 XT. |
| NVMe prin DMI pe LGA1151/1200 | Toate M.2 stau în spatele DMI 3.0 x4 (~3,9 GB/s), împărțit cu SATA/USB/rețea ([TPU](https://techpowerup.com/review/intel-core-i9-10900k/3.html)). Regula rămâne: modelele pe NVMe legat de CPU, acolo unde există. |

## Concluzie

Runda mută întrebarea de la „pe ce hardware merge pancake mai bine” la „cât din reglajul de azi e specific acestei cutii”. Răspunsul e: mai mult decât părea. Valoarea câștigătoare `--pcie-frac 0`, pragul de admisie, mărimea tier-ului și bugetul de swap sunt toate funcții de două rapoarte, CPU:promovare și VRAM:set de experți, care variază de ~7 ori și de ~10 ori între platforme. Mai mult, calculul static al raportului (PcieAuto) a greșit deja pe hardware-ul de acasă. Calibrarea pe timpul ferestrei (55) împreună cu valorile după VRAM (61) nu sunt deci un lux de portabilitate. Sunt singura cale prin care orice upgrade, inclusiv un simplu Zen 3 pe B550, își livrează câștigul fără o rundă de reglaj manual.

A doua concluzie e despre unde stă viteza nouă. Pe CPU, nicio platformă consumer nu dublează decodarea la un token prin lățime de bandă singură: plafonul de CCD pe Zen 4/5 și eficiența de ~80% a kernelului o limitează. Câștigul mare pe CPU vine din ISA pe ferestre multi-token (AVX-512, 2–5x măsurat pe altă cale), iar pe GPU din capacitate (24–48 GB). Ambele sunt condiționate de hardware pe care nu îl avem, așa că tot ce poate fi construit azi se încheie cu o linie de log care spune, pe orice mașină nouă, exact ce cale a ales pancake și de ce.

[mp]: file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md
[notes]: file:///home/dan/research_notes/Capabilit%C4%83%C8%9Bi%20platforme%20consumer/
[dd]: Tehnologii%20drum%20de%20date%20hardware.md
[amdhip]: file:///home/dan/clone-of-d/src/Strata/docs/AMD_HIP.md
[arc]: file:///home/dan/clone-of-d/src/Strata/docs/INTEL_ARC.md
[el]: file:///home/dan/clone-of-d/src/Strata/src/kernels/cpu/expert_layout.cpp
[pool]: file:///home/dan/clone-of-d/src/Strata/src/kernels/cpu/pool.cpp
[ds4]: file:///home/dan/clone-of-d/src/Strata/src/dsv41/ds4_gg_cpu.cpp
[halo]: file:///home/dan/clone-of-d/src/Strata/docs/STRIX_HALO.md
[readme]: file:///home/dan/clone-of-d/src/Strata/README.strata.md
[gre]: file:///home/dan/clone-of-d/src/Strata/bench/results/2026-10-04-community-rx-7900-gre/README.md
[multi]: file:///home/dan/clone-of-d/src/Strata/docs/MULTI_GPU.md
[second]: file:///home/dan/clone-of-d/src/Strata/docs/SECOND_GPU.md
