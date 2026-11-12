# One shared calibration beats ten per-platform tunings

On consumer platforms, pancake rests on two numbers: **what one expert costs when computed on the CPU** and **what one expert costs when promoted to the GPU**. The ratio between them varies almost tenfold from one platform to another. On today's box (5700G, PCIe 3.0, DDR4) it is **42 µs versus 115 µs, that is 1:2.7**, and it is the only measured point. On AM4 with a Zen 3 without iGPU on B550/X570 it would be ~1:1.2–1.8. On LGA1700/1851 with DDR5 it would be ~1:2–4. On AM5 with a Gen5 board it would be ~1:1 or even inverted. On a Xeon W-3400 with 8 channels it would be ~1:4–7. On Strix Halo there is no promotion at all. All of these are arithmetic estimates; nothing was measured on other hardware. The conclusion is that **no fixed value of `--pcie-frac` or of the reuse threshold can be right everywhere**. Pancake already has the negative example in its own data: PcieAuto computes a share of 0.20–0.25 from the per-expert ratio, but on the 7900 XT the best value was 0 (+6%). On the R9700, upstream found 0.15 instead of 0.55. That is why the shared work comes first: startup calibration on window time, not only on the per-expert ratio (item 55), defaults by VRAM size (61), the direct loader (38), CPU/GPU overlap, the per-turn cache and the quality gates. All of them help on any platform and can be built and verified on the current box. The specific work comes after and needs almost entirely hardware we do not have: P/E affinity on hybrid Intel (56), a CCD-aware pool on Zen 4/5 with two CCDs (58), AVX-512 on verify windows (59), the Strix Halo unified preset (60), re-measurement on gfx12 (62) and the ISA dispatch gaps (57). For the owner's own box, the facts are simple. A Zen 3 without iGPU on B550/X570 doubles the PCIe link without changing ISA or DRAM. AM5 brings DDR5 and AVX-512, but a single CCD is capped at ~64 GB/s. A 24/32 GB GPU matters more on this host than the PCIe generation. One check remains open: the board chipset is not known.

Conventions. "Item N" refers to the phase 11 list in [MOD_PLAN][mp]. Items 55–62 were added on 2026-11-09 from this round. "Est." marks an arithmetic estimate. **All timing figures on platforms other than 5700G + RX 7900 XT are estimates.** The only local measurements are 42 µs for CPU compute (the plan's figure), 53–55 µs for the pool time per missed expert (PcieAuto), 115 µs for one promotion and 13.4 GB/s for the PCIe probe. "Needs hardware" means the item cannot be measured on the current box. The research notes are in [the round notes][notes].

## The matrix shows two axes: the PCIe link and the DRAM seen by the CPU

### AMD platforms

| Platform | PCIe to GPU / NVMe from CPU | Memory, channels, measured BW | L3 | ISA for pancake | Cores | Max RAM | iGPU/NPU useful? |
|---|---|---|---|---|---|---|---|
| **AM4 5700G (Zen 3 APU), today** | **3.0 x16 / 3.0 x4**; even X570 chipset lanes drop to Gen3 ([TPU](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)) | DDR4 2ch; ~33 GB/s seen by pancake ([MOD_PLAN][mp]) | 16 MB (unverified) | **AVX2 + FMA, no AVX-512/AVX-VNNI** (`/proc/cpuinfo`) | 8C/16T, one CCX | 128 GB (unverified) | Vega gfx90c Vulkan only; no |
| AM4 Zen 3 without iGPU (5800X3D/5950X) on B550/X570 | **4.0 x16 / 4.0 x4** ([AnandTech](https://www.anandtech.com/show/15850)) | DDR4 2ch; AIDA 50.7–54.7 GB/s ([hwcooling](https://www.hwcooling.net/wp-content/uploads/2022/04/amd-ryzen-7-5700x-g302.html)) | 32 MB/CCD; 96 MB on X3D | AVX2 | up to 16C/32T, 2 CCD | 128 GB | — |
| AM4 on B450/X470 | 3.0 x16 / 3.0 x4 | as above | as above | AVX2 | as above | 128 GB | — |
| AM5 Zen 4 (7000, 7x00X3D) | 5.0 x16 on X670E/B650E/X870(E), otherwise 4.0 ([Puget](https://pugetsystems.com/labs/articles/amd-x870e-vs-x870-vs-x670e-vs-x670-vs-b650e-vs-b650)); NVMe 5.0 on E/B850 boards | DDR5 2ch; **~64 GB/s read per CCD** (32 B/cycle IF link) ([C&C](https://old.chipsandcheese.com/?p=11867)) | 32 MB/CCD; 96 MB X3D | **AVX-512** F/BW/VL/DQ/VNNI/VBMI/BF16, double-pumped on 256 bits ([numberworld](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)) | up to 16C/32T, 2 CCD | 192–256 GB (4×64 GB, ~6000 MT/s) ([TweakTown](https://www.tweaktown.com/news/105825/msi-releases-amd-agesa-1-2-0-3e-bios-officially-supports-4-64gb-sticks-up-to-256gb-ram/index.html)) | — |
| AM5 8700G/8600G (Phoenix 1) | **4.0 x8** / 4.0 x4+x4 ([AnandTech](https://www.anandtech.com/show/21242)) | DDR5 2ch | n/a | AVX-512 (Zen 4, assumed) | 8C | as AM5 | 780M gfx1103: missing from the ROCm matrix |
| AM5 8500G/8300G (Phoenix 2) | **4.0 x4** / 4.0 x4+x2 ([TPU](https://techpowerup.com/317997/amds-phoenix-1-and-phoenix-2-apus-differ-in-pcie-lane-count-affects-nvme-drive-performance-and-gpu-pcie-lane-count)) | DDR5 | — | AVX-512 (assumed) | Zen 4 + 4c | as AM5 | no |
| AM5 Zen 5 (9000, 9x00X3D) | depends on chipset | DDR5 2ch; ~60 GB/s AIDA at DDR5-5000 on 16C; still 64 GB/s/CCD ([numberworld](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)) | 96 MB 9800X3D; 128 MB 9950X3D (V-Cache on one CCD); X3D2 on both ([StorageReview](https://www.storagereview.com/review/amd-ryzen-9-9950x3d2-dual-edition-review-3d-v-cache-on-both-ccds)) | **AVX-512 on real 512 bits + AVX-VNNI** ([Tom's](https://www.tomshardware.com/pc-components/cpus/amd-zen-5-support-arrives-with-five-new-instructions-gnu-compiler-collection-posted)) | up to 16C/32T, 2 CCD | 256 GB | — |
| Strix Point (HX 370) | 4.0 (lane count not found) | LPDDR5X-7500; GPU 96 GB/s ([C&C](https://old.chipsandcheese.com/2024/08/24/amds-radeon-890m-strix-points-bigger-igpu/)) | 24 MB | AVX-512 on 256-bit hardware | 4 Zen 5 + 8 Zen 5c | up to 256 GB | gfx1150 in ROCm 7.11 preview |
| **Strix Halo (AI Max+ 395)** | 4.0, 16 lanes (PRO) | **LPDDR5X-8000 256-bit, 256 GB/s theoretical, ~103 GB/s from one CCD** ([C&C](https://old.chipsandcheese.com/?p=37437)) | 64 MB | AVX-512 (Zen 5; width unconfirmed) | 16C/32T | **128 GB unified** | **gfx1151 supported in ROCm** ([ROCm](https://rocm.docs.amd.com/en/docs-7.9.0/install/compatibility-matrix.html)); 50 TOPS NPU unused |

### Intel platforms

| Platform | PCIe to GPU / NVMe from CPU / DMI | Memory, theoretical/measured BW | L2 / L3 | ISA for pancake | Cores | Max RAM |
|---|---|---|---|---|---|---|
| LGA1151 (8th/9th gen) | 3.0 x16 / **none, NVMe via DMI 3.0 x4 (~3.9 GB/s)** | 2× DDR4-2666; 42.7 / ? | 256 KB / ~16 MB | AVX2 | homogeneous, HT | 128 GB |
| LGA1200 10th gen | 3.0 x16 / none / DMI 3.0 x4 ([TPU](https://techpowerup.com/review/intel-core-i9-10900k/3.html)) | 2× DDR4-2933; 46.9 / ? | 256 KB / 20 MB | AVX2 | homogeneous, HT | 128 GB |
| LGA1200 11th gen (Rocket Lake) | **4.0 x16 / 4.0 x4** / DMI x8 ([AnandTech](https://www.anandtech.com/show/16523)) | 2× DDR4-3200 | ? | **AVX2 + AVX-512 with VBMI, so it passes `cpu_avx512_ok`** | homogeneous, HT | 128 GB |
| LGA1700 (12th–14th gen) | 5.0 x16 / 4.0 x4 / DMI 4.0 x8 | DDR5-5600 ~82 GB/s, DDR4 ~50 GB/s ([CapFrameX](https://www.capframex.com/tests/Alder%20Lake-S%2012900K%20is%20the%20new%20Gaming%20King)) | 2 MB/P / 36 MB | AVX2 + **AVX-VNNI**, no AVX-512 (fused off at the factory) ([HotHardware](https://hothardware.com/news/intel-fusing-avx-512-alder-lake-silicon)) | 8P(HT) + up to 16E | 192 GB |
| LGA1851 (Core Ultra 200S) | 5.0 x16 / 5.0 x4 (higher latency) + 4.0 x4 / DMI 4.0 x8 | DDR5: 76 GB/s at 5600, ~90 at 6000, ~118 at 7800 ([Igor's Lab](https://www.igorslab.de/en/arrow-lake-ddr5-detail-memory-clock-uncore-timings-core-ultra-7-265k-two-platforms/)) | 3 MB/P / 36 MB ([C&C](https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the)) | AVX2 + AVX-VNNI | **8P + 16E, no SMT** | 192 GB (?) |
| Xeon W-3400 (W790) | 5.0, 112 lanes | **8× DDR5-4800 RDIMM, 307 GB/s theoretical** ([AnandTech](https://anandtech.com/show/18741)) | 2 MB / 105 MB | full AVX-512 + AMX | homogeneous, HT | 4 TB |
| Xeon E5 v3/v4 (used) | 3.0 / CPU lanes | 4× DDR4-2400, ~77 theoretical; 119 GB/s STREAM on 2 sockets ([AnandTech](https://www.anandtech.com/show/10158/the-intel-xeon-e5-v4-review/9)) | 256 KB / up to 55 MB | AVX2 | homogeneous, HT | 768 GB+ |

On Intel, the iGPUs do not run pancake. UHD 630 stayed on the frozen "legacy" compute-runtime ([Fedora](https://www.fedoraproject.org/wiki/Changes/IntelCompute2025)). Xe-LP and the 4 Xe cores in Arrow Lake have no DPAS int8, and the SYCL port has no planning for shared memory ([INTEL_ARC.md][arc]). The 13 TOPS NPU has no code in the repo.

### Consumer GPUs and expert pair coverage

Coverage is computed for Qwen3.8-Flash-Next IQ2_XS: 24,576 pairs, ~1.4 MB each. The starting point is the only measured one, 20 GB → 10,909 pairs (44%), with ~4.7 GB of non-expert overhead in the pancake preset. The other rows are linear est. ([AMD_HIP.md][amdhip]; [ROCm specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html); [NVIDIA whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf); [Intel ARK B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)).

| Card | Target | VRAM | BW GB/s | Card link width | Matrix instructions | pancake path | Est. pairs |
|---|---|---|---|---|---|---|---|
| RX 6800/6900 XT | gfx1030 | 16 GB | ~512 | 4.0 x16 | none (only `sdot4`) | generic HIP, package untested | ~8,100 (33%) |
| RX 7600 / 7600 XT | gfx1102 | 8 / 16 GB | ~288 | **4.0 x8** | WMMA FP16/BF16/IU8/IU4 | HIP gfx11, "reported working" | ~2,400 / ~8,100 |
| RX 7800 XT | gfx1101 | 16 GB | ~624 | 4.0 x16 | RDNA3 WMMA | HIP gfx11, validated by owner | ~8,100 (33%) |
| **RX 7900 XT (today)** | gfx1100 | **20 GB** | **800** | 4.0 x16, runs at 3.0 | RDNA3 WMMA | **the tuned pancake path** | **10,909 (44%, measured)** |
| RX 7900 XTX | gfx1100 | 24 GB | ~960 | 4.0 x16 | RDNA3 WMMA | same binary and presets | ~13,800 (56%) |
| PRO W7900 | gfx1100 | 48 GB | ~864 | 4.0 x16 | RDNA3 WMMA | same binary (untested) | all 24,576 |
| RX 9060 XT | gfx1200 | 16 GB | ~320 | **5.0 x16** | WMMA + FP8/BF8 + SWMMAC 4:2 | HIP gfx12 | ~8,100 |
| RX 9070 XT | gfx1201 | 16 GB | 640 | 5.0 x16 ([Guru3D](https://www.guru3d.com/review/review-radeon-rx-9070-xt-reference-edition/)) | as above | HIP gfx12, tested upstream | ~8,100 |
| AI PRO R9700 | gfx1201 | 32 GB | ~640 | 5.0 x16 | as above | HIP gfx12, upstream development machine | ~19,500 (79%) |
| RTX 3060 / 3090 | sm_86 | 12 / 24 GB | 360 / 936 | 4.0 x16 | TC INT8/INT4, 2:4 | CUDA (the former card) | ~5,200 / ~13,800 |
| RTX 4060 Ti / 4090 | sm_89 | 8–16 / 24 GB | ~288 / 1008 | **4.0 x8** / x16 | + FP8 | CUDA | ~2,400–8,100 / ~13,800 |
| RTX 5060 Ti / 5090 | sm_120 | 16 / 32 GB | ~448 / 1792 | **5.0 x8** / 5.0 x16 ([TechSpot](https://www.techspot.com/review/3004-nvidia-rtx-5060-ti-pcie-benchmark/)) | + FP4/FP6 | CUDA; unlocks strata-poly | ~8,100 / ~19,500 |
| Arc B580 / Pro B60 / Pro B70 | Xe2 | 12 / 24 / 32 GB | 456 / ? / ? | **4.0 x8** (B580) | XMX + DPAS int8 | separate SYCL port | ~5,200 / ~13,800 / ~19,500 |

The column that matters on the current host is VRAM. The hit curve is concave: at 44% of pairs, Belady already reaches 94.7%, per-layer LRU 89.5%, the engine tier 79.6% ([MOD_PLAN][mp]). Each extra GB is worth a lot up to about 80% coverage and little after. The second most important column is the card link width: **RX 7600, RTX 4060 Ti/5060 Ti and Arc B580 are x8**, so on a PCIe 3.0 host they reach ~6.5 GB/s.

## The promotion/CPU ratio varies from ~1:1 to ~1:7

### Promotion scales with the link, CPU compute with DRAM and core count

On the current box, promoting one 1.4 MB expert costs **~115 µs** (Gen3 x16, 13.4 GB/s probe, ~93% of the link). Computing it on the CPU costs **~42 µs**, with the Q2_0 kernel at ~80% of the ~33 GB/s of DDR4 ([MOD_PLAN][mp]). On other platforms, both numbers can only be estimated. For promotion, the engine's startup probe measured 28 GB/s on an RX 7900 GRE on Gen4 ([GRE bench][gre]). That gives **~50 µs at 4.0 x16**. With a Gen5 board and host, ~26–30 µs (est., at an unmeasured 50–55 GB/s). At 3.0 x8, ~215 µs. Gen4 x8 is about the same as Gen3 x16.

For CPU compute, scaling with bandwidth gives a range, not a number. The published figures are AIDA reads, while the local 33 GB/s is what pancake sees. The optimistic variant divides by 33 GB/s, the conservative one by ~50 GB/s, which is what AIDA shows on a desktop Zen 3. Strata also has two direct measurements, both on the DeepSeek IQ2 CPU path, at one token. On a **Zen 4 Ryzen 5 7600 with 6 threads, 19.7–24.2 GB/s of weights**, that is less than on the 5700G. On **Strix Halo with 16 threads, 44–48 GB/s** ([ds4_gg_cpu.cpp][ds4]). The lesson: a 6-core AM5 is not automatically faster per expert than the 5700G, because below a certain core count the kernel becomes compute-bound, not DRAM-bound.

| Platform (with RX 7900 XT, Gen4 card) | CPU per expert, est. | Promotion, est. | CPU : promotion ratio |
|---|---|---|---|
| **AM4 5700G, today** | **42 µs (measured)** | **115 µs (measured)** | **1 : 2.7** |
| AM4 Zen 3 without iGPU, B550/X570 | 28–42 µs | ~50 µs | 1 : 1.2–1.8 |
| AM4 Zen 3 without iGPU, B450/X470 | 28–42 µs | 115 µs | 1 : 2.7–4 |
| LGA1151 / LGA1200 10th gen | ~38–45 µs | 115 µs | ~1 : 2.6–3 |
| LGA1200 Rocket Lake | 28–42 µs | ~50 µs | 1 : 1.2–1.8 |
| LGA1700 DDR5 | 17–26 µs | ~50 µs | 1 : 1.9–2.9 |
| LGA1851 DDR5-6000…7800 | 12–23 µs | ~50 µs (Gen5 card: ~26–30) | 1 : 2.2–4.2 (Gen5: 1 : 1.1–2.5) |
| AM5 one CCD (7600…9800X3D) | 22–33 µs; 6 cores can be slower | ~50 µs (Gen5 card on an E board: ~26–30) | 1 : 1.5–2.3 (Gen5: ~1 : 0.8–1.4) |
| AM5 two CCDs (7950X/9950X) | 18–35 µs | ~50 µs | 1 : 1.4–2.8 |
| AM5 8700G (GPU at 4.0 x8) | 22–35 µs | ~105–115 µs | 1 : 3–5 |
| Xeon W-3400 | ≤7 µs, probably core-bound | 26–50 µs | 1 : 4–7+ |
| Xeon E5 v4, one socket | 23–35 µs (AVX2, old cores) | 115 µs | 1 : 3.3–5 |
| Strix Halo | ~29–32 µs (from 44–48 GB/s) | **does not exist** (VRAM = GTT from the same RAM) | n/a |

The table says two things. On platforms with DDR5 and a Gen4 link, the CPU wins even more clearly than today, so the reuse threshold for promotion must be **raised**. On a desktop Zen 3 on B550/X570, on Rocket Lake or on AM5 with a Gen5 board, the two paths reach parity, so promotion gains ground. More important, the per-expert ratio is not enough as a rule. **PcieAuto already does exactly this calculation** (53–55 µs per expert from the pool, versus 13.4 GB/s) and picks a share of 0.20–0.25 on all 48 layers. Still, **`--pcie-frac 0` won on the 7900 XT, with 111.7 → 118.4 tok/s** ([MOD_PLAN][mp]). On the R9700, upstream went down from 0.55 to 0.15 and gained +9.9% at 8K and +25.6% at 32K ([AMD_HIP.md][amdhip]). The plausible explanation is contention: DMA and the pool read from the same DDR, and promotions compete with pulling rows in `combine_cpu`. The static ratio sees none of this. Hence the correct shape of item 55: calibration must measure window time at a few values, not only the two unit costs.

### DRAM and the CCD cap set the CPU gain

On Zen 2 and Zen 3, Infinity Fabric runs at DDR frequency, so one CCD is not throttled. On desktop Zen 4 and Zen 5, **one CCD reads at most 32 B/cycle, ~64 GB/s at FCLK 2000**, and the highest bandwidth appears only with threads on both CCDs ([C&C Zen 4](https://old.chipsandcheese.com/?p=11867); [C&C 9950X](https://old.chipsandcheese.com/2024/08/14/amds-ryzen-9950x-zen-5-on-desktop/)). A 7700X or a 9800X3D therefore brings CPU experts only ~1.2–1.3x over DDR4 at a single one-token window. A 7950X/9950X with the pool split across CCDs can pass the cap, but still below double (est. 1.5–1.8x). On Arrow Lake, any group of P-cores reaches the full DRAM bandwidth ([C&C Arrow Lake](https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the)), so there the problem is not topology but core heterogeneity. X3D does not change the math for weights. A 96 MB L3 holds ~65 experts, and prefetch into L3 (`STRATA_ARENA_WARM`) came out worse on the current box, because experts are read once per window.

### The ISA paths exist, but with precise gaps

Dispatch is at runtime, in a single binary ([expert_layout.cpp][el]). `cpu_avx512_ok()` requires F+BW+VL+VNNI+VBMI plus the ZMM state saved by the OS (XCR0 & 0xE6). This means that **any Zen 4/Zen 5 and Rocket Lake / Ice Lake / Sapphire Rapids take the AVX-512 kernels**: `iq_avx512.cpp`, `nvfp4_avx512.cpp`, `tq_avx512.cpp`, plus the 512-bit Q2_0 kernel in `expert.cpp`. All AM4, LGA1151/1200-10 and LGA1700/1851 stay on AVX2. Skylake-X and Cascade Lake HEDT fail the VBMI test. For them there is an opt-in tier, `STRATA_VNNI_TIER=1`, which rounds differently, so part of item 57 is already in the code. The AVX-VNNI copies of the AVX2 kernels (`target("avxvnni")`) are the path for Alder/Raptor/Arrow Lake. On Zen 5 the AVX-VNNI test would also pass, but AVX-512 wins the dispatch first. `cpu_gather_fast()` returns true **only for GenuineIntel with AVX-VNNI**, and not on E-core-only parts, so gathers are off on all AMD and on parts affected by Downfall. On hybrids, `cpu_gather_fast_here()` allows them only for threads on P-cores (CPUID 0x1A). `STRATA_FORCE_ISA` accepts only `avx2`/`avx`/`sse`, so you cannot force an A/B on AVX-512 or AVX-VNNI, only turn them off (`STRATA_FORCE_AVX2=1`). Old Q2_0 folders without `native_experts.txt` require AVX-512, so on desktop Intel they work only on Rocket Lake ([INTEL_ARC.md][arc]).

The AVX-512 gain grows with the number of tokens in the window. On the DeepSeek path, Strata measured **1.05–1.3x at one token, 2.4–3.5x at 4 tokens and 2.8–4.8x at 16 tokens** on Zen 4/5 ([ds4_gg_cpu.cpp][ds4]). Pancake's verify windows have ~3.05 tokens, and the AVX-512 row kernels are already templated on token count (`row_dot<TY, NT>` in `iq_avx512.cpp`, `row_dot<NT>` in `nvfp4_avx512.cpp`). A ~2x gain on CPU rows on Zen 4/5 is therefore plausible (est.). That is the reason for item 59, and the code-reading part can be done on the 5700G too.

### Hybrid cores and unified memory change the pool rules

On hybrid Intel, the default of `PoolAffinity` is `All`: one worker per physical core, including E-cores, with P-cores sorted first. Pool phases end with barriers, and decode with gather runs only on P-cores. An equal split therefore makes the P-cores wait for the E-cores. The scheduler test already notes that on 8P+12E "12–16 workers beat 19" ([pool.cpp][pool]). `--pool-affinity auto|p-cores` exists, but it is opt-in. On Arrow Lake, without SMT, `p-cores` leaves only 7 workers, so it must be measured against `auto` (item 56).

On Strix Halo, "promotion" disappears: GPU VRAM is a GTT pool from the same LPDDR5X. Strata already has the unified path. `device_free_bytes()` counts MemAvailable minus 6 GiB, and copying the CPU experts into RAM on the iGPU overcommitted memory and stalled an 8K prompt. The maintainers' box boots with `ttm.pages_limit` for 112 GiB GTT and `amd_iommu=off` ([STRIX_HALO.md][halo]). The limit becomes DRAM bandwidth shared between CPU and iGPU. CPU latency rises from ~140 ns to ~200 ns under GPU load ([C&C](https://old.chipsandcheese.com/?p=37437)). The measured reference is **19–21 tok/s on code** ([README.strata.md][readme]), far below the ~126 tok/s of the fast preset on the 7900 XT. Strix Halo is therefore a capacity platform, not a speed platform (item 60).

## The common ground: six mechanisms that help on any platform

The first common mechanism is **startup calibration (item 55)**. The table above shows that the right value of `--pcie-frac`, of the reuse threshold and of `--adapt-swaps` depends on the platform by a factor of up to ~7. Pancake already has the unit measurements: PcieAuto for pool time per expert and the PCIe probe. What is missing is the step that turns them into defaults verified on window time: 2–3 values of `--pcie-frac` (0, the PcieAuto share, one in between) over a few warm-up windows, with the best value written to the log and to a per-machine cache. The item is built and verified on the current box. The measured point is known (0 beats 0.20–0.25), so calibration must find it on its own. That is its acceptance test. The ~20% run-to-run variability is the main risk: calibration needs several windows and a hysteresis threshold.

The second mechanism is **defaults by VRAM size (item 61)**. Coverage goes from ~10% on 8 GB to 100% on 48 GB. Tier size, PCIe share and swap budget should follow from the number of pairs that fit, not from a preset written for 20 GB. The rule for x8 cards on a Gen3 host (`--pcie-frac 0`) must come from the link probe, not from a card table. That way it also catches cases where an x16 card sits in an x8 or x4 slot ([MULTI_GPU.md][multi]). Part of it can be verified locally, by emulating smaller cards through limiting cache slots on the 7900 XT. For the 20 GB card, item 61 must reproduce today's preset exactly.

The third mechanism is **the direct loader on Linux (item 38)**. On Linux, the 33 GiB arena is read today through the page cache, at 0.49 GiB/s, and competes with the locked arena. It is the most plausible mechanism for oomd. Porting to `direct_file.cpp` (16 threads, O_DIRECT) helps any Linux platform. The gain grows with the NVMe: est. ~10–14 s at startup on Gen3 x4, less on a CPU-attached Gen4/Gen5 M.2. On LGA1151/1200, where NVMe goes through DMI 3.0 x4, it is also the only way to avoid doubling traffic through the page cache.

The fourth is **work on overlap and scheduling** (items 42–44, 46, 48–50, 53). The 37.4 ms edit window is close to the sum of its parts, and the floor with ideal overlap is ~20 ms ([Hardware data path technologies][dd]). This budget exists on any CPU+dGPU combination. The proportions change: on DDR5, the CPU part shrinks, so GPU work before the doorbell (48) and the spin in `combine_cpu` (46, 49) become a larger part of the window. These items therefore gain relative value precisely on the faster platforms. The max-plus simulator (42) is the same on any platform. Only its calibration differs, and that ties it to item 55.

The fifth is **the per-turn cache** (35 → 39, 16, 31, 11). Re-reads at every turn are measured in seconds and depend on restore correctness, not on hardware. The restore equivalence test (35) is the precondition. The disk tier (39) wins on any NVMe, especially CPU-attached ones.

The sixth is **the quality gates**. The chain is token forcing → BFCL v4 multi-turn + LiveCodeBench → HELMET → Terminal-Bench, with repeated runs as distributions ([MOD_PLAN][mp]). Any lossy item goes through them, whatever the platform: the 8/28 deferral, turbo. On platforms with a different CPU:promotion ratio, the same lossy item can give different gains, but the gate is the same. Added to these is a cheap piece of item 57: a **platform line in the log at startup** (ISA chosen, gather yes/no, pool affinity, link width and generation, CCDs, VRAM size) and the values `avx512`/`avx-vnni` in `STRATA_FORCE_ISA`. Both are XS in effort and make reports from other machines useful.

## The specific work: seven directions, almost all need hardware

**Hybrid P/E (item 56)** changes the default to `auto` on hybrid CPUs, or weights the task split by core class (dynamic work stealing or smaller chunks). The code is small, and detection already exists (hybrid PMU, `cpu_capacity`, `EfficiencyClass` on Windows). The measurement needs Raptor Lake or Arrow Lake. On those, `auto`, `p-cores` and the scheduler default must be compared. On Arrow Lake the result is not obvious, because Skymont E-cores are close to P-cores, and without SMT there is no other reserve.

**CCD-aware pool (item 58)** splits workers across both CCDs on Zen 4/5 with two CCDs, to pass the ~64 GB/s per-CCD cap. The rule sits next to the one from 56, in the same affinity code. On the 9950X3D it conflicts with the idea of keeping hot data on the V-Cache CCD. The simpler variant is the 9950X or 9950X3D2, where the CCDs are symmetric. Needs hardware.

**AVX-512 on verify windows (item 59)**: first a code check, possible today, that the AVX-512 rows for the daily formats (Q2_0 and IQ2_XS/IQ3) use the window's tokens like the DeepSeek path. Only then comes the measurement with `kq_window_bench` on Zen 4/5, which needs hardware. Nothing AVX-512 can be measured on the 5700G.

**The ISA gaps (item 57, the rest)** are three. An AVX-512 without VBMI for Skylake-X/Cascade Lake: it exists as `STRATA_VNNI_TIER=1`, but must be validated on real hardware before it becomes the default. Old Q2_0 folders on desktop Intel: either documented clearly, or with an AVX2 path. And AMX, which was rejected (see below). Priority is low: the target platforms are used HEDT, not the daily target.

**The Strix Halo unified preset (item 60)** starts from the existing unified path and from the `rocm-gfx1151` Docker build. Without promotion, the preset must rebalance CPU and iGPU on the same DRAM bandwidth. CPU compute makes sense only when the iGPU is saturated, not to avoid transfers. Needs hardware.

**Re-measurement on gfx12 (item 62)**: the `*-hip` presets and the ports from Strata-rdna3 (`rdna_gr`, `rdna_mmvq`, the fusions) are tuned on gfx11. gfx12 has its own `_gfx12` WMMA paths, its own defaults table and its own `--pcie-frac` (0.15). A sweep must be done, not assumed. Needs an RDNA4 card.

**x8 cards on PCIe 3.0** are the edge case: ~6.5 GB/s and ~215 µs per promotion, so promotion is practically unusable. The rule from 61 (`--pcie-frac 0`, larger tier) covers them. A partial test is possible without a new card: many BIOSes allow forcing the slot to Gen2, which gives about Gen3 x8 bandwidth. That is a settings decision for the owner, not mine. Also here are the 2×x8 configurations on LGA1700 or with two boards: Strata measured on 2× RX 6900 XT at 4.0 x8 a loss of 43 ms per window from the helper's PCIe share before the fix ([SECOND_GPU.md][second]). This belongs to phase 12 (second GPU), parked.

## The plan: the common stage on the current box, then the ports when hardware appears

| Stage / order | Item | What | Effort | Risk | Hardware |
|---|---|---|---|---|---|
| 1.1 | 38 | Direct loader on Linux over `direct_file.cpp` (interim `FADV_DONTNEED`) | S–M | low | here |
| 1.2 | 57 (partial) | Platform line in the log; `STRATA_FORCE_ISA=avx512\|avx-vnni` | XS | low | here (on the 5700G only the AVX2/AVX paths are tested) |
| 1.3 | 49, 50, 44 | The GPU measurement batch (`combine_cpu` timer, spin vs rows, critical path) | S | low | here |
| 1.4 | 55 | Startup calibration on window time: `--pcie-frac`, reuse threshold, swaps; per-machine cache | M | medium (~20% noise) | here; acceptance = finds `--pcie-frac 0` on the 7900 XT |
| 1.5 | 61 | Defaults by VRAM and by link probe; the x8/Gen3 rule | S–M | low–medium (4.7 GB overhead from a single point) | here, with limited slots; the rest needs hardware |
| 1.6 | 46, 48, 40, 41, 53 | Exact and cheap: a single poller, fusions before the doorbell, Q2_0 prefetch, thread count, online swap budget | S–M each | low | here |
| 1.7 | 42, 43 | Calibrated max-plus simulator; `seed_costs` from traces | 1–2 days | medium | here |
| 1.8 | 35 → 39, 16, 31, 11 | Equivalent restore, then disk tier and per-turn checkpoints | M | medium | here |
| 1.9 | gates | Token forcing → BFCL v4 + LiveCodeBench → HELMET → Terminal-Bench | M | low | here |
| 2.1 | 59 (code reading) | Check whether the AVX-512 rows for Q2_0/IQ use the multi-token window | S | low | here (reading only) |
| 2.2 | 61 (x8) | Validating the x8/Gen3 rule | XS | low | slot forced to Gen2 in BIOS (owner decision) or an x8 card: **needs hardware/setting** |
| 2.3 | 56 | `auto` default on hybrids or split weighted by core class | S code + bench | low–medium | **needs hardware** (Raptor/Arrow Lake) |
| 2.4 | 58 | Pool split across CCDs on Zen 4/5 with two CCDs | S–M | medium (asymmetric X3D conflict) | **needs hardware** (7950X/9950X) |
| 2.5 | 59 (measurement) | `kq_window_bench` AVX-512 at 1/3/4 tokens | S | low | **needs hardware** (Zen 4/5) |
| 2.6 | 55 re-sweep | Calibration verified on Gen4/Gen5 and DDR5 | XS (if 55 is done) | low | **needs hardware** (any platform upgrade) |
| 2.7 | 62 | gfx12 sweep of the presets and the rdna3 ports | M | medium | **needs hardware** (RDNA4) |
| 2.8 | 60 | Strix Halo unified preset | M | medium | **needs hardware** (AI Max+ 395) |
| 2.9 | 57 (rest) | AVX-512 without VBMI as default; old Q2_0 on AVX2 | M | medium (different rounding) | **needs hardware** (Skylake-X/Cascade Lake), low priority |

The order in stage 1 follows two rules. First come the items that do not touch decode and have a sure gain (38, the platform line). Next come the measurements that decide what gets built (49/50/44), then the two mechanisms that make pancake portable (55, 61). Items 55 and 61 are built together: 61 gives the starting point by VRAM and link, and 55 corrects it on the real machine. In stage 2, the only items that progress without new hardware are the code reading in 59 and, with the owner's consent, x8 emulation through the BIOS. The rest wait for a specific machine. If 55 is done well, a good part of "porting" to a new platform reduces to one startup and reading the platform line.

## What an upgrade would change on the owner's box

Everything below is facts and estimates from data, not purchase recommendations. **The board chipset is not identified**, so the open check is `sudo dmidecode -t baseboard`. The real card link width can be seen with `sudo lspci -vv -s 12:00.0 | grep LnkCap`.

**A Zen 3 without iGPU (5800X3D, 5950X) on the same board.** The 5700G forces PCIe 3.0 on the whole platform, and with it even X570 chipset lanes run at Gen3 ([TPU](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)). On **B550/X570**, a Zen 3 without iGPU gives **4.0 x16 to the GPU and 4.0 x4 to the CPU-attached M.2** ([AnandTech](https://www.anandtech.com/show/15850)): promotion would drop from ~115 to ~50 µs (est.), the ratio to ~1:1.2–1.8. On **B450/X470** it stays Gen3, so nothing is gained on the link. ISA stays AVX2, and DRAM stays DDR4: CPU compute per expert changes little. AIDA shows 54.7 GB/s on the 5950X versus 50.7 on the 5800X3D, that is ~+8%. The 16 cores on two CCDs of a 5950X can help if pancake is partly compute-bound. X3D has no evidence for weights. Losing the iGPU moves the display to the 7900 XT, which will use some VRAM (unmeasured), but frees the iGPU carve-out from RAM. MOD_PLAN lists reducing it as an owner decision. The real value of 4.0 x16 depends on item 55: with `--pcie-frac 0` winning today, a faster link helps only if recalibration moves the share or the admission up.

**An AM5 build.** It brings DDR5 and **AVX-512 automatically**, without a build flag: `cpu_avx512_ok` passes on any Zen 4/5. It brings PCIe 4.0 or 5.0 depending on the chipset, Gen5 NVMe on E/B850 boards and 192–256 GB of RAM. The last point would hold the whole expert set in RAM without oomd pressure. The ~64 GB/s per-CCD cap, however, limits the one-token gain to ~1.2–1.3x on single-CCD parts, and a 6-core can even be slower per expert than the 5700G (7600: 19.7–24.2 GB/s on the DeepSeek path). The big gain appears on multi-token windows (59) and on two-CCD parts (58). Both must be measured. **8000G** cuts the GPU slot to 4.0 x8 (8700G/8600G) or x4 (8500G) ([AnandTech](https://www.anandtech.com/show/21242)), so for pancake it is a step back from today's link or at most equal.

**A 24/32 GB GPU.** On this host, VRAM capacity beats PCIe generation. **RX 7900 XTX (24 GB)** runs the same gfx1100 binary and the same presets, with +~2,900 pairs (56%). **AI PRO R9700 (32 GB)** brings +~8,600 pairs (79%), but needs the gfx12 build and re-measurement (62). It also has lower bandwidth (~640 versus 800 GB/s), and the Gen5 link drops to Gen3 on this host. **RTX 5090 (32 GB, 1,792 GB/s)** is the strongest consumer card, but it moves to the CUDA path and unlocks strata-poly. **RX 9070 XT (16 GB, 640 GB/s)** would be a step back for resident experts. All x8 cards (RX 7600/7600 XT, RTX 4060 Ti/5060 Ti, Arc B580) would run on this host at ~6.5 GB/s.

## What was not adopted in this round and why

| Technique / platform | Why not |
|---|---|
| Expert kernels on AMX | Of the target platforms, only Sapphire Rapids W has AMX. Pancake has no AMX path in its own kernels, and the only evidence of a gain is a vendor-adjacent blog on llama.cpp, with an unspecified method ([OpenMetal](https://openmetal.io/resources/blog/intel-amx-ai-inference-performance/)). One-token decode stays bandwidth-bound. |
| Intel iGPU and NPU | Integrated Arc untested, no planning for shared memory. The DPAS kernels need Xe2, and Gen9 is on a frozen runtime ([Debian](https://sources.debian.org/src/intel-compute-runtime/25.44.36015.8-1/LEGACY_PLATFORMS.md)). There is no NPU code in the repo. |
| 8000G APUs as host | The GPU slot drops to 4.0 x8/x4, so promotion is at most at today's Gen3 x16 level ([TPU](https://techpowerup.com/317997/amds-phoenix-1-and-phoenix-2-apus-differ-in-pcie-lane-count-affects-nvme-drive-performance-and-gpu-pcie-lane-count)). |
| X3D for experts | ~65–90 experts in L3, versus hundreds per window. Prefetch into L3 was worse here (measured). On the 9950X3D, the asymmetric V-Cache complicates affinity. It remains a hypothesis for activations and KV, not for weights. |
| Vega (gfx90c) and 780M (gfx1103) as compute targets | Missing from the ROCm matrices ([ROCm 7.11](https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html)). Vega is Vulkan only on this box. |
| FP8/BF8 WMMA and SWMMAC 4:2 on RDNA4 | Pancake experts are integer codes decoded to int8, so the IU8 paths matter, not FP8 ([C&C RDNA4](https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/)). Flash-Next experts came out dense (cos 0.844 at 70% sparsity, item 13). |
| RDNA2 support | No WMMA and no iu8 dot, package untested on a real card, missing from the ROCm list ([AMD ROCm Linux](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)). The 7900 XTX values cost 8.6–13.6 tok/s on the 6900 XT. |
| NVMe via DMI on LGA1151/1200 | All M.2 slots sit behind DMI 3.0 x4 (~3.9 GB/s), shared with SATA/USB/network ([TPU](https://techpowerup.com/review/intel-core-i9-10900k/3.html)). The rule stays: models on CPU-attached NVMe, where it exists. |

## Conclusion

The round moves the question from "on what hardware does pancake run better" to "how much of today's tuning is specific to this box". The answer is: more than it seemed. The winning value `--pcie-frac 0`, the admission threshold, the tier size and the swap budget are all functions of two ratios, CPU:promotion and VRAM:expert set, which vary ~7x and ~10x between platforms. Moreover, the static ratio calculation (PcieAuto) has already been wrong on the home hardware. Calibration on window time (55) together with the VRAM-based defaults (61) are therefore not a portability luxury. They are the only way for any upgrade, including a simple Zen 3 on B550, to deliver its gain without a round of manual tuning.

The second conclusion is about where new speed lies. On the CPU, no consumer platform doubles one-token decode through bandwidth alone: the CCD cap on Zen 4/5 and the kernel's ~80% efficiency limit it. The big CPU gain comes from ISA on multi-token windows (AVX-512, 2–5x measured on another path), and on the GPU from capacity (24–48 GB). Both depend on hardware we do not have, so everything that can be built today ends with a log line that says, on any new machine, exactly which path pancake chose and why.

[mp]: file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md
[notes]: file:///home/dan/research_notes/Consumer%20platform%20capabilities/
[dd]: Hardware%20data%20path%20technologies.md
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
