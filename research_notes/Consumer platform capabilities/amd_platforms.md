# AMD consumer platforms: capability matrix for pancake (hybrid CPU+GPU MoE, NVMe/RAM/VRAM expert streaming)

Scope: AM4 (Zen 2/3, X3D, Cezanne APUs), AM5 (Zen 4/5, X3D, 8000G APUs), mobile-class unified-memory APUs (Strix Point, Strix Halo / Ryzen AI Max+ 395). Current box: Ryzen 7 5700G (AM4, Zen 3 APU), DDR4, RX 7900 XT.

Source labels used below besides web links:
- [repo:FILE] = read-only grep of /home/dan/clone-of-d/src/Strata (pancake) on 2026-11-09.
- [local] = read-only commands on the current box on 2026-11-09 (`/proc/cpuinfo`, `lspci -vv -s 12:00.0`, `/proc/cmdline`).

Note on sources: AMD's own product/chipset pages (amd.com) timed out on fetch twice; chipset and SKU figures below come from AnandTech, TechPowerUp, Puget, Tom's Hardware and similar, which reproduce AMD tables. Where those disagree it is noted.

## Q1. PCIe generation and lane counts (GPU slot and NVMe; CPU vs chipset), by platform/chipset

### Takeaway
On the current 5700G the whole platform is PCIe 3.0 (GPU x16, one CPU M.2 x4, and even X570 chipset lanes drop to Gen3). Any Zen 2/Zen 3 non-APU CPU on B550/X570 gives PCIe 4.0 x16 to the GPU and a 4.0 x4 CPU M.2; AM5 gives PCIe 4.0 or 5.0 depending on chipset, but the 8000G APUs cut the GPU slot to x8 (8600G/8700G) or x4 (8500G). Since the RX 7900 XT itself is a PCIe 4.0 card, PCIe 4.0 x16 is the useful ceiling for this GPU; PCIe 5.0 only matters for NVMe (and a future Gen5 GPU).

### Cited Findings
**AM4**
- Ryzen 5000G (Cezanne) APUs use PCIe 3.0, unlike other Zen 3 parts which have PCIe 4.0 — [TechPowerUp 5700G review](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)
- 5700G's 24 CPU lanes: 16 to the graphics slot, 4 to a CPU-attached M.2 NVMe, 4 to the chipset link — [TechPowerUp 5700G review](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)
- With 5000G APUs, X570's downstream chipset lanes run at Gen3 even though they are Gen4-capable; TPU notes PCIe 3.0 vs 4.0 x16 changes GPU gaming perf only ~1% but holds back fast Gen4 M.2 SSDs — [TechPowerUp 5700G review](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)
- AnandTech chipset table: CPU link PCIe 4.0 on X570 and B550, 3.0 on B450; chipset uplink 4.0 (X570), 3.0 (B550), 2.0-class downstream (B450) — [AnandTech B550 overview](https://www.anandtech.com/show/15850)
- B550's main gain over B450 is official PCIe 4.0 on the CPU-driven x16 slot; B550 boards usually have one PCIe 4.0 x4 M.2 from the CPU, further M.2 slots are PCIe 3.0 from the chipset — [AnandTech B550 overview](https://www.anandtech.com/show/15850)
- X570 has its own PCIe 4.0 lanes off the chipset; B550 has none — [GamersNexus chipset comparison](https://gamersnexus.net/guides/3582-amd-chipset-differences-b550-vs-x570-b450-x470-zen-3)
- Conflict: some secondary sites claim B550 gives PCIe 4.0 only to the GPU slot (not M.2), contradicting AnandTech; board wiring varies — [Tech4Gamers](https://tech4gamers.com/b550-vs-x570/) vs [AnandTech](https://www.anandtech.com/show/15850)

**AM5**
- X870E, X870, X670E and B650E guarantee PCIe 5.0 on the primary x16 slot; B850, B840, A620 (and B650/X670) guarantee only PCIe 4.0 for graphics — [Puget Systems AM5 chipset comparison](https://pugetsystems.com/labs/articles/amd-x870e-vs-x870-vs-x670e-vs-x670-vs-b650e-vs-b650); [Colfax AM5 chipset comparison](https://colfax-intl.com/workstations/amd5-chipset-comparison)
- PCIe 5.0 M.2 is mandated on B850 but optional on B650 (Tom's Hardware, TechSpot); another source says optional on both — [Tom's Hardware](https://www.tomshardware.com/pc-components/chipsets/amd-discontinues-b650-chipset-to-transition-to-the-newer-b850-chipset-affordable-am5-motherboards-just-got-a-bit-pricier); [TechSpot](https://www.techspot.com/news/109184-amd-phases-out-b650-chipset-b850-becomes-new.html)
- B650E promotes 4 additional CPU lanes from 4.0 to 5.0 for an NVMe SSD; AvaDirect lists B650E 36 total / 24 Gen5-capable vs B650 36/0 — [AvaDirect chipset guide](https://avadirect.com/blog/amd-chipset-guide-from-mainstream-gaming-pcs-and-desktops-to-workstation-pc/)
- Total usable lanes: X870E 44, X870 and B850 36, B840 34; of which up to 24 Gen5 on X870E/X870, 4 on B850, 0 on B840 — [AvaDirect](https://avadirect.com/blog/amd-chipset-guide-from-mainstream-gaming-pcs-and-desktops-to-workstation-pc/) (one bullet in the same article has an apparent typo for X870)
- X870/X670 are dual-chipset (up to 12 PCIe 4.0 + 8 PCIe 3.0 chipset lanes); B850/B650 single chipset (up to 8 PCIe 4.0 + 4 PCIe 3.0) — [Colfax](https://colfax-intl.com/workstations/amd5-chipset-comparison)
- Ryzen 8000G Phoenix 1 (8600G/8700G): 20 PCIe 4.0 lanes total, 16 usable, GPU x8, NVMe x4/x4 — [AnandTech 8700G/8600G review](https://www.anandtech.com/show/21242)
- Ryzen 8000G Phoenix 2 (8500G/8300G): 14 lanes total, 10 usable, GPU x4, NVMe x4/x2 — [AnandTech 8700G review](https://www.anandtech.com/show/21242); [TechPowerUp](https://techpowerup.com/317997/amds-phoenix-1-and-phoenix-2-apus-differ-in-pcie-lane-count-affects-nvme-drive-performance-and-gpu-pcie-lane-count); [Wccftech](https://wccftech.com/amd-ryzen-8000g-phoenix-2-desktop-apus-limited-single-channel-ddr5-memory-pcie-4-0-x4-for-dgpu-pcie-4-0-x2-for-m-2-ssds/amp/)

**Mobile/unified**
- Ryzen AI Max+ (PRO) 395: FP11 package, PCIe 4.0, 16 total lanes (lane count found for PRO variant only) — [LaptopMedia](https://laptopmedia.com/?p=1713936); [ChipVersus](https://chipversus.com/hardware/cpu/ryzen-ai-max-395)
- Ryzen AI 9 HX PRO 370 (Strix Point) uses PCIe Gen 4; lane count not found for the non-PRO HX 370 — [TechPowerUp CPU DB](https://techpowerup.com/cpu-specs/ryzen-ai-9-hx-pro-370.c4003)

**pancake's own PCIe numbers**
- Plan assumes PCIe 3.0 x16 ≈ 12 GB/s effective, ~115 us per 1.4 MB expert promotion vs ~42 us CPU compute — [repo:MOD_PLAN.md lines 321, 496]
- A Strata result with 2x RX 6900 XT on PCIe 4.0 x8 each: helper PCIe share cost the primary 43 ms per verify window before a fix (19 ms after) — [repo:docs/SECOND_GPU.md line 129]

### Inferences
- Promotion cost per 1.4 MB expert scales roughly with link bandwidth: PCIe 3.0 x16 ~115 us (measured in plan) → PCIe 4.0 x16 ~55-60 us (≈2x link rate) → PCIe 5.0 x16 ~30 us. At PCIe 4.0 the promote (~58 us) approaches the current CPU compute cost (~42 us on DDR4), so the "compute on CPU vs promote" break-even shifts toward promoting more experts; at Gen5 promotion would be cheaper than DDR4 CPU compute. (Arithmetic from the plan's numbers; not measured.)
- Cheapest upgrade on the current motherboard: a non-APU Zen 3 (e.g. 5800X3D/5950X) on B550/X570 lifts the GPU slot and CPU M.2 to Gen4 with the same DDR4. If the board is B450/X470 it stays Gen3 (AnandTech table). The board model was not checked here.
- An 8000G APU on AM5 would REDUCE GPU link width (x8 Gen4 ≈ x16 Gen3 bandwidth; 8500G x4 Gen4 is worse than today). Avoid 8000G for a dGPU+pancake build.
- PCIe 5.0 GPU slot buys nothing for the 7900 XT (a Gen4 card; see Gaps for local verification), but PCIe 5.0 NVMe (B650E/B850/X870) doubles the NVMe→RAM tier bandwidth vs Gen4.
- The NVMe tier should sit on CPU lanes, not chipset lanes: chipset M.2 shares the x4 chipset uplink (Gen3 on 5700G/B550) with USB/SATA/network.

### Gaps
- RX 7900 XT link capability (Gen4 x16) not confirmed from a cited AMD page in this pass; `lspci -vv` LnkCap needs root here (`sudo lspci -vv -s 12:00.0 | grep LnkCap`). The current motherboard chipset was not identified (check `sudo dmidecode -t baseboard`).
- No measured host→device copy bandwidth numbers for PCIe 4.0/5.0 with RDNA3 found in this pass.
- amd.com chipset pages timed out; AM5 M.2 Gen5 mandatory/optional per chipset has conflicting secondary sources.

## Q2. Memory: type, channels, speed, capacity, measured bandwidth

### Takeaway
All socketed consumer AMD parts are 2-channel (128-bit). Measured CPU read bandwidth is ~50 GB/s on DDR4 AM4 and ~60-64 GB/s on DDR5 AM5, where single-CCD Zen 4/5 parts are capped by the CCD's 32 B/cycle Infinity Fabric read link (~64 GB/s at FCLK 2000). Strix Halo is 256-bit LPDDR5X-8000 (256 GB/s theoretical, ~103 GB/s from one CCD measured, 128 GB max). AM5 boards now accept 256 GB (4x64 GB) at reduced speed.

### Cited Findings
- AIDA64 memory read: Ryzen 7 5800X3D ~50.67 GB/s, 5700X 50.70, 5950X 54.67 (memory config not shown on chart page) — [hwcooling.net chart](https://www.hwcooling.net/wp-content/uploads/2022/04/amd-ryzen-7-5700x-g302.html)
- On Zen 2/Zen 3 the Infinity Fabric generally runs at DDR clock, so single-CCD fabric bandwidth wasn't a limit — [Chips and Cheese, Zen 4 part 3](https://old.chipsandcheese.com/?p=11867)
- Zen 4 desktop (7950X, DDR5-6000, FCLK 2000): a single CCD is limited by its 32 B/cycle read link to the fabric; writes are 16 B/cycle per CCD; highest bandwidth came from spreading threads across both CCDs — [Chips and Cheese, Zen 4 part 3](https://old.chipsandcheese.com/?p=11867); [Chips and Cheese, Zen 4 part 2](https://chipsandcheese.com/p/amds-zen-4-part-2-memory-subsystem-and-conclusion?open=false)
- Zen 5 desktop (9950X): per-CCD bandwidth limits are the same as Zen 4 — 64 GB/s read, 32 GB/s write per cluster at that FCLK (quoted from the C&C article via forum; verify against original) — [Chips and Cheese 9950X](https://old.chipsandcheese.com/2024/08/14/amds-ryzen-9950x-zen-5-on-desktop/)
- AIDA64 ~60 GB/s at DDR5-5000 on a 16-core Zen 5 (≈0.75 B/cycle/core); the author calls memory bandwidth "the biggest achilles heel of Zen5" — [numberworld Zen 5 AVX-512 teardown](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)
- C&C suggested fast DDR5 with single-CCD parts may not pay off for bandwidth-bound work (per search summary of their Zen 4 coverage) — [Chips and Cheese, Zen 4 part 3](https://old.chipsandcheese.com/?p=11867)
- AM5 256 GB: MSI AGESA 1.2.0.3e BIOS supports 4x64 GB (256 GB); stable at DDR5-6000 per MSI, with 2DPC frequency penalty noted by ASUS; 2-slot boards top out at 128 GB — [TweakTown](https://www.tweaktown.com/news/105825/msi-releases-amd-agesa-1-2-0-3e-bios-officially-supports-4-64gb-sticks-up-to-256gb-ram/index.html); [Tom's Hardware](https://www.tomshardware.com/pc-components/ram/new-desktop-pc-chips-now-support-an-incredible-256gb-of-memory-asrock-and-msi-update-ddr5-support-on-intel-z790-and-amd-x670e-motherboards); [Wccftech](https://wccftech.com/msi-officially-rolls-out-amd-agesa-1-2-0-3e-bios)
- Strix Halo: 256-bit LPDDR5X, 256 GB/s theoretical; one Zen 5 CCD has 128 GB/s theoretical and C&C observed just over 103 GB/s; a single CCD cannot saturate DRAM and read-only loads fall "far short" of 256 GB/s due to cross-die bandwidth; CCD write link widened to 32 B/cycle vs desktop — [Chips and Cheese Strix Halo memory subsystem](https://old.chipsandcheese.com/?p=37437); [C&C Strix Halo overview](https://old.chipsandcheese.com/?p=37385)
- Strix Halo CPU latency ~140 ns baseline, ~200 ns (spikes >300 ns) under heavy GPU load; 32 MB Infinity Cache serves the GPU only (CPU accesses don't fill it) — [C&C memory subsystem](https://old.chipsandcheese.com/?p=37437); [C&C Infinity Cache in Strix Halo](https://old.chipsandcheese.com/?p=37421)
- Ryzen AI Max+ 395: LPDDR5X-8000, max 128 GB — [LaptopMedia](https://laptopmedia.com/?p=1713936); 256 GB/s — [flopper.io spec sheet](https://flopper.io/gpu/amd-ryzen-ai-max-plus-395-128gb/spec-sheet.pdf)
- Strix Point (HX 370): DDR5-5600 / LPDDR5X-7500, max 256 GB listed by one spec site; C&C measured 96.11 GB/s from Vulkan (GPU) on LPDDR5-7500 — [Notebookcheck HX 370](https://www.notebookcheck.net/AMD-Ryzen-AI-9-HX-370-Processor-Benchmarks-and-Specs.836729.0.html); [C&C Radeon 890M](https://old.chipsandcheese.com/2024/08/24/amds-radeon-890m-strix-points-bigger-igpu/)
- Strata measured on Strix Halo: CPU AVX-512 expert rows reach 44-48 GB/s of weights (IQ2_XXS/IQ2_XS, one token, 16 threads) vs ggml 35-37; on Zen 4 Ryzen 5 7600 (6 threads): 19.7-24.2 GB/s — [repo:src/dsv41/ds4_gg_cpu.cpp lines 190-193]
- Strix Halo maintainers' box boots with `ttm.pages_limit=29360128 ttm.page_pool_size=29360128` (112 GiB GTT; default ≈ half of RAM) and `amd_iommu=off`; ~109 GB of a 132 GB expert set stays in the 128 GB pool, 23 GB read from SSD — [repo:docs/STRIX_HALO.md lines 46, 136-137]

### Inferences
- CPU expert compute is memory-bandwidth-bound: the plan's 42 us per 1.4 MB expert ≈ 33 GB/s of weights, i.e. a large fraction of the ~50 GB/s DDR4 platform ceiling. On AM5 a single-CCD part (7600/7700X/9700X/9800X3D) raises the ceiling only to ~64 GB/s read (IF link cap), roughly 1.2-1.3x; a dual-CCD part (7950X/9950X) with threads across both CCDs can exceed that (DDR5-6000 dual-channel theoretical 96 GB/s). Expect at most ~1.5-1.8x CPU expert throughput going DDR4→DDR5, not more, unless the kernel was compute-bound on Zen 3 (AVX2).
- Strata's own numbers support this: 6-core Zen 4 tops at ~20-24 GB/s of weights; 16-core Strix Halo at 44-48 GB/s, i.e. still below the single-CCD 103 GB/s measured read, so on Strix Halo it is probably compute/latency-bound and more threads help.
- Capacity: AM4 128 GB (4x32) vs AM5 192/256 GB matters for keeping the whole expert set in RAM (the NVMe tier disappears for ~130-250 GB packs). Strix Halo at 128 GB still needs SSD for a 132 GB set (repo).
- Pancake should pin CPU expert threads across CCDs on dual-CCD parts (C&C found best bandwidth that way); on X3D dual-CCD (9950X3D) this conflicts with "keep hot data on the V-Cache CCD".

### Gaps
- No clean STREAM-triad numbers found for 5700G DDR4-3200 or for 9800X3D/9950X at DDR5-6000 from a single review; AIDA figures are partial. Official max DDR speeds per SKU (DDR4-3200 on AM4; DDR5-5200 Zen 4 / DDR5-5600 Zen 5) and AM4 128 GB max are from training knowledge, not verified this pass (amd.com timed out).
- Strix Halo dual-CCD aggregate CPU read bandwidth not given numerically by C&C.

## Q3. ISA: AVX2, AVX-512 subsets, Zen 4 double-pump vs Zen 5 full-width, AVX-VNNI; and how pancake selects paths

### Takeaway
Zen 2/Zen 3 (incl. the 5700G and 5800X3D) are AVX2-only. Zen 4 (Ryzen 7000, 8000G, 7x00X3D) has AVX-512 F/CD/BW/DQ/VL/IFMA/VBMI/VBMI2/VNNI/BF16/BITALG/VPOPCNTDQ (no FP16), executed double-pumped on 256-bit units; Zen 5 desktop (Ryzen 9000, 9x00X3D) has full 512-bit datapaths, adds AVX-VNNI and VP2INTERSECT; Zen 5 mobile (Strix Point, and per repo Strix Halo uses the same AVX-512 path) keeps the ISA but on 256-bit hardware per numberworld. pancake's `cpu_avx512_ok()` needs F+BW+VL+VNNI+VBMI, so every Zen 4/Zen 5 part takes the AVX-512 kernels and every AM4 part falls back to AVX2.

### Cited Findings
- Local box: 5700G flags include fma, avx, f16c, avx2, bmi2 and no avx512*/avx_vnni — [local: /proc/cpuinfo]
- Zen 4 (7950X) /proc/cpuinfo: avx512f, dq, ifma, cd, bw, vl, bf16, vbmi, vbmi2, vnni, bitalg, vpopcntdq — [Phoronix Zen 4 AVX-512](https://www.phoronix.com/review/amd-zen4-avx512/1) (as quoted in search result; direct fetch was 403)
- Zen 4 does not support AVX-512 FP16 — [Neowin correction](https://www.neowin.net/amp/amd-ryzen-to-leave-intel-12th-13th-14th-gen-in-the-dust-in-benchmarks-due-to-avx-512/)
- Zen 4 double-pumps 512-bit ops on 256-bit hardware (4x256 = 1024 bits/cycle); Zen 5 desktop doubles to 2048 bits/cycle: 2x512 FADD, 2x512 FMA/IMUL/IFMA, 4x512 integer add/logic, 2x512 loads + 1x512 store per cycle (Zen 4: 1 load, ~0.5 store of 512-bit) — [numberworld Zen 5 teardown](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)
- Zen 5 mobile (Strix Point) has 256-bit hardware; 512-bit ops at half throughput, full ISA retained — [numberworld](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)
- Zen 5 regresses formerly 1-cycle SIMD ops to 2-cycle latency (throughput unchanged) — [numberworld](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown)
- Zen 5 adds AVX-VNNI, MOVDIRI, MOVDIR64B, AVX512_VP2INTERSECT, PREFETCHI (znver5 GCC patch) — [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/amd-zen-5-support-arrives-with-five-new-instructions-gnu-compiler-collection-posted); [HotHardware](https://hothardware.com/news/amd-zen5-gcc-patch)
- Ryzen 9000 drops only ~10% frequency under AVX-512 — [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/ryzen-9000-cpus-drop-10-frequency-executing-avx-512-instructions-intel-cpus-typically-suffer-from-more-substantial-clock-speed-drops); Zen 5's FPU unbothered by 512-bit ops at 5.7 GHz — [Chips and Cheese](https://old.chipsandcheese.com/?p=35152); Phoronix found full AVX-512 pipeline costs only a couple of watts — [Tom's Hardware on Phoronix](https://www.tomshardware.com/pc-components/cpus/amds-zen-5-avx-512-performance-tested-zen-5-performs-significantly-better-than-zen-4-on-linux-without-consuming-any-more-power)

**pancake ISA paths (read-only grep)**
- Kernel files: iq_avx512.cpp, nvfp4_avx512.cpp, tq_avx512.cpp (AVX-512), iq_avx2.cpp / q2_avx2.cpp / kq_avx2.cpp / nvfp4_avx2.cpp / tq_avx2.cpp (AVX2), iq_avx1/kq_avx1/q2_avx1 (AVX-only floor), plus expert.cpp — [repo:src/kernels/cpu/, CMakeLists.txt 2226-2229]
- AVX-512 TUs compiled per-source with `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mavx512vpopcntdq -mbmi2 -mf16c -mfma`; engine picks kernels at run time — [repo:CMakeLists.txt 2107-2115, 2252-2253]
- `cpu_avx512_ok()`: CPUID leaf 7 EBX bits 16 (F), 30 (BW), 31 (VL) and ECX bits 11 (VNNI), 1 (VBMI), plus XCR0 & 0xE6 (OS saves ZMM state); `STRATA_FORCE_AVX2=1` disables — [repo:src/kernels/cpu/expert_layout.cpp 75-111]
- `cpu_vnni_ok()`: a VNNI-without-VBMI tier (Cascade Lake) that is opt-in via `STRATA_VNNI_TIER=1`; `cpu_vbmi_ok()` gates byte-permute unpack — [repo:expert_layout.cpp ~176-205]
- `cpu_avxvnni_ok()`: CPUID 7.1 EAX bit 4, vendor-agnostic; used by AVX2 kernels' vpdpbusd/vpdpwssd copies (comments mention only Alder Lake/Sapphire Rapids, but the check would also fire on Zen 5) — [repo:expert_layout.cpp 361-373; iq_avx2.cpp line 18; q2_avx2.cpp line 169]
- `cpu_gather_fast()` returns true only for GenuineIntel with AVX-VNNI (so false on all AMD) — [repo:expert_layout.cpp 319-339]
- DeepSeek path `want_512()` uses AMD family: 0x19 (Zen 3/4) / 0x1A (Zen 5); measured Zen 5 Strix Halo and Zen 4 7600 gains from AVX-512 rows (1 token 1.05-1.3x; 4 tokens 2.4-3.5x; 16 tokens 2.8-4.8x); non-AMD AVX-512 not measured → AVX2 — [repo:src/dsv41/ds4_gg_cpu.cpp 150-200]
- Zen 5 tuning comments in tq_avx2.cpp (cycles/block measured on Zen 5) — [repo:src/kernels/cpu/tq_avx2.cpp 10-23]

### Inferences
- Moving from 5700G to any Zen 4/5 unlocks pancake's AVX-512 VNNI/VBMI kernels automatically (no build flag needed beyond a normal build). The repo's own measurements show the gain is modest at 1 token (bandwidth-bound) and large at 4-16 tokens (MTP/speculative verify windows of ~3 tokens in the plan, so ~2-3x on CPU expert rows is plausible on Zen 4/5; repo MOD_PLAN 10.1 cites 3.05-token windows).
- Zen 5 desktop full-width matters more for multi-token batches (compute-bound) than single-token decode (memory-bound). For pancake's batched verify windows a 9000-series likely beats a same-core-count 7000-series beyond its bandwidth gain; at 1 token they converge on DRAM bandwidth.
- On Zen 5, the AVX-512 path wins dispatch; the AVX-VNNI path only matters if AVX-512 is forced off (STRATA_FORCE_AVX2=1) — this is an inference from the dispatch order seen in expert_layout.cpp 470-477 (cpu_avx512_ok checked before cpu_avx2_ok).
- Zen 3 (incl. 5800X3D) has neither AVX-512 nor AVX-VNNI, so an AM4 CPU swap does not change ISA path, only PCIe gen/cache/cores.

### Gaps
- Whether Strix Halo's Zen 5 cores are full 512-bit (desktop-style CCDs) or 256-bit like Strix Point was not confirmed in this pass (numberworld covers Granite Ridge vs Strix Point only). Zen 4 8000G (Phoenix) AVX-512 assumed same as Zen 4 desktop; not separately verified.
- Zen 5 `avx_vnni` cpuinfo flag string not verified from a Zen 5 dump.

## Q4. Cores/SMT, L3 and X3D, clocks/power

### Takeaway
Consumer AMD tops out at 16 cores/32 threads (two CCDs) on AM4, AM5 and Strix Halo. X3D parts have 96 MB L3 on one CCD (9800X3D) or 128 MB total with V-Cache on one CCD (9950X3D), and a 9950X3D2 with V-Cache on both CCDs (~192 MB L3). Strix Halo is 16C/32T Zen 5, 45-120 W cTDP; Strix Point is 12 cores (4 Zen 5 + 8 Zen 5c), 24 MB L3.

### Cited Findings
- Ryzen 7 9800X3D: 8 cores, 96 MB L3 + 8 MB L2 — [HotHardware](https://hothardware.com/news/amd-reveals-ryzen-9-9950x3d-compare-9800x3d-games)
- Ryzen 9 9950X3D: 128 MB L3 + 16 MB L2; V-Cache on only one CCD, the other has 32 MB L3 — [HotHardware](https://hothardware.com/news/amd-reveals-ryzen-9-9950x3d-compare-9800x3d-games); [TechPowerUp review](https://techpowerup.com/review/amd-ryzen-9-9950x3d)
- Ryzen 9 9950X3D2: V-Cache on both CCDs, 208 MB total L2+L3 — [StorageReview](https://www.storagereview.com/review/amd-ryzen-9-9950x3d2-dual-edition-review-3d-v-cache-on-both-ccds)
- Ryzen 9 9950X: two CCDs + IO die (chiplet layout since Zen 2) — [Chips and Cheese 9950X](https://old.chipsandcheese.com/2024/08/14/amds-ryzen-9950x-zen-5-on-desktop/)
- Ryzen AI Max+ 395: 16 Zen 5 cores / 32 threads, 3.0 GHz base / 5.1 GHz boost, 40 RDNA 3.5 CUs (Radeon 8060S), 45-120 W configurable TDP (55 W in AMD's test config); L3 64 MB per Notebookcheck (AMD quotes 80 MB total cache) — [PC Gamer](https://pcgamer.com/hardware/processors/amd-ryzen-ai-max-is-finally-here-the-most-advanced-mobile-x86-processor-ever-created-with-40-rdna-3-5-cus-and-16-zen-5-cores); [Notebookcheck](https://www.notebookcheck.com/AMD-Ryzen-AI-Max-395-Prozessor-Benchmarks-und-Specs.942325.0.html); [AMD how-to-sell guide](https://www.amd.com/content/dam/amd/en/documents/partner-hub/ryzen/ryzen-ai-max-series-how-to-sell-guide-competitive.pdf)
- Ryzen AI 9 HX 370: 12 cores (4 Zen 5 + 8 Zen 5c), 2.0/5.1 GHz, 12 MB L2, 24 MB L3, cTDP 15-54 W, Radeon 890M (16 CUs) — [Notebookcheck](https://www.notebookcheck.net/AMD-Ryzen-AI-9-HX-370-Processor-Benchmarks-and-Specs.836729.0.html); [PC Gamer](https://www.pcgamer.com/hardware/processors/amd-launches-killer-new-12-core-zen-5-apu-for-laptops-and-handhelds/)
- Pancake measured no gain from L3 prefetch (STRATA_ARENA_WARM was worse) — [repo:MOD_PLAN.md line 418]

### Inferences
- A 1.4 MB expert: 96 MB L3 holds ~65 experts; 128 MB V-Cache CCD ~90. That is small vs a per-layer expert pool, so X3D helps only if a very hot subset is reused within the L3 lifetime (e.g. shared experts, or the same routed expert across tokens in one verify window). The repo's negative L3-prefetch result suggests weights are streamed once per token and L3 residency doesn't help; X3D is more likely to help activations/KV/router state than expert weights. Treat as unproven; measure with a 5800X3D/9800X3D if available.
- 5700G has 16 MB L3 (training knowledge, not verified this pass) vs 32 MB per CCD on desktop Zen 3 — minor for pancake for the same reason.
- 9950X3D's asymmetric CCDs complicate thread pinning; the 9950X3D2 or a plain 9950X is simpler for a bandwidth-bound engine.

### Gaps
- 5800X3D/7800X3D L3 (96 MB) and 5700G L3 (16 MB), core counts of AM4/AM5 SKUs, and official TDPs not cited from sources this pass (amd.com unreachable). AVX-512 clock behavior on Zen 4 not checked.

## Q5. iGPU/NPU, ROCm iGPU support, Resizable BAR/SAM, IOMMU defaults, unified memory

### Takeaway
ROCm officially supports the Strix Halo iGPU (gfx1151) and lists Strix Point (gfx1150) in recent releases; desktop APU iGPUs (Vega gfx90c in the 5700G, RDNA3 780M gfx1103 in the 8000G) are not in the matrices found. Resizable BAR is active on the current box (32 GB BAR for the 20 GB 7900 XT). Strix Halo removes PCIe from the expert path entirely: the GPU "VRAM" is GTT carved from the same LPDDR5X pool, and pancake/Strata already has a dedicated UMA path.

### Cited Findings
- ROCm matrices list gfx1151 for Ryzen AI Max (PRO) 300 series on Ubuntu 24.04.3 (HWE kernel 6.14) — [ROCm 7.9.0 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.9.0/install/compatibility-matrix.html)
- ROCm 7.11.0 preview lists Ryzen AI 9 HX 375/370/365 under gfx1150 and AI Max PRO 395/390/385/380 under gfx1151 — [ROCm 7.11.0 release notes](https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html)
- gfx1151 needs specific KFD kernel fixes; the table covers AMD prebuilt binaries only — [ROCm RDNA3.5 system optimization](https://rocm.docs.amd.com/en/develop/how-to/system-optimization/rdna3-5.html)
- Radeon/Ryzen Linux matrix covers gfx1150/gfx1151; its ROCm 7.2.1 entry lists only Ryzen AI Max+ 395 — [Radeon-Ryzen Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
- No mention of gfx1103 (780M) or gfx90c in the matrices searched — same sources (absence)
- Vega iGPU (gfx90c) on the current box is Vulkan-only under ROCm 10.0.0 — [local memory notes, MEMORY.md] (user's own observation)
- Strata ships a `rocm-gfx1151` docker variant ("halo"); Strata's DeepSeek engine runs on AMD only gfx1201 and gfx1151 (gfx1100 to be added in pancake) — [repo:docs/DOCKER.md 24, 339; MOD_PLAN.md 156]
- Strix Halo in Strata: GPU "VRAM" is the GTT pool of the same RAM; `device_free_bytes()` counts MemAvailable minus 6 GiB; unified memory counted once; copying CPU-lane experts into RAM on the iGPU overcommits shared memory and stalled an 8K prompt; gfx11 WMMA kernels for prompt paths — [repo:docs/STRIX_HALO.md 91-104, 236-238]
- Strata README: Ryzen AI Max+ 395, 128 GB unified: 19-21 tok/s (code), 11 (chat), prompt 87-125 tok/s — [repo:README.strata.md 83]
- Strix Halo CPU→GPU copy bandwidth is high, GPU→CPU is not; copies to the GPU bypass Infinity Cache; GPU sees host memory without the ~1 us penalty a dGPU (RX 9070) pays — [C&C Strix Halo memory subsystem](https://old.chipsandcheese.com/?p=37437)
- Strix Halo NPU: up to 50 TOPS — [AMD blog](https://www.amd.com/en/blogs/2025/experience-unparalleled-performance-with-the-amd-ryzen.html)
- Resizable BAR on current box: 7900 XT BAR0 = 32 GB prefetchable at 0x7000000000 (i.e. full-VRAM BAR, ReBAR/SAM active even on Zen 3 APU/PCIe 3.0) — [local: lspci -vv -s 12:00.0]
- IOMMU: current kernel cmdline has no `amd_iommu=` / `iommu=` override (kernel/BIOS defaults apply); Strix Halo maintainers boot with `amd_iommu=off` — [local: /proc/cmdline]; [repo:docs/STRIX_HALO.md 137]

### Inferences
- Unified memory APUs remove the PCIe promotion cost entirely: "promotion" becomes a no-op (or a GTT remap), so pancake's CPU-vs-promote trade-off collapses into "which compute unit reads the weights from the same DRAM". The bottleneck becomes shared DRAM bandwidth (256 GB/s theoretical) contended by CPU and iGPU, with CPU latency rising to ~200+ ns under GPU load (C&C). CPU expert compute should then be used only when the iGPU is saturated, not to avoid transfers.
- Strix Halo's 40-CU iGPU with ~256 GB/s is much slower than the 7900 XT's local GDDR6 for hot experts, so the trade is: 7900 XT box (fast VRAM, 20 GB, PCIe-bound misses) vs Strix Halo (slow-but-flat 128 GB). Strata's measured 19-21 tok/s on Strix Halo is the reference point for that comparison.
- iGPU of desktop APUs (5700G Vega, 8000G 780M) is not useful as a ROCm compute target for pancake; at best Vulkan.
- ReBAR is already enabled, so CPU-side direct writes into VRAM are possible on the current box; upgrading platform won't add this. IOMMU on: for peer/DMA-heavy NVMe→VRAM paths, `iommu=pt` or off can matter (Strix Halo maintainers chose off); not measured on dGPU here.
- NPU (XDNA 2, 50 TOPS on Strix Halo/Point) has no role in pancake today (no repo references found).

### Gaps
- ROCm status for gfx1103 (780M) and gfx90c not found in official docs (absence only); the ROCm 10.x matrix on this machine was not fetched.
- Resizable BAR support per AM4 chipset/CPU (official AMD SAM list) and kernel IOMMU defaults for Ubuntu (CONFIG_IOMMU_DEFAULT_PASSTHROUGH or not) not verified this pass.
- No measured NVMe→VRAM (P2P/DirectStorage-like) numbers on AMD consumer platforms found.

## Summary matrix (compiled from sections above; see citations there)

| Platform | GPU slot | CPU NVMe | DRAM (ch x width) | Measured CPU read BW | Max RAM | ISA for pancake | L3 | ROCm iGPU |
|---|---|---|---|---|---|---|---|---|
| AM4 5700G (Zen 3 APU) — current | 3.0 x16 | 3.0 x4 | DDR4 2ch | ~50 GB/s class (AIDA, Zen 3) | 128 GB (unverified) | AVX2 only | 16 MB (unverified) | No (Vega gfx90c, Vulkan) |
| AM4 Zen 3 desktop / 5800X3D on B550/X570 | 4.0 x16 | 4.0 x4 | DDR4 2ch | 50.7-54.7 GB/s AIDA | 128 GB (unverified) | AVX2 only | 32 MB/CCD; 96 MB X3D (unverified) | n/a |
| AM4 on B450/X470 | 3.0 x16 | 3.0 x4 | DDR4 2ch | same | same | AVX2 | same | n/a |
| AM5 Zen 4 7000 / X3D | 5.0 x16 (X670E/B650E/X870/E), else 4.0 | 5.0 x4 on E-boards/B850, else 4.0 | DDR5 2ch | ~64 GB/s per CCD read cap | 192-256 GB (4x64, ~6000 MT/s) | AVX-512 VNNI/VBMI/BF16, double-pumped | 32 MB/CCD; 96 MB X3D | n/a |
| AM5 Zen 4 8700G/8600G | 4.0 x8 | 4.0 x4/x4 | DDR5 2ch | not found | as AM5 | AVX-512 (Zen 4, assumed) | not found | 780M gfx1103: not in matrix |
| AM5 8500G/8300G | 4.0 x4 | 4.0 x4/x2 | DDR5 | not found | as AM5 | AVX-512 (Zen 4 + 4c, assumed) | — | not in matrix |
| AM5 Zen 5 9000 / X3D | as chipset | as chipset | DDR5 2ch | ~60 GB/s AIDA @5000 (16C); 64 GB/s/CCD cap | 256 GB | AVX-512 full 512-bit + AVX-VNNI | 32 MB/CCD; 96 MB 9800X3D; 128 MB 9950X3D; ~192 MB X3D2 | n/a |
| Strix Point HX 370 | 4.0 (lanes ?) | 4.0 | LPDDR5X-7500 / DDR5-5600 | GPU 96 GB/s (Vulkan) | up to 256 GB listed | AVX-512 on 256-bit HW | 24 MB | gfx1150 in ROCm 7.11 preview |
| Strix Halo AI Max+ 395 | 4.0, 16 lanes (PRO) | 4.0 | LPDDR5X-8000 256-bit, 256 GB/s theor. | ~103 GB/s from one CCD | 128 GB | AVX-512 (Zen 5; width unverified) | 64 MB | gfx1151 supported |
