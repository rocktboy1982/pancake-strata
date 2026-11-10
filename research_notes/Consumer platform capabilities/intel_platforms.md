# Intel consumer and hobbyist-HEDT platforms: capability matrix for pancake (hybrid CPU+GPU MoE streaming)

Reference box for every comparison: AMD AM4 Ryzen 7 5700G, dual-channel DDR4, RX 7900 XT on PCIe 3.0 x16. The figures below come from the user's own measurements, not from outside sources: CPU expert compute reaches about 27 GB/s, roughly 80% of 33 GB/s. Promoting one 1.4 MB expert over PCIe 3.0 takes about 115 us, against about 42 us to compute it on the CPU.

Repo facts were read from /home/dan/clone-of-d/src/Strata (read-only, file:line given). Web facts have inline links. Some facts were taken from search-result summaries of a page rather than a full fetch; those carry "(search summary)" so the writer knows the confidence level.

---

## Q1. Platform matrix: PCIe, DMI, memory, caches, ISA, topology, capacity

### Takeaway
Pancake's balance is set by two numbers: DRAM bandwidth, which drives CPU expert compute, and the PCIe link to the GPU, which drives promotion. Against the 5700G reference:
- LGA1151 and 10th-gen LGA1200 match it on PCIe 3.0 and dual-channel DDR4.
- Rocket Lake (11th gen) adds PCIe 4.0 and AVX-512.
- LGA1700 on DDR5 roughly doubles to triples the usable DRAM bandwidth.
- Arrow Lake (LGA1851) adds faster DDR5 and drops SMT.
- Used HEDT has the most bandwidth: Sapphire Rapids W-3400 with 8-channel DDR5, or Broadwell-EP with quad-channel DDR4. Only Sapphire Rapids also gets pancake's AVX-512 path.

### Cited Findings

**LGA1151 (Coffee Lake 8th/9th gen, e.g. i9-9900K)**
- 16 CPU PCIe 3.0 lanes, used as x16 for the GPU. The chipset link is DMI 3.0, 4 lanes at 8 GT/s. Max memory is 128 GB, rated DDR4-2666 (spec listing: [hmc-tech](https://hmc-tech.com/cpus/intel-core-i9-9900k), search summary).
- No CPU-attached NVMe lanes, so every M.2 slot hangs off the chipset behind DMI 3.0 x4. Same source as above; this is the standard Coffee Lake layout.

**LGA1200, 10th gen Comet Lake (e.g. i9-10900K)**
- 16 PCIe 3.0 lanes, max memory 128 GB, DDR4-2933, no ECC ([Intel ARK i9-10900K](https://www.intel.com/content/www/us/en/products/sku/199332/specifications.html), search summary).
- DMI 3.0 is "comparable to PCI-Express 3.0 x4" in bandwidth ([TechPowerUp 10900K review](https://techpowerup.com/review/intel-core-i9-10900k/3.html), search summary).
- Lower-end Comet Lake parts stay at DDR4-2666 ([Guru3D 10900K review](https://www.guru3d.com/articles-pages/intel-core-i9-10900k-processor-review,2.html), search summary).

**LGA1200, 11th gen Rocket Lake (e.g. i9-11900K)**
- 20 PCIe 4.0 lanes from the CPU, x16 for the GPU plus x4 for NVMe ([AnandTech preview](https://www.anandtech.com/show/16390), [AnandTech launch](https://www.anandtech.com/show/16523)).
- Adds AVX-512 ([AnandTech Rocket Lake review](https://at-web1.www.anandtech.com/show/16495/intel-rocket-lake-14nm-review-11900k-11700k-11600k/2)).
- Z590 widens DMI from x4 to x8, effectively PCIe 3.0 x8 (same AnandTech review). The 500-series chipset lanes themselves are PCIe 3.0 only ([Tom's Hardware](https://www.tomshardware.com/reviews/intel-core-i9-11900k-and-i5-11600k-review/3)).
- Resizable BAR came to 10th and 11th gen via board-vendor BIOS on 400/500-series boards: Above 4G Decoding on, CSM off ([Gigabyte BIOS list](https://www.gigabyte.com/WebPage/785/NVIDIA_resizable_bar.html); [PC Gamer, Z490/B460/H410 updates](https://www.pcgamer.com/intel-motherboards-z490-b460-h410-resize-bar-support); [TechSpot, Intel brings ReBAR to Rocket Lake](https://www.techspot.com/community/topics/intel-says-it-will-bring-resizable-bar-to-11th-gen-rocket-lake-cpus.267699)).

**LGA1700, 12th-14th gen Alder Lake / Raptor Lake (e.g. i9-14900K)**
- 20 CPU lanes: 16 PCIe 5.0 plus 4 PCIe 4.0. Configurations are 1x16+4 or 2x8+4. DMI 4.0 x8 to the chipset ([ocinside 14900K spec](https://www.ocinside.de/review/intel_core_i9_14900k/2/), search summary).
- Dual-channel memory, DDR5-5600 or DDR4-3200, max 192 GB (same source; [memoryexpress listing](https://www.memoryexpress.com/Products/MX00126901)).
- Caches, Alder Lake (12900K): 1.25 MB L2 per P-core, 2 MB per 4-core E-cluster, 30 MB L3 ([AnandTech Alder Lake](https://at-web1.www.anandtech.com/show/16959/intel-innovation-alder-lake-november-4th/2)).
- Caches, Raptor Lake: 2 MB L2 per P-core, 4 MB per E-cluster, 36 MB L3 on the 13900K ([AnandTech 13900K review](https://at-web1.www.anandtech.com/show/17601/intel-core-i9-13900k-and-i5-13600k-review/2)).
- AVX-512 was removed so P-cores and E-cores share one ISA:
  - Early BIOSes exposed it on the P-cores when the E-cores were disabled; Intel then fused it off ([AnandTech 12900K review, "Disabled AVX-512, but Not Really"](https://at-web1.www.anandtech.com/show/17047/the-intel-12th-gen-core-i912900k-review-hybrid-performance-brings-hybrid-complexity/2); [HotHardware, fusing off](https://hothardware.com/news/intel-fusing-avx-512-alder-lake-silicon)).
  - A community wiki dates the fused silicon from about 2022 week 2 ([zingaburga/alderlake_avx512](https://github.com/zingaburga/alderlake_avx512/wiki/Home/e193914970edb39fada75d9951c201ed978149e7)).
- AVX-VNNI (the 256-bit VEX form) is on both the P-cores and the E-cores ([AnandTech Alder Lake deep dive, ISA page](https://at-web1.www.anandtech.com/show/16881/a-deep-dive-into-intels-alder-lake-microarchitectures/5)).
- Measured memory bandwidth on the 12900K:
  - About 50 GB/s read on DDR4 and about 82 GB/s read on DDR5 ([CapFrameX](https://www.capframex.com/tests/Alder%20Lake-S%2012900K%20is%20the%20new%20Gaming%20King), search summary).
  - A second source spans about 33.5 GB/s (DDR4-2400 Gear 2) to about 88 GB/s (DDR5-6200) ([cloudspress](https://www.cloudspress.com/?p=578607), search summary, lower-authority).

**LGA1851, Arrow Lake / Core Ultra 200S (e.g. 285K)**
- 20 PCIe 5.0 lanes plus 4 PCIe 4.0 from the CPU, and DMI 4.0 x8 to the chipset ([TechSpot 285K review](https://www.techspot.com/review/2911-intel-core-ultra-9-285k-review/), search summary).
- Sources conflict on the chipset lane count: HotHardware says Z890 adds 4 lanes, Tom's Hardware says 20, TechPowerUp says 48 for the whole platform ([HotHardware](https://hothardware.com/reviews/intel-core-ultra-200s-arrow-lake-cpu-review?page=2); [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/intel-core-ultra-9-285k-cpu-review); [TechPowerUp](https://techpowerup.com/review/intel-core-ultra-9-285k/4.html)).
- Intel acknowledged higher latency on the Gen5 root port for lanes 21-24, which sit on the I/O tile, compared with lanes 1-16 (reported in the same review set, search summary). Those x4 Gen5 lanes are the CPU M.2 slot.
- No Hyper-Threading. 8 Lion Cove P-cores and 16 Skymont E-cores. 13 TOPS NPU ([TechSpot](https://www.techspot.com/review/2911-intel-core-ultra-9-285k-review/); [HWCooling](https://www.hwcooling.net/en/intel-core-ultra-9-285k-1-in-multi-threaded-tasks-even-without-ht-review/), search summary).
- Caches and topology ([Chips and Cheese, Arrow Lake system level](https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the), fetched):
  - 3 MB L2 per P-core, 4 MB L2 per 4-core E-cluster, 36 MB L3 (12 slices of 3 MB).
  - Each E-cluster has its own ring stop.
  - The iGPU has 4 Xe cores, on its own tile.
- Bandwidth and latency behaviour (same Chips and Cheese article):
  - DRAM read bandwidth "approaches theoretical"; writes are not halved.
  - Any group of P-cores can reach full DRAM bandwidth.
  - DRAM latency is above 100 ns, worse than Raptor Lake.
  - E-cores see higher idle latency because the memory controller drops into a low-power mode unless a P-core is active.
- Measured bandwidth on a 265K ([Igor's Lab](https://www.igorslab.de/en/arrow-lake-ddr5-detail-memory-clock-uncore-timings-core-ultra-7-265k-two-platforms/), search summary):

  | Memory | Theoretical | Measured |
  |---|---|---|
  | DDR5-5600 | 89.6 GB/s | about 76 GB/s |
  | DDR5-6000 CL28 | not given | about 90 GB/s |
  | DDR5-7800 CL36 | not given | about 118 GB/s |

- CUDIMM DDR5 at 8000-10000+ is supported or demonstrated ([Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/intels-core-ultra-200-desktop-cpus-reportedly-support-ddr5-10000-ram-arrow-lake-s-takes-advantage-of-cudimms-with-clock-drivers)).

**HEDT / workstation, Sapphire Rapids Xeon W-3400 (e.g. w9-3495X, W790)**
- 8 channels of DDR5-4800, RDIMM only, ECC, up to 4 TB. 307.2 GB/s theoretical. 112 PCIe 5.0 lanes plus DMI x8 to W790 ([AnandTech launch](https://anandtech.com/show/18741); [ServeTheHome](https://www.servethehome.com/new-workstation-kingpins-intel-xeon-w-3400-xeon-w-2400-and-w790-launch/); [Intel pre-brief PDF](https://www.igorslab.de/wp-content/uploads/2023/02/Intel-Xeon-W-3400-and-Xeon-W-2400-Workstation_PressPre-Briefing_Rev1.0_POST-EMBARGO-1.pdf)).
- The memory speed falls to DDR5-4400 at 2 DIMMs per channel ([Tom's Hardware, SPR specs](https://www.tomshardware.com/news/intel-sapphire-rapids-xeon-scalable-specifications-and-features)).
- AMX with INT8 and BF16 ([ServeTheHome](https://www.servethehome.com/new-workstation-kingpins-intel-xeon-w-3400-xeon-w-2400-and-w790-launch/)).
- L2 is 2 MB per core ([Chips and Cheese, Sapphire Rapids](https://chipsandcheese.com/p/a-peek-at-sapphire-rapids)).
- Measured llama.cpp gain from AMX: about 57 vs 28 tok/s on a 3B Q4/Q8 model. This is vendor-adjacent and the method is unspecified ([OpenMetal](https://openmetal.io/resources/blog/intel-amx-ai-inference-performance/)).
- Users report that llama.cpp sometimes detects only AMX_INT8, and the AMX buffer is unusable in MinGW builds ([Intel community](https://community.intel.com/t5/Intel-Xeon-Processor-and-Server/llama-cpp-How-to-enable-AMX-on-Windows-11-when-using-llama-cpp/m-p/1690197)).

**HEDT, Broadwell-EP Xeon E5 v4 / Haswell-EP v3 (used, LGA2011-3)**
- Quad-channel DDR4-2400: about 77 GB/s theoretical per socket ([cpu-monkey E5-2699 v4](https://www.cpu-monkey.com/en/cpu-intel_xeon_e5_2699_v4)).
- Two sockets reach about 119 GB/s on STREAM triad, and v4 is about 15% faster than v3 with icc ([AnandTech Xeon E5 v4 review](https://www.anandtech.com/show/10158/the-intel-xeon-e5-v4-review/9)).
- ISA is AVX2/FMA only. No AVX-512 and no VNNI (well known for Haswell/Broadwell; no separate citation fetched, see Gaps).

**Mobile (brief)**
- Lunar Lake:
  - On-package LPDDR5X-8533, 16 or 32 GB. The memory controller shares a tile with the cores. SMT was removed from Lion Cove. 4 Skymont cores on a low-power island with 4 MB L2 ([Chips and Cheese, Lion Cove](https://old.chipsandcheese.com/2024/09/27/lion-cove-intels-p-core-roars/); [Notebookcheck](https://www.notebookcheck.net/Intel-Core-Ultra-7-258V-Processor-Benchmarks-and-Specs.892883.0.html)).
  - Bus width is disputed: 128-bit per AnandTech, 160-bit per TechPowerUp ([AnandTech](https://www.anandtech.com/show/21425/intel-lunar-lake-architecture-deep-dive-lion-cove-xe2-and-npu4); [TechPowerUp](https://techpowerup.com/review/intel-lunar-lake-technical-deep-dive/single-page.html)).
- Meteor Lake runs LPDDR5-7467, and 4 Lion Cove cores on Lunar Lake pull more bandwidth than all of Meteor Lake's cores ([Chips and Cheese, Lion Cove](https://old.chipsandcheese.com/2024/09/27/lion-cove-intels-p-core-roars/)).

**Errata relevant to gather-heavy kernels**
- Downfall (GDS) affects AVX2/AVX-512 gathers on older Intel parts, roughly Skylake through Ice Lake/Tiger Lake/Rocket Lake. The microcode fix is "costly" for hot gather paths ([Phoronix, Downfall](https://Phoronix.com/review/downfall); [Phoronix benchmarks](https://phoronix.com/review/intel-downfall-benchmarks)).
- Intel reportedly told partners the slowdown could reach 50%. A TechPowerUp write-up of Phoronix's testing gives 11-39% on an i7-1165G7 (search summary, not confirmed on Phoronix's own page). The latest Intel client and server CPUs are not affected (Phoronix).

### Matrix (synthesised from the findings above; "?" means not verified this session)

| Platform | GPU link | NVMe from CPU | DMI to chipset | Memory (ch x type, rated) | Theoretical / measured BW | L2 / L3 (top SKU) | ISA relevant to pancake | Cores / SMT | Max RAM |
|---|---|---|---|---|---|---|---|---|---|
| LGA1151 8/9th | PCIe 3.0 x16 | none (chipset) | DMI 3.0 x4 (~3.9 GB/s) | 2x DDR4-2666 | 42.7 / ? | 256 KB per core / 16 MB (? not fetched) | AVX2, FMA | homogeneous, HT | 128 GB |
| LGA1200 10th | PCIe 3.0 x16 | none (chipset) | DMI 3.0 x4 | 2x DDR4-2933 | 46.9 / ? | 256 KB / 20 MB (ARK title "20M cache") | AVX2 | homogeneous, HT | 128 GB |
| LGA1200 11th RKL | PCIe 4.0 x16 | x4 Gen4 | DMI 3.0 x8 (Z590) | 2x DDR4-3200 (?) | 51.2 / ? | ? | AVX2 + AVX-512 (Ice Lake core set) | homogeneous, HT | 128 GB (?) |
| LGA1700 12-14th | PCIe 5.0 x16 | x4 Gen4 | DMI 4.0 x8 (~15.8 GB/s) | 2x DDR5-5600 or DDR4-3200 | 89.6 / ~82 (DDR5), ~50 (DDR4) | 2 MB per P, 4 MB per E-cluster / 36 MB | AVX2 + AVX-VNNI, no AVX-512 | 8P (HT) + up to 16E | 192 GB |
| LGA1851 200S | PCIe 5.0 x16 | x4 Gen5 (higher latency) + x4 Gen4 | DMI 4.0 x8 | 2x DDR5-6400 (CUDIMM 8000+) | 89.6 at 5600 / 76-118 measured | 3 MB per P, 4 MB per E-cluster / 36 MB | AVX2 + AVX-VNNI, no AVX-512 | 8P + 16E, no SMT | 192 GB (? see Gaps) |
| W-3400 SPR | PCIe 5.0, 112 lanes | many CPU lanes | DMI x8 to W790 | 8x DDR5-4800 RDIMM | 307 / ? | 2 MB per core / 105 MB | AVX-512 (full incl. VBMI, VNNI, BF16), AMX | homogeneous, HT | 4 TB |
| E5 v3/v4 | PCIe 3.0 (40 lanes, ?) | CPU lanes via adapters | DMI 2.0 (?) | 4x DDR4-2133/2400 | 77 / about 60 per socket (inferred from 119 for 2S) | 256 KB / up to 55 MB (?) | AVX2 only | homogeneous, HT | 768 GB-1.5 TB (?) |

### Inferences
- The 5700G reference box is a "PCIe 3.0 + 2ch DDR4" machine, the same class as LGA1151 and 10th-gen LGA1200. Pancake's tuning should carry over almost unchanged to Coffee Lake and Comet Lake.
- On LGA1151 and LGA1200-10th, NVMe expert streaming goes through DMI 3.0 x4 (about 3.9 GB/s ceiling). That ceiling is shared with SATA, USB and NIC traffic. On AM4 the reference box has CPU-attached NVMe lanes, so these Intel boards will be worse for NVMe-tier streaming.
- LGA1700, LGA1851 and Rocket Lake all have a CPU-attached x4 NVMe slot. Use that slot, not a chipset M.2, for the model folder.
- The RX 7900 XT is a PCIe 4.0 x16 card (AMD spec; not fetched this session). In any LGA1700 or LGA1851 Gen5 slot it links at Gen4. Only a Gen5 GPU benefits from Gen5, for example RTX 50-series or an Arc B-series card if Gen5. Check this per card.
- The "x16 + x4" split on LGA1700 can fall to 2x8 when the second CPU slot is populated (ocinside). A user with two GPUs gets x8/x8. For PCIe 5.0 cards that is still equal to Gen4 x16.

### Gaps
- Not found or fetched: independent measured STREAM/AIDA bandwidth for Sapphire Rapids W-3400 (KitGuru ran AIDA64, but the numbers were not retrieved), single-socket E5 v4, Rocket Lake and Coffee Lake.
- Not verified this session: 40 PCIe 3.0 lanes on E5 v4, the Rocket Lake memory rating and max RAM, Arrow Lake max RAM (192 GB vs 256 GB after updates), and the L3 sizes for Coffee Lake, Comet Lake, Rocket Lake and E5. These are marked "?" in the table.
- No reliable source found on VT-d/IOMMU defaults per platform. Board-vendor dependent: often enabled by default on Z690 and later boards, often disabled on older consumer boards. Treat this as unverified.

---

## Q2. What each capability means for pancake

### Takeaway
On DDR5 Intel desktops (LGA1700 and LGA1851), CPU expert compute should run about 2.5-3.5x faster than on the reference box. A Gen4 GPU link only halves promote time. So the CPU-vs-promote break-even moves strongly toward "compute missed experts on the CPU, promote fewer". The opposite holds on PCIe 3.0 + DDR4 Intel boards, which behave like the reference box.

Hybrid cores need `--pool-affinity auto` or `p-cores` and worker counts below the core count. Pancake already has the detection code, but it is opt-in. Pancake's own AVX-512 path runs only on Rocket Lake, Ice Lake, Sapphire Rapids and later Xeons. Skylake-X and Cascade Lake HEDT fail the VBMI check. No AMX path exists in pancake's own expert kernels.

### Cited Findings (repo, read-only)

**ISA selection**
- `cpu_isa_cap()` reads `STRATA_FORCE_ISA`: `avx2`=2, `avx`=1, `sse`/`sse4.2`/`none`=0, unset=3. Any other value is ignored with a warning. It affects only the engine's own dispatch, not ggml-cpu (src/kernels/cpu/expert_layout.cpp:26-40).
- `STRATA_FORCE_AVX2=1` also disables the AVX-512 probes (expert_layout.cpp:75-78, 121).
- `cpu_avx512_ok()` requires all of the following (expert_layout.cpp:80-111):
  - OSXSAVE, and XCR0 & 0xE6 (the OS saves ZMM state)
  - AVX512F (leaf 7 EBX bit 16), BW (bit 30), VL (bit 31)
  - VNNI (ECX bit 11) and VBMI (ECX bit 1)

  Consequences:
  - Skylake-X / Cascade Lake HEDT (i9-7980XE, i9-10980XE, Xeon W-2200/3200): no VBMI, so they fall back to AVX2.
  - Rocket Lake, Ice Lake-SP and Sapphire Rapids pass.
  - Alder/Raptor/Arrow Lake: no AVX-512 visible, so AVX2.
- The build compiles the AVX-512 kernels per source file with `-mavx512f/bw/vl/dq/vnni/vbmi/vpopcntdq` (MSVC `/arch:AVX512`), so one binary serves every CPU and dispatch happens at run time. Files: iq_avx512.cpp, nvfp4_avx512.cpp, tq_avx512.cpp (CMakeLists.txt:2226-2260). The comment at :1956 says the expert kernel uses `_mm256_dpbusd_epi32` and `_mm256_multishift_epi64_epi8`.
- AVX-VNNI copies of the AVX2 kernels (q2_avx2.cpp, iq_avx2.cpp, *_rows.inl) are compiled per function with `target("avxvnni")` and taken at run time when `cpu_avxvnni_ok()` (CMakeLists.txt:2117-2145, 2272). This is the path Alder, Raptor and Arrow Lake get.
- `cpu_gather_fast()` returns true only when all hold (expert_layout.cpp:320-340):
  - the CPU is GenuineIntel
  - it has AVX-VNNI (CPUID 7.1 EAX bit 4)
  - it is not an E-core-only part (ADL-N/Twin Lake 6/BEh, Grand Ridge, Sierra Forest, Clearwater Forest)

  So Rocket Lake, Skylake-X, Cascade Lake and other Downfall-affected AVX-512 parts do not get gathers by default, while Alder Lake and later P-cores and Sapphire Rapids do.
- `cpu_gather_fast_here()` checks the core type per thread via CPUID leaf 0x1A (0x40 = P-core, 0x20 = E-core). On hybrid parts only threads on P-cores use the gathered IQ decode (expert_layout.cpp:342-359; iq_avx2.cpp:16, 523).
- The override knobs are `STRATA_IQ256_GATHER` and `STRATA_IQ_GATHER`. The iq_avx512 gather path defaults on where `cpu_gather_fast()` and off on Zen 4 (microcoded) (iq_avx512.cpp:40-44; native_expert.cpp:442-450).
- AMX: a grep for amx/_tile_ in src/include/sycl finds only the vendored `src/dsv41/vendor/exl3_cpu.cpp` plus CUDA files. The ggml build flags `GGML_AMX_TILE/INT8/BF16` exist, but they are forced OFF only in the ISA-floor build; otherwise ggml uses GGML_NATIVE (CMakeLists.txt:2190-2206). Pancake's own expert kernels have no AMX path.
- Old Q2_0 model folders (no `native_experts.txt`) need AVX-512 for the CPU expert kernel. Newer IQ3/NVFP4/Unsloth folders run AVX2 on Ryzen 3000-5000 and Core 12th-14th gen (docs/INTEL_ARC.md:51-55).

**Hybrid pools**
- `PoolAffinity` has three values. `All` is the default: one worker per physical core in OS order, no hybrid distinction. `Auto` and `PCores` are opt-in via `--pool-affinity auto|p-cores` (include/strata/kernels/cpu/pool.hpp:72-78).
  - With Auto or PCores on a hybrid CPU, the worker count defaults to p_cores-1 (pool.cpp:824).
  - In Auto mode E-cores are used only as overflow (pool.cpp:367-380).
  - With All on a hybrid CPU, the P-cores are still sorted first (#642, pool.cpp:351-357, 531-532).
- E-core detection on Linux, in order (pool.cpp:421-507; pool_plan_test.cpp:104-118):
  - the hybrid PMU lists `/sys/devices/cpu_core/cpus` and `cpu_atom/cpus`
  - `cpu_lowpower/cpus` for LP-E cores
  - fallback: `cpu_capacity` below 90% of the maximum
- On Windows, detection uses `EfficiencyClass` (pool.cpp:304-339).
- Planner defaults (pool_plan_test.cpp:86-100):
  - 8P+12E (Core Ultra 7 265) gives 13 workers; the comment says "measured: 12-16 beat the 19 of one per core".
  - 8P+16E gives 15 workers.
  - 8P+8E gives every core but the host's.
- The tuner offers `--pool-affinity` only when a hybrid CPU is detected (strata_control/tune_knobs.py:164).
- `STRATA_POOL_UMWAIT` (experimental): parked workers use WAITPKG UMONITOR/UMWAIT instead of spinning. WAITPKG is present on Alder Lake and later and on Sapphire Rapids (pool.cpp:42-73). Freeing the spinning cores could leave more power and turbo headroom for the busy ones on a hybrid part (code comment intent; not measured here).

**SYCL / oneAPI path and iGPU / NPU**
- `sycl/` is a SYCL port of the engine for discrete Arc A-series and B-series. Integrated Arc GPUs (B390 / Panther Lake, #515) are "not tested by anyone", and "shared-memory planning does not exist" (docs/INTEL_ARC.md:179-181).
- Windows has no native SYCL path (docs/INTEL_ARC.md:183-187).
- The fused int8 DPAS expert kernels need the Xe2 s8 8x16x32 matrix combination (Battlemage, Lunar Lake). There is no DP4a fallback; without DPAS the FP16 path runs (docs/INTEL_INT8_EXPERTS.md:43-46).
- The SYCL build needs oneAPI 2025.3+ icpx and oneMKL plus Level Zero (docs/INTEL_ARC.md:46-56).
- Intel compute-runtime moved Gen8, Gen9 and Gen11 iGPUs (e.g. UHD 630 on Coffee/Comet Lake) to "legacy1" packages frozen at 24.35. Mainline dropped them from 24.39 (October 2024) ([Fedora change page](https://www.fedoraproject.org/wiki/Changes/IntelCompute2025); [Debian LEGACY_PLATFORMS.md](https://sources.debian.org/src/intel-compute-runtime/25.44.36015.8-1/LEGACY_PLATFORMS.md)).
- No NPU code exists in the repo (no grep hits searched specifically, but INTEL*.md never mentions an NPU; the matching lines above show none).

### Inferences (pancake-specific, quantitative where possible)

**1. DRAM bandwidth sets the CPU expert time.** Assume the per-expert CPU time scales inversely with effective bandwidth, at about 80% efficiency as on the reference box. The reference is 42 us at about 27 GB/s for 1.4 MB.

| Platform and memory | Effective GB/s | CPU time per 1.4 MB expert | Notes |
|---|---|---|---|
| LGA1151 / LGA1200-10th, 2ch DDR4-2666/2933 | about 27-30 | 38-42 us | same as reference |
| LGA1700 on DDR4 | about 40 (if ~50 measured) | about 28 us | |
| LGA1700 on DDR5-5600/6000 | about 65 (~82 measured x 0.8) | about 17-18 us | |
| Arrow Lake DDR5-6000 to 7800 | about 72-94 (90-118 measured) | about 12-16 us | if the kernel stays bandwidth-bound |
| W-3400, 8ch DDR5-4800 | up to ~245 at 80% of 307 | about 5 us | only if the pool has enough cores and NUMA/sub-NUMA is handled (STRATA_NUMA_SPLIT exists, pool.cpp:769) |
| E5 v4, 4ch DDR4-2400 | about 55-60 | about 20 us | AVX2 only, older cores, so it may become compute-bound |

At these speeds the kernel may stop being bandwidth-bound. Eight cores without SMT on Arrow Lake running AVX2+VNNI could become the limit. This is untested and needs `kq_cpu_bench`.

**2. PCIe generation sets promote time.** 1.4 MB / 115 us is about 12.2 GB/s effective on Gen3 x16.
- Gen4 x16 (the 7900 XT in any Rocket Lake, LGA1700 or LGA1851 slot): about 58 us.
- Gen5 x16 with a Gen5 GPU: about 29 us.
- Gen3 x8, for example a second card on LGA1700 2x8 Gen3 or old boards: about 230 us.

Break-even on the reference: the CPU (42 us) already beats a promote (115 us). The promote only pays off when an expert is reused often enough on the GPU to amortise it.

On a DDR5 Intel desktop with the 7900 XT, CPU compute (12-18 us) to promote (58 us) is about 1:3.5-4.5, worse for promoting than the reference's 1:2.7. Pancake should raise the reuse threshold for promotion and lean harder on CPU compute.

With a Gen5 GPU on Arrow Lake: 29 us against 12-16 us is about 1:2, close to the reference ratio. The current heuristics would transfer roughly as-is.

**3. Hybrid cores.**
- Default `All` puts one worker per physical core including E-cores. The pool's barrier-style phases then wait on the slowest worker, and E-cores are slower per thread. The repo's own measurement on 8P+12E is that 12-16 workers beat 19.
- Recommended settings:
  - `--pool-affinity auto` on Raptor Lake.
  - On Arrow Lake, either auto or the planner default. Skymont E-cores are much closer to P-cores than Gracemont, and there is no SMT to fall back on, so excluding E-cores leaves only 7 workers. This is worth benchmarking both ways.
- Thread Director only hints the OS scheduler. Pancake pins explicitly via affinity, so Thread Director matters mainly for unpinned host or I/O threads.
- Arrow Lake E-cores see higher DRAM latency when no P-core is active (Chips and Cheese). Pancake's host thread sits on a P-core, so this should not bite during decode.
- Gathered IQ decode runs only on P-core threads (`cpu_gather_fast_here`). E-core workers use the scalar-grid decode. Per-thread speed is asymmetric, so static equal task splits across P and E cores will idle the P-cores at barriers. Dynamic task stealing or chunked tasks would help (design note, not verified in code beyond the phase/task structure).

**4. AVX-512, AMX and VNNI.**
- Rocket Lake is the only consumer Intel desktop where pancake's AVX-512 expert path runs. Its AVX-512 gathers are Downfall-affected, which is probably why `cpu_gather_fast()` excludes it.
- Sapphire Rapids W-3400/2400 runs the AVX-512 path with gathers.
- An AMX tile kernel for experts would be new work. It matters mostly for prefill (batched tokens, compute-bound), not single-token decode, which stays bandwidth-bound. ggml-cpu's AMX path covers only what goes through ggml.
- Skylake-X / Cascade Lake HEDT are popular used buys with quad-channel DDR4. Pancake treats them as AVX2-only because of the VBMI requirement. A VBMI-free AVX-512 variant would be needed to use their AVX-512 units.
- Q2_0 legacy folders fail on every Intel desktop except Rocket Lake, because they need AVX-512.

**5. NVMe tier.**
- Chipset NVMe on LGA1151 and LGA1200-10th is capped near 3.9 GB/s by DMI 3.0 x4, shared with everything else.
- LGA1700 and LGA1851 chipset NVMe goes through DMI 4.0 x8 (about 15.8 GB/s), adequate for one Gen4 drive.
- A CPU-attached Gen5 x4 on Arrow Lake can feed about 12-14 GB/s drives, approaching the effective rate of a Gen3 x16 GPU link. NVMe-to-RAM promotion stops being the bottleneck there.

**6. ReBAR.** Practically universal from 10th gen with a BIOS update. Pancake's host-to-GPU staging does not obviously require ReBAR (no repo hits for ReBAR/Resizable in docs). It matters for Arc cards (Intel requires it) and for any direct CPU writes into VRAM.

**7. iGPU and NPU.**
- UHD 630 (Gen9.5) is on frozen legacy compute-runtime and not worth targeting.
- Xe-LP iGPUs (11th-14th gen UHD 730/770) and Arrow Lake's 4 Xe cores have no DPAS int8 (Xe2 only) and no shared-memory planning in pancake. They are not usable today.
- Lunar Lake's Xe2 iGPU is the only integrated part that meets the int8 DPAS requirement, but it is mobile-only with 16-32 GB unified memory and untested.
- The NPU (13 TOPS on Arrow Lake) has no pancake path, and an OpenVINO-only target is unlikely to fit expert streaming.

**8. Max RAM.** 128 GB on LGA1151 and LGA1200. 192 GB on LGA1700 (four 48 GB DIMMs), typically at reduced speed with 2 DIMMs per channel. That lets the full expert set of a roughly 100-150 GB quant sit in RAM and removes the NVMe tier. HEDT scales to TB-class.

### Gaps
- No measured pancake numbers exist on any Intel platform. All per-expert timings above are linear extrapolations from the reference box and assume the kernel stays bandwidth-bound.
- No authoritative per-platform source found for VT-d/IOMMU default state, or for its effect on GPU DMA throughput on Linux (iommu=pt vs translated).
- No measured AMX gain for MoE expert GEMV/GEMM; the only figure is a vendor-adjacent llama.cpp blog.
- Arrow Lake chipset lane count conflicts across reviews (4, 20, or 48 platform total). An Intel Z890 platform brief was not retrieved.
- Whether the Arrow Lake Gen5 x4 (lanes 21-24) latency penalty affects NVMe streaming throughput is unknown.
