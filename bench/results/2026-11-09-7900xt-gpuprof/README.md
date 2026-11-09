# 7900 XT GPU profile, fast-hip preset (2026-11-09)

rocprofv3 --kernel-trace --stats, engine run directly (gpuprof2.sh + prof-engine.sh), one `short` request
(34-token prompt, 256 new tokens). The profiler slows decode: 79 tok/s here vs ~126 tok/s without it,
so the gap figures below overstate what a normal run sees.

Decode phase (first to last combine_cpu): 3,638 ms wall, 2,682 ms of kernels.

| Kernel | Share of GPU time | Note |
| --- | ---: | --- |
| combine_cpu_k10_vec4 | 38.3% | spin-waits for the CPU expert rows: CPU time seen from the GPU |
| rdna_kq (Q6/Q8 matvec, all shapes) | ~10% | |
| expdec_gu + native_down (GPU experts) | ~13% | |
| rdna_gr_fused | 5.6% | |
| bf16_f32_mmvf_multi | 5.1% | |
| wait_flag_copy_i32 | 4.1% | waits as well |
| route_publish_res | 2.7% | |
| fillBufferUnAligned | 1.8% | all 256 calls at load/setup, none during decode: no decode gain |

- 395 kernels per token on the main stream; gaps under 50 us average 5.4 us (540 ms in total, inflated by the profiler).
- Gaps over 1 ms total 525 ms: host-side stalls at window and request edges (kv_append -> copyBuffer, copyBuffer -> copy_from_mapped).
- Conclusion: the CPU experts still gate decode (about 42% of GPU time is spent waiting). The cache hit rate and expert
  deferral are worth more than GPU kernel work. On the GPU side the levers are fewer kernels per token (fusion) and the
  host stalls at window edges.
