// h2d_bw.hip.cpp - LEADS.md Part T T15: pinned host-to-device and device-to-host bandwidth, to reconcile "PCIe 3.0, 13.4 GB/s" (MOD_PLAN.md) with the
// 16 GT/s x16 seen in sysfs, and to compare before / after the Gen4 upgrade.  Not part of the engine build (no CMake target): compile by hand.
//
// RUN ONLY WHEN THE BOX IS IDLE (a compile uses the clang busy pattern, a run uses the GPU's link; do neither during a screen):
//   cd /home/dan/clone-of-d/src
//   /opt/rocm/core-10.0/bin/hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/h2d_bw.hip.cpp -o /home/dan/auto-runs/snap/h2d_bw
//   /home/dan/auto-runs/snap/h2d_bw 256 20 > /home/dan/auto-runs/snap/h2d-$(date +%F-%H%M).json       (MiB per copy, repetitions)
//   python3 tools/pancake/gpu_snapshot.py --phase idle --out /home/dan/auto-runs/snap --h2d-json /home/dan/auto-runs/snap/h2d-<stamp>.json
// Last stdout line is one JSON object: {"h2d_gbs":..,"d2h_gbs":..,"mib":..,"reps":..}.  Allocates MiB of pinned host memory and MiB of device memory only.
// Expected: Gen3 x16 about 12-13.5 GB/s, Gen4 x16 about 24-28 GB/s.  Coded on a PC without ROCm: NOT compiled, NOT run.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 256;
  int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  size_t bytes = mib << 20;
  void *h = nullptr, *d = nullptr;
  CK(hipHostMalloc(&h, bytes, hipHostMallocDefault));
  CK(hipMalloc(&d, bytes));
  for (size_t i = 0; i < bytes; i += 4096) static_cast<char*>(h)[i] = 1;   // touch every page
  hipStream_t s;
  CK(hipStreamCreate(&s));
  CK(hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s));               // warm-up
  CK(hipStreamSynchronize(s));
  double gb[2] = {0, 0};
  for (int dir = 0; dir < 2; ++dir) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
      CK(dir == 0 ? hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s) : hipMemcpyAsync(h, d, bytes, hipMemcpyDeviceToHost, s));
    CK(hipStreamSynchronize(s));
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    gb[dir] = (double)bytes * reps / sec / 1e9;
  }
  std::printf("{\"h2d_gbs\":%.2f,\"d2h_gbs\":%.2f,\"mib\":%zu,\"reps\":%d}\n", gb[0], gb[1], mib, reps);
  hipFree(d);
  hipHostFree(h);
  return 0;
}
