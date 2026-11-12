#pragma once
// MOD_PLAN A53: one start-up line when a GPU's PCIe DMA is address-translated (its IOMMU group type is DMA or DMA-FQ).
// Identity (passthrough) is silent, and so is a machine with no GPU.  Sysfs reads only; nothing changes for the run.
// The same check, run by hand, is tools/pancake/os_tune.sh check (iommu_check).
#include <cstdio>
#include <string>

#if defined(__linux__)
#include <filesystem>
#include <fstream>
#endif

namespace strata::platform {

/// The note for the first translated GPU under `sys` (the sysfs root), or "" when there is none to report.
/// `sys` is a parameter so a test can use a fake tree.
inline std::string iommu_gpu_note(const std::string& sys = "/sys") {
#if defined(__linux__)
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path devs = fs::path(sys) / "bus" / "pci" / "devices";
    for (const auto& d : fs::directory_iterator(devs, ec)) {
        auto read = [&](const char* name) {
            std::ifstream f(d.path() / name);
            std::string s;
            std::getline(f, s);
            return s;
        };
        if (read("class").rfind("0x03", 0) != 0) continue;             // display controllers only
        const std::string dev = read("device");
        if (dev == "0x1636" || dev == "0x1638" || dev == "0x15e7") continue;   // the APU's own graphics (not a discrete card)
        std::ifstream tf(d.path() / "iommu_group" / "type");
        std::string t;
        std::getline(tf, t);
        if (t.empty()) continue;                                        // no IOMMU group for this device
        if (t == "DMA" || t == "DMA-FQ")
            return "strata: GPU " + d.path().filename().string() + " is behind a translated IOMMU group (" + t +
                   "): PCIe DMA is address-translated and can cost throughput; iommu=pt on the kernel command line avoids it";
        return "";                                                      // identity (passthrough): nothing to say
    }
#else
    (void) sys;
#endif
    return "";
}

}  // namespace strata::platform
