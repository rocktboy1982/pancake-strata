// mem_pressure_test (MOD_PLAN 86): PSI parsing, the cgroup path, the recent figure, the switches, and the probe on fixture files.
// CPU only; reads no real /proc file except in the last, telemetry-agnostic check.
#include "strata/platform/mem_pressure.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <atomic>
#include <cstdlib>
#include <thread>

using namespace strata::platform;

static int checks = 0, bad = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::printf("FAIL %s\n", label); ++bad; }
}
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static void write_file(const std::filesystem::path& p, const std::string& text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

int main() {
    // 1. parsing
    const std::string sample =
        "some avg10=12.34 avg60=5.00 avg300=1.25 total=987654321\n"
        "full avg10=3.50 avg60=1.00 avg300=0.10 total=12345\n";
    auto psi = parse_psi(sample);
    check(psi.has_value(), "parse: a well-formed file");
    check(psi && psi->some.present && near(psi->some.avg10, 12.34) && near(psi->some.avg60, 5.0) &&
          near(psi->some.avg300, 1.25) && psi->some.total_us == 987654321ull, "parse: some figures");
    check(psi && psi->full.present && near(psi->full.avg10, 3.5) && psi->full.total_us == 12345ull, "parse: full figures");
    psi = parse_psi("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");   // an older kernel's cpu file has no full line
    check(psi && psi->some.present && !psi->full.present && psi->some.total_us == 0, "parse: some only, idle");
    check(!parse_psi(""), "parse: empty");
    check(!parse_psi("full avg10=1.00 avg60=0 avg300=0 total=5\n"), "parse: no some line");
    check(!parse_psi("some avg10=abc avg60=0 avg300=0 total=5\n"), "parse: avg10 not a number");
    check(!parse_psi("some avg10=1.0 avg60=0 avg300=0\n"), "parse: no total");
    psi = parse_psi("garbage\nsome avg10=99.99 avg60=50 avg300=10 total=18446744073709551615\r\n");
    check(psi && near(psi->some.avg10, 99.99) && psi->some.total_us == 18446744073709551615ull, "parse: CRLF, junk line, max total");

    // 2. the cgroup path
    check(cgroup_memory_pressure_path("0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-x.scope\n", "/sys/fs/cgroup") ==
              "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/run-x.scope/memory.pressure",
          "cgroup: v2 path");
    check(cgroup_memory_pressure_path("0::/\n", "/r") == "/r/memory.pressure", "cgroup: the root group");
    check(cgroup_memory_pressure_path("12:memory:/a\n11:cpu:/a\n0::/b\n", "/r") == "/r/b/memory.pressure", "cgroup: hybrid v1+v2");
    check(cgroup_memory_pressure_path("12:memory:/a\n", "/r").empty(), "cgroup: v1 only: none");
    check(cgroup_memory_pressure_path("", "/r").empty(), "cgroup: empty: none");

    // 3. the recent figure and the trip rule
    check(near(psi_recent_pct(1000000, 1100000, 0.5), 20.0), "recent: 100 ms stalled in 500 ms = 20%");
    check(near(psi_recent_pct(0, 0, 0.2), 0.0), "recent: idle");
    check(near(psi_recent_pct(0, 5000000, 1.0), 100.0), "recent: clamped to 100");
    check(psi_recent_pct(10, 5, 1.0) < 0.0, "recent: counter went back: unknown");
    check(psi_recent_pct(0, 5, 0.0) < 0.0, "recent: no time: unknown");
    check(pressure_at_or_above(30.0, 5.0, 20.0) == false, "trip: the recent figure wins over a high avg10");
    check(pressure_at_or_above(5.0, 25.0, 20.0), "trip: recent above");
    check(pressure_at_or_above(25.0, -1.0, 20.0), "trip: no recent: avg10");
    check(!pressure_at_or_above(19.9, -1.0, 20.0), "trip: avg10 below");

    // 4. the switches
    PressureConfig c = pressure_config(nullptr, nullptr, nullptr, nullptr, nullptr);
    check(c.on && near(c.pause_pct, 20) && near(c.resume_pct, 10) && c.max_pause_ms == 30000 && near(c.park_pct, 20), "config: defaults");
    c = pressure_config("0", nullptr, nullptr, nullptr, nullptr);
    check(!c.on, "config: STRATA_PSI_GUARD=0");
    c = pressure_config("1", "40", "15", "5000", "30");
    check(c.on && near(c.pause_pct, 40) && near(c.resume_pct, 15) && c.max_pause_ms == 5000 && near(c.park_pct, 30), "config: set");
    c = pressure_config(nullptr, "30", "50", nullptr, nullptr);
    check(near(c.resume_pct, 15) && near(c.park_pct, 30), "config: resume kept below pause; park follows pause");
    c = pressure_config(nullptr, "abc", "-1", "x", "200");
    check(near(c.pause_pct, 20) && near(c.resume_pct, 10) && c.max_pause_ms == 30000 && near(c.park_pct, 20), "config: junk keeps defaults");

    // 5. the probe on fixture files
    const auto dir = std::filesystem::temp_directory_path() / ("strata_psi_test_" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(dir);
    write_file(dir / "pressure_memory", "some avg10=2.00 avg60=1 avg300=0 total=1000\nfull avg10=0 avg60=0 avg300=0 total=0\n");
    write_file(dir / "self_cgroup", "0::/scope.x\n");
    write_file(dir / "cg" / "scope.x" / "memory.pressure", "some avg10=35.50 avg60=10 avg300=2 total=5000\n");
    MemoryPressureProbe::Paths paths;
    paths.system = (dir / "pressure_memory").string();
    paths.self_cgroup = (dir / "self_cgroup").string();
    paths.cgroup_root = (dir / "cg").string();
    MemoryPressureProbe probe(paths);
#if defined(__linux__)
    check(probe.usable(), "probe: both fixture sources found");
    PressureReading r = probe.read();
    check(r.known && near(r.avg10, 35.5) && r.recent < 0.0, "probe: first read: max avg10, no recent");
    check(r.where == "system 2.0% / cgroup 35.5%", "probe: where");
    write_file(dir / "cg" / "scope.x" / "memory.pressure", "some avg10=36.00 avg60=10 avg300=2 total=1005000\n");
    r = probe.read();
    check(r.known && r.recent > 0.0, "probe: second read has a recent figure");
    // no cgroup file: the system source alone
    std::filesystem::remove(dir / "cg" / "scope.x" / "memory.pressure");
    MemoryPressureProbe sys_only(paths);
    r = sys_only.read();
    check(sys_only.usable() && r.known && near(r.avg10, 2.0), "probe: system file only");
    // nothing readable: unusable, reads unknown
    MemoryPressureProbe::Paths none;
    none.system = (dir / "missing").string();
    none.self_cgroup = (dir / "missing2").string();
    none.cgroup_root = (dir / "missing3").string();
    MemoryPressureProbe empty(none);
    check(!empty.usable() && !empty.read().known, "probe: no files: unusable");
    // 6. the loader guard on fixture files: a load that starts under pressure holds the readers until the stall stops
    {
        const auto mem = dir / "pressure_memory";
        MemoryPressureProbe::Paths sys;
        sys.system = mem.string();
        sys.self_cgroup = (dir / "missing").string();
        sys.cgroup_root = (dir / "missing").string();
        // a stall: `total` grows with the wall clock (100% stalled) while `stalling`, else it stands still
        std::atomic<bool> stalling{true}, stop{false};
        write_file(mem, "some avg10=90.00 avg60=10 avg300=2 total=1000000\n");
        std::thread stall([&] {
            uint64_t total = 1000000;
            auto last = std::chrono::steady_clock::now();
            while (!stop) {
                const auto now = std::chrono::steady_clock::now();
                if (stalling) total += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
                last = now;
                const std::string tmp = mem.string() + ".tmp";
                write_file(tmp, "some avg10=90.00 avg60=10 avg300=2 total=" + std::to_string(total) + "\n");
                std::filesystem::rename(tmp, mem);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        loader_pressure_begin_with("fixture", sys);          // avg10 90% >= 20%: held
        std::atomic<bool> done{false};
        const auto t0 = std::chrono::steady_clock::now();
        std::thread reader([&] { loader_pressure_check(); done = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        check(!done.load(), "guard: a reader is held under pressure");
        stalling = false;   // the stall stops (avg10 stays high: the recent figure decides) -> resumed at a later sample
        reader.join();
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check(done.load() && waited < 3.0, "guard: resumed once the stall stopped");
        loader_pressure_end();
        const LoaderPressureStats s = loader_pressure_last();
        check(s.pauses == 1 && s.paused_ms > 500.0 && !s.budget_spent, "guard: one pause, counted");

        // a stall that never stops: the pause budget (STRATA_PSI_MAX_PAUSE_MS) lets the readers go
        setenv("STRATA_PSI_MAX_PAUSE_MS", "600", 1);
        stalling = true;
        loader_pressure_begin_with("fixture", sys);
        const auto t1 = std::chrono::steady_clock::now();
        loader_pressure_check();   // held until the budget is spent
        const double held = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        loader_pressure_check();   // the budget is spent: not held again
        const double again = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() - held;
        loader_pressure_end();
        stop = true;
        stall.join();
        unsetenv("STRATA_PSI_MAX_PAUSE_MS");
        const LoaderPressureStats b = loader_pressure_last();
        check(held >= 0.5 && held < 3.0, "guard: held about the pause budget");
        check(again < 0.1, "guard: not held again once the budget is spent");
        check(b.pauses == 1 && b.budget_spent, "guard: budget spent recorded");

        // STRATA_PSI_GUARD=0: never held
        setenv("STRATA_PSI_GUARD", "0", 1);
        write_file(mem, "some avg10=99.00 avg60=10 avg300=2 total=1000000\n");
        loader_pressure_begin_with("fixture", sys);
        const auto t2 = std::chrono::steady_clock::now();
        loader_pressure_check();
        check(std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count() < 0.1, "guard off: not held");
        loader_pressure_end();
        check(loader_pressure_last().pauses == 0, "guard off: no pauses");
        unsetenv("STRATA_PSI_GUARD");
    }
#else
    check(!probe.usable(), "probe: no PSI off Linux");
#endif
    std::filesystem::remove_all(dir);

    // 7. the loader guard runs without PSI assumptions: begin/check/end nest and never block an idle box for long
    loader_pressure_begin("test");
    loader_pressure_begin("nested");
    loader_pressure_check();
    loader_pressure_end();
    loader_pressure_end();
    loader_pressure_end();   // an extra end is ignored
    std::string why;
    (void) memory_pressure_blocks_parking(&why);

    std::printf(bad == 0 ? "mem_pressure_test: %d checks passed\n" : "mem_pressure_test: %d checks, failures above\n", checks);
    return bad == 0 ? 0 : 1;
}
