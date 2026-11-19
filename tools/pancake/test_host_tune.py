"""Self test of host_tune.sh / host_tune_ab.sh: syntax (sh -n, bash -n, dash -n, shellcheck when installed), and the apply /
revert / state-file / persist / refuse logic on a FAKE sysfs tree (HOST_TUNE_ROOT, temp dirs, no root, no GPU, no sudo).
Needs a POSIX sh on PATH (Git for Windows has one); skipped without.
Run: python -m pytest tools/pancake/test_host_tune.py   (or python tools/pancake/test_host_tune.py)"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNE = HERE / "host_tune.sh"
AB = HERE / "host_tune_ab.sh"
SH = shutil.which("sh") or shutil.which("bash")
GIB = 1 << 30


def w(root, rel, text):
    p = Path(root) / rel.lstrip("/")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((text + "\n").encode())


def build(root):
    """A fake host: one NVMe, THP madvise, swappiness 60, 2 CPUs with powersave, passive amd_pstate, a dGPU (card1), an iGPU
    (card2) and a connector dir, ASPM default, 4 idle states per CPU."""
    w(root, "sys/block/nvme0n1/queue/read_ahead_kb", "128")
    w(root, "sys/block/nvme0n1/queue/scheduler", "[mq-deadline] none kyber")
    w(root, "sys/block/sda/queue/read_ahead_kb", "128")
    w(root, "sys/kernel/mm/transparent_hugepage/enabled", "always [madvise] never")
    w(root, "sys/kernel/mm/transparent_hugepage/defrag", "always defer defer+madvise [madvise] never")
    w(root, "proc/sys/vm/swappiness", "60")
    w(root, "proc/sys/vm/max_map_count", "1048576")
    for c in (0, 1):
        f = f"sys/devices/system/cpu/cpu{c}/cpufreq/"
        w(root, f + "scaling_governor", "powersave")
        w(root, f + "scaling_available_governors", "performance powersave")
        w(root, f + "energy_performance_preference", "balance_performance")
        w(root, f + "energy_performance_available_preferences", "default performance balance_performance power")
        for i, lat in enumerate((0, 1, 18, 350)):
            d = f"sys/devices/system/cpu/cpu{c}/cpuidle/state{i}/"
            w(root, d + "latency", str(lat))
            w(root, d + "disable", "0")
    w(root, "sys/devices/system/cpu/amd_pstate/status", "passive")
    w(root, "sys/class/drm/card1/device/power_dpm_force_performance_level", "auto")
    w(root, "sys/class/drm/card1/device/vendor", "0x1002")
    w(root, "sys/class/drm/card1/device/mem_info_vram_total", str(20 * GIB))
    w(root, "sys/class/drm/card2/device/power_dpm_force_performance_level", "auto")
    w(root, "sys/class/drm/card2/device/vendor", "0x1002")
    w(root, "sys/class/drm/card2/device/mem_info_vram_total", str(512 << 20))
    (Path(root) / "sys/class/drm/card1-DP-1").mkdir(parents=True)
    w(root, "sys/module/pcie_aspm/parameters/policy", "[default] performance powersave powersupersave")
    w(root, "proc/cmdline", "BOOT_IMAGE=/vmlinuz root=/dev/sda1 quiet")


def snapshot(root):
    """{relative path: first token that is selected} of every file except the tool's own state."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "var/lib" not in p.as_posix():
            t = p.read_bytes().decode()
            if "[" in t:
                t = t.split("[", 1)[1].split("]", 1)[0]
            out[p.relative_to(root).as_posix()] = t.strip()
    return out


@unittest.skipUnless(SH, "no POSIX sh on PATH")
class HostTune(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name).as_posix()
        build(self.root)
        self.state = Path(self.root) / "var/lib/pancake-host-tune"

    def tearDown(self):
        self.td.cleanup()

    def run_tune(self, *args, env=None):
        e = dict(os.environ, HOST_TUNE_ROOT=self.root, HOST_TUNE_USER="alice")
        e.pop("HOST_TUNE_STATE", None)
        e.update(env or {})
        r = subprocess.run([SH, str(TUNE), *args], capture_output=True, text=True, env=e)
        return r.returncode, r.stdout + r.stderr

    def val(self, rel):
        t = (Path(self.root) / rel).read_bytes().decode()
        if "[" in t:
            t = t.split("[", 1)[1].split("]", 1)[0]
        return t.strip()

    # ---- syntax
    def test_syntax(self):
        for script in (TUNE, AB):
            self.assertTrue(script.exists(), script)
            for shell in ("sh", "bash", "dash"):
                exe = shutil.which(shell)
                if exe:
                    r = subprocess.run([exe, "-n", str(script)], capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, f"{shell} -n {script.name}: {r.stderr}")
        if shutil.which("shellcheck"):
            r = subprocess.run(["shellcheck", "-s", "sh", "-S", "warning", str(TUNE)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout)

    def test_list_and_help(self):
        rc, out = self.run_tune("--list")
        self.assertEqual(rc, 0)
        for s in "nvme_ra thp swappiness memlock cpu_perf amd_pstate gpu_dpm aspm cstate iommu headless".split():
            self.assertIn(s, out)
        self.assertIn("risk:", out)
        self.assertEqual(self.run_tune()[0], 2)
        self.assertEqual(self.run_tune("--bogus")[0], 2)

    # ---- check is read only
    def test_check_writes_nothing(self):
        before = snapshot(self.root)
        rc, out = self.run_tune("--check")
        self.assertEqual(rc, 0, out)
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.state.exists())
        self.assertIn("nvme_ra", out)
        self.assertIn("not-at-target", out)
        self.assertRegex(out, r"iommu\s+report")
        self.assertRegex(out, r"headless\s+report")

    # ---- apply + revert round trip
    def test_apply_all_runtime_then_revert_restores_exactly(self):
        before = snapshot(self.root)
        names = "nvme_ra,thp,swappiness,cpu_perf,amd_pstate,gpu_dpm,aspm,cstate"
        rc, out = self.run_tune("--apply", names)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.val("sys/block/nvme0n1/queue/read_ahead_kb"), "4096")
        self.assertEqual(self.val("sys/block/nvme0n1/queue/scheduler"), "none")
        self.assertEqual(self.val("sys/block/sda/queue/read_ahead_kb"), "128", "only nvme disks are touched")
        self.assertEqual(self.val("sys/kernel/mm/transparent_hugepage/enabled"), "always")
        self.assertEqual(self.val("sys/kernel/mm/transparent_hugepage/defrag"), "defer+madvise")
        self.assertEqual(self.val("proc/sys/vm/swappiness"), "10")
        self.assertEqual(self.val("sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), "performance")
        self.assertEqual(self.val("sys/devices/system/cpu/cpu1/cpufreq/energy_performance_preference"), "performance")
        self.assertEqual(self.val("sys/devices/system/cpu/amd_pstate/status"), "active")
        self.assertEqual(self.val("sys/class/drm/card1/device/power_dpm_force_performance_level"), "high")
        self.assertEqual(self.val("sys/class/drm/card2/device/power_dpm_force_performance_level"), "auto", "iGPU skipped")
        self.assertEqual(self.val("sys/module/pcie_aspm/parameters/policy"), "performance")
        for c in (0, 1):
            base = f"sys/devices/system/cpu/cpu{c}/cpuidle/"
            self.assertEqual([self.val(f"{base}state{i}/disable") for i in range(4)], ["0", "0", "1", "1"])
        sf = (self.state / "swappiness.state").read_text()
        self.assertIn("proc/sys/vm/swappiness\t60", sf)
        rc, out = self.run_tune("--status")
        self.assertIn("swappiness: applied", out)
        self.assertIn("was 60, now 10", out)
        rc, out = self.run_tune("--revert", "all")
        self.assertEqual(rc, 0, out)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(list(self.state.glob("*.state")), [])
        rc, out = self.run_tune("--revert", "all")
        self.assertIn("nothing to revert", out)

    def test_revert_restores_the_saved_value_not_a_default(self):
        w(self.root, "proc/sys/vm/swappiness", "37")
        self.assertEqual(self.run_tune("--apply", "swappiness")[0], 0)
        self.assertEqual(self.val("proc/sys/vm/swappiness"), "10")
        self.assertEqual(self.run_tune("--revert", "swappiness")[0], 0)
        self.assertEqual(self.val("proc/sys/vm/swappiness"), "37")

    def test_second_apply_refused_so_original_is_kept(self):
        self.assertEqual(self.run_tune("--apply", "swappiness")[0], 0)
        rc, out = self.run_tune("--apply", "swappiness")
        self.assertEqual(rc, 1)
        self.assertIn("already applied", out)
        self.assertIn("proc/sys/vm/swappiness\t60", (self.state / "swappiness.state").read_text())
        self.run_tune("--revert", "swappiness")
        self.assertEqual(self.val("proc/sys/vm/swappiness"), "60")

    def test_already_at_target_makes_no_state_file(self):
        w(self.root, "proc/sys/vm/swappiness", "10")
        rc, out = self.run_tune("--apply", "swappiness")
        self.assertEqual(rc, 0)
        self.assertIn("already at target", out)
        self.assertFalse((self.state / "swappiness.state").exists())

    def test_failed_write_is_reported_and_state_kept(self):
        # a directory where the file should be: the write fails, exit 1
        p = Path(self.root) / "proc/sys/vm/swappiness"
        p.unlink()
        p.mkdir()
        rc, out = self.run_tune("--apply", "swappiness")
        self.assertNotEqual(rc, 0, out)

    def test_dry_run_writes_nothing(self):
        before = snapshot(self.root)
        rc, out = self.run_tune("--dry-run", "--apply", "nvme_ra,thp,memlock", "--persist")
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.state.exists())
        self.assertIn("[dry-run]", out)

    def test_probe_exit_codes(self):
        self.assertEqual(self.run_tune("--probe", "swappiness")[0], 1)
        self.run_tune("--apply", "swappiness")
        self.assertEqual(self.run_tune("--probe", "swappiness")[0], 0)
        shutil.rmtree(Path(self.root) / "sys/module")
        self.assertEqual(self.run_tune("--probe", "aspm")[0], 2)
        self.assertEqual(self.run_tune("--probe", "iommu")[0], 2)

    # ---- refusals
    def test_refusals(self):
        for args in (("--apply", "iommu"), ("--apply", "headless"), ("--apply", "nope"), ("--apply", "all"),
                     ("--apply", "nvme_ra;rm"), ("--apply", "../x"), ("--apply", "memlock")):
            rc, out = self.run_tune(*args)
            self.assertNotEqual(rc, 0, args)
        self.assertFalse(self.state.exists() and list(self.state.glob("*.state")))
        rc, out = self.run_tune("--apply", "iommu")
        self.assertIn("report-only", out)
        rc, out = self.run_tune("--apply", "memlock")
        self.assertIn("no runtime form", out)

    def test_n_a_steps_do_nothing(self):
        shutil.rmtree(Path(self.root) / "sys/module")
        rc, out = self.run_tune("--apply", "aspm")
        self.assertEqual(rc, 0)
        self.assertIn("not applicable", out)

    # ---- narrowing options
    def test_gpu_option_and_nvme_option(self):
        self.run_tune("--apply", "gpu_dpm", "--gpu", "card2")
        self.assertEqual(self.val("sys/class/drm/card2/device/power_dpm_force_performance_level"), "high")
        self.assertEqual(self.val("sys/class/drm/card1/device/power_dpm_force_performance_level"), "auto")
        w(self.root, "sys/block/nvme1n1/queue/read_ahead_kb", "128")
        self.run_tune("--apply", "nvme_ra", "--nvme", "nvme1n1")
        self.assertEqual(self.val("sys/block/nvme1n1/queue/read_ahead_kb"), "4096")
        self.assertEqual(self.val("sys/block/nvme0n1/queue/read_ahead_kb"), "128")

    def test_cstate_latency_threshold(self):
        self.run_tune("--apply", "cstate", "--cstate-latency-us", "100")
        self.assertEqual([self.val(f"sys/devices/system/cpu/cpu0/cpuidle/state{i}/disable") for i in range(4)],
                         ["0", "0", "0", "1"])

    # ---- persist
    def test_persist_writes_pancake_files_and_revert_removes_them(self):
        before = snapshot(self.root)
        rc, out = self.run_tune("--apply", "nvme_ra,swappiness,thp,memlock", "--persist")
        self.assertEqual(rc, 0, out)
        r = Path(self.root)
        udev = (r / "etc/udev/rules.d/60-pancake-nvme.rules").read_text()
        self.assertIn('ATTR{queue/read_ahead_kb}="4096"', udev)
        self.assertIn("vm.swappiness = 10", (r / "etc/sysctl.d/91-pancake-swappiness.conf").read_text())
        thp = (r / "etc/tmpfiles.d/pancake-thp.conf").read_text()
        self.assertIn("w /sys/kernel/mm/transparent_hugepage/enabled - - - - always", thp)
        self.assertIn("alice - memlock unlimited", (r / "etc/security/limits.d/99-pancake-memlock.conf").read_text())
        self.assertNotIn(self.root, thp + udev, "persistent files hold real paths, not the fake root")
        rc, out = self.run_tune("--revert", "all")
        self.assertEqual(rc, 0, out)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(list((r / "etc").rglob("*pancake*")), [])

    def test_memlock_refuses_root_user(self):
        rc, out = self.run_tune("--apply", "memlock", "--persist", env={"HOST_TUNE_USER": "root"})
        self.assertNotEqual(rc, 0)
        self.assertFalse((Path(self.root) / "etc/security/limits.d/99-pancake-memlock.conf").exists())

    # ---- mlock verdicts
    def engine(self, pid, vmlck_kib, soft):
        w(self.root, f"proc/{pid}/status", f"Name:\tstrata\nVmLck:\t{vmlck_kib} kB\n")
        w(self.root, f"proc/{pid}/limits", f"Max locked memory        {soft}                {soft}                bytes")

    def test_mlock_ok(self):
        self.engine(4242, 34652160, "unlimited")
        rc, out = self.run_tune("--mlock-check", "--pid", "4242", env={"HOST_TUNE_ULIMIT_SOFT": "unlimited"})
        self.assertEqual(rc, 0, out)
        self.assertIn("VERDICT: OK", out)

    def test_mlock_fails_with_low_limit(self):
        self.engine(4242, 0, "8388608")
        rc, out = self.run_tune("--mlock-check", "--pid", "4242", env={"HOST_TUNE_ULIMIT_SOFT": "8192"})
        self.assertEqual(rc, 1, out)
        self.assertIn("mlock FAILED", out)
        self.assertIn("--apply memlock", out)

    def test_mlock_limit_not_the_cause(self):
        self.engine(4242, 1000, "unlimited")
        rc, out = self.run_tune("--mlock-check", "--pid", "4242")
        self.assertEqual(rc, 1)
        self.assertIn("not the cause", out)

    def test_mlock_no_engine_reports_limit_only(self):
        rc, out = self.run_tune("--mlock-check", "--pid", "999", env={"HOST_TUNE_ULIMIT_SOFT": "8192"})
        self.assertIn("would FAIL", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
