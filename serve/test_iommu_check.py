"""A53: the IOMMU start-up note.  Fake sysfs trees in a temp dir; no GPU, no root.

    python -m unittest serve.test_iommu_check
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from serve import iommu_check


def make_sys(root: Path, units=("ivhd0",), gpus=(), cmdline=""):
    """gpus: (bdf, class, device id, group type or None)."""
    (root / "sys/class/iommu").mkdir(parents=True)
    for u in units:
        (root / "sys/class/iommu" / u).mkdir()
    pci = root / "sys/bus/pci/devices"
    pci.mkdir(parents=True)
    for bdf, cls, dev, gtype in gpus:
        d = pci / bdf
        d.mkdir()
        (d / "class").write_text(cls + "\n")
        (d / "device").write_text(dev + "\n")
        if gtype is not None:
            (d / "iommu_group").mkdir()
            (d / "iommu_group/type").write_text(gtype + "\n")
    (root / "cmdline").write_text(cmdline + "\n")
    return str(root / "sys"), str(root / "cmdline")


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux only")
class IommuCheck(unittest.TestCase):
    def run_case(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            sysfs, cmd = make_sys(Path(d), **kw)
            return iommu_check.state(sysfs, cmd), iommu_check.note({}, sysfs, cmd)

    def test_translated_dma_fq_warns(self):
        st, note = self.run_case(gpus=[("0000:12:00.0", "0x030000", "0x744c", "DMA-FQ")])
        self.assertEqual(st["mode"], "translated")
        self.assertIn("0000:12:00.0", note)
        self.assertIn("iommu=pt", note)

    def test_translated_strict_dma_warns(self):
        st, note = self.run_case(gpus=[("0000:12:00.0", "0x030000", "0x744c", "DMA")])
        self.assertEqual(st["mode"], "translated")
        self.assertIsNotNone(note)

    def test_identity_is_passthrough_no_note(self):
        st, note = self.run_case(gpus=[("0000:12:00.0", "0x030000", "0x744c", "identity")])
        self.assertEqual(st["mode"], "passthrough")
        self.assertIsNone(note)

    def test_apu_graphics_and_non_display_devices_are_ignored(self):
        st, note = self.run_case(gpus=[("0000:30:00.0", "0x030000", "0x1638", "DMA-FQ"),     # the APU's own graphics
                                       ("0000:01:00.0", "0x010802", "0xa80a", "DMA-FQ")])    # an NVMe drive
        self.assertEqual(st["devices"], [])
        self.assertIsNone(note)

    def test_no_iommu_unit_is_off(self):
        st, note = self.run_case(units=(), gpus=[("0000:12:00.0", "0x030000", "0x744c", "DMA-FQ")])
        self.assertEqual(st["mode"], "off")
        self.assertIsNone(note)

    def test_vfio_group_is_unknown(self):
        st, note = self.run_case(gpus=[("0000:12:00.0", "0x030000", "0x744c", "unmanaged")])
        self.assertEqual(st["mode"], "unknown")
        self.assertIsNone(note)

    def test_cmdline_fallback_without_device_info(self):
        st, _ = self.run_case(cmdline="ro quiet iommu=pt")
        self.assertEqual(st["mode"], "passthrough")
        st, _ = self.run_case(cmdline="ro amd_iommu=off")
        self.assertEqual(st["mode"], "off")
        st, _ = self.run_case(cmdline="ro quiet")
        self.assertEqual(st["mode"], "unknown")

    def test_one_translated_gpu_among_two_warns(self):
        st, note = self.run_case(gpus=[("0000:12:00.0", "0x030000", "0x744c", "identity"),
                                       ("0000:13:00.0", "0x030200", "0x2504", "DMA-FQ")])
        self.assertEqual(st["mode"], "translated")
        self.assertIn("0000:13:00.0", note)
        self.assertNotIn("0000:12:00.0", note)

    def test_switch_off(self):
        with tempfile.TemporaryDirectory() as d:
            sysfs, cmd = make_sys(Path(d), gpus=[("0000:12:00.0", "0x030000", "0x744c", "DMA-FQ")])
            self.assertIsNone(iommu_check.note({"STRATA_IOMMU_CHECK": "0"}, sysfs, cmd))
            self.assertIsNotNone(iommu_check.note({"STRATA_IOMMU_CHECK": "1"}, sysfs, cmd))

    def test_missing_sysfs_never_raises(self):
        st = iommu_check.state("/no/such/dir", "/no/such/cmdline")
        self.assertIn(st["mode"], ("unknown", "off"))


if __name__ == "__main__":
    unittest.main()
