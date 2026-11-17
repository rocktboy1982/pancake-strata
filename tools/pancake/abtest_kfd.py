#!/usr/bin/env python3
"""abtest_kfd.py - abtest.py with the amdkfd eviction time of every engine start recorded as `kfd_evict_ms` in its result (LEADS.md Part O O3).

Same arguments as abtest.py.  Linux/ROCm only (abtest.py reads /proc); see tools/pancake/kfd_evict.py for what the number is and what it is not."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402
import kfd_evict  # noqa: E402

if __name__ == "__main__":
    kfd_evict.install(abtest)
    abtest.main()
