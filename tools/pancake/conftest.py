"""Shared pytest setup for tools/pancake.

abtest.py reads os.sysconf("SC_CLK_TCK") at import (Linux only).  On a host without os.sysconf (Windows) that
fails every module that imports abtest at collection time, so give those modules a harmless stand-in.  The guard is
a hasattr check: on Linux os.sysconf exists and this file changes nothing.
"""
import os

if not hasattr(os, "sysconf"):
    os.sysconf = lambda name: 100  # clock ticks per second, the usual Linux value
