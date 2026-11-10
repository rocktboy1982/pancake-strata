#!/bin/sh
# os_tune.sh - Linux memory settings for pancake's 33 GiB expert arena (check without root, install with sudo).
#
# Why: the engine locks ~33 GiB of experts in RAM in a few seconds at start.  On a machine that has been up for a while
# (many engine starts and stops, desktop apps) free RAM is split into small blocks; the kernel then stalls in direct
# reclaim and compaction while the arena is locked, memory pressure goes above 50% for 20 s, and systemd-oomd kills the
# engine during its load (2026-11-10: 13 starts in a row, at ~34 GB of 39).  The same fragmentation, below the kill
# threshold, made runs ~25% slower.
#
#   tools/pancake/os_tune.sh                 check: prints the current state and what to do (no root, changes nothing)
#   sudo tools/pancake/os_tune.sh install    1 (recommended): background compaction, vm.compaction_proactiveness=50,
#                                            in /etc/sysctl.d/90-pancake.conf, applied now; also compacts once
#   sudo tools/pancake/os_tune.sh install --hugepages
#                                            1 + 2 (optional): reserve 16920 x 2 MiB huge pages at boot for the arena
#                                            (MAP_HUGETLB: no fragmentation, fewer TLB misses; two forks measured +5-7%
#                                            decode).  The pool is taken out of ordinary RAM (33 GiB): only for a PC
#                                            that runs pancake most of the time.  Applied at the next boot.
#   sudo tools/pancake/os_tune.sh uninstall  removes /etc/sysctl.d/90-pancake.conf (the pool is freed at the next boot)
#   sudo tools/pancake/os_tune.sh compact    one compaction pass now (seconds; nothing persistent)
CONF=/etc/sysctl.d/90-pancake.conf
PAGES=16920   # the IQ2_XS arena: 33.02 GiB in 2 MiB pages (the engine's "needed N 2 MiB pages" log line)

big_free_gib() {   # GiB of free memory in blocks of 2 MiB or more (buddy order >= 9): what the arena can get unfragmented
    awk '$4=="Normal" || $4=="DMA32" { for (o = 9; o <= NF - 5; o++) s += $(o + 5) * 2 ^ o * 4096 } END { printf "%d", s / 2^30 }' \
        /proc/buddyinfo
}

check() {
    pro=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
    hp=$(cat /proc/sys/vm/nr_hugepages)
    hpf=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
    avail=$(awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo)
    big=$(big_free_gib)
    echo "pancake memory check:"
    echo "  RAM available:                       $avail GiB"
    echo "  free in blocks of 2 MiB or more:     $big GiB   (the arena locks 33 GiB; low = fragmented)"
    echo "  vm.compaction_proactiveness:         $pro        (pancake recommends 50; the kernel default is 20)"
    echo "  vm.nr_hugepages:                     $hp ($hpf free)   (optional: $PAGES holds the IQ2_XS arena)"
    [ -f "$CONF" ] && echo "  $CONF: present" || echo "  $CONF: not installed"
    if [ "$pro" != "?" ] && [ "$pro" -lt 50 ]; then
        echo "  -> recommended: sudo $0 install      (background compaction; prevents the oomd kill during the load)"
    fi
    if [ "$hp" -lt "$PAGES" ]; then
        echo "  -> optional:    sudo $0 install --hugepages   (reserves 33 GiB at boot for the arena; see the header)"
    fi
    if [ "$big" -lt 20 ] && [ "$hpf" -lt "$PAGES" ]; then
        echo "  -> RAM is fragmented now: sudo $0 compact   before starting the engine"
    fi
}

need_root() { [ "$(id -u)" = 0 ] || { echo "$0 $1: run it with sudo" >&2; exit 1; }; }

case "${1:-check}" in
    check) check ;;
    compact)
        need_root compact
        echo 1 > /proc/sys/vm/compact_memory && echo "compacted: $(big_free_gib) GiB free in blocks of 2 MiB or more" ;;
    install)
        need_root install
        {
            echo "# pancake-strata (tools/pancake/os_tune.sh): keep free RAM in large blocks for the 33 GiB expert arena"
            echo "vm.compaction_proactiveness = 50"
            if [ "$2" = "--hugepages" ]; then
                echo "# optional: 2 MiB huge pages reserved at boot for the expert arena (taken out of ordinary RAM)"
                echo "vm.nr_hugepages = $PAGES"
            fi
        } > "$CONF"
        sysctl -q -w vm.compaction_proactiveness=50
        echo 1 > /proc/sys/vm/compact_memory
        echo "installed $CONF (compaction_proactiveness=50 now$( [ "$2" = "--hugepages" ] && echo ", $PAGES huge pages from the next boot"))"
        check ;;
    uninstall)
        need_root uninstall
        rm -f "$CONF" && sysctl -q -w vm.compaction_proactiveness=20 && echo "removed $CONF" ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
