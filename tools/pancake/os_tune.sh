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
#   sudo tools/pancake/os_tune.sh install    1 (recommended): vm.compact_unevictable_allowed=0 (compaction leaves the
#                                            locked arena alone) + vm.compaction_proactiveness=50 (background compaction),
#                                            in /etc/sysctl.d/90-pancake.conf, applied now; also compacts once
#
# compact_unevictable_allowed=0 matters most: by default the kernel's compaction also migrates mlock'ed pages.  The
# arena's pages are mlock'ed AND registered with the GPU driver (KFD); every migrated page fires an MMU notifier, KFD
# evicts the GPU's queues and restores them (kernel log: "svm_range_restore_work hogged CPU ... 2051 times").  During
# the load that storm kept memory pressure above oomd's limit; during decode it stalls the GPU (the ~25% slow runs).
#   sudo tools/pancake/os_tune.sh install --hugepages
#                                            1 + 2 (optional): reserve 16920 x 2 MiB huge pages at boot for the arena
#                                            (MAP_HUGETLB: no fragmentation, fewer TLB misses; two forks measured +5-7%
#                                            decode).  The pool is taken out of ordinary RAM (33 GiB): only for a PC
#                                            that runs pancake most of the time.  Applied at the next boot.
#   sudo tools/pancake/os_tune.sh cpu        CPU clocks for decode (now, until reboot): the APU's iGPU (Vega, it drives the
#                                            desktop) to its low clock state - at its 2000 MHz top state it takes package
#                                            power the 8 cores need, and the cores fell to 3.4-3.7 GHz under the engine's
#                                            AVX2 load (4.2 GHz all-core without it) - and the deepest core idle state C3
#                                            (350 us wake-up) off, so pool threads wake fast.  'cpu-off' undoes both.  The
#                                            bigger lever is in the BIOS: Precision Boost Overdrive (raises PPT/TDC/EDC;
#                                            Tctl stays at 56-63 C under the engine, so there is room).
#   sudo tools/pancake/os_tune.sh cpu-install   makes 'cpu' permanent: copies this script to /usr/local/sbin and installs
#                                            pancake-cpu.service (runs 'cpu' at every boot), applied now; 'cpu-uninstall'
#                                            removes both and undoes the setting
#   sudo tools/pancake/os_tune.sh uninstall  removes /etc/sysctl.d/90-pancake.conf (the pool is freed at the next boot)
#   sudo tools/pancake/os_tune.sh compact    one compaction pass now (seconds; nothing persistent)
CONF=/etc/sysctl.d/90-pancake.conf
UNIT=/etc/systemd/system/pancake-cpu.service
SBIN=/usr/local/sbin/pancake-os-tune
PAGES=16920   # the IQ2_XS arena: 33.02 GiB in 2 MiB pages (the engine's "needed N 2 MiB pages" log line)

big_free_gib() {   # GiB of free memory in blocks of 2 MiB or more (buddy order >= 9): what the arena can get unfragmented
    awk '$4=="Normal" || $4=="DMA32" { for (o = 9; o <= NF - 5; o++) s += $(o + 5) * 2 ^ o * 4096 } END { printf "%d", s / 2^30 }' \
        /proc/buddyinfo
}

check() {
    pro=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
    unev=$(cat /proc/sys/vm/compact_unevictable_allowed 2>/dev/null || echo "?")
    hp=$(cat /proc/sys/vm/nr_hugepages)
    hpf=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
    avail=$(awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo)
    big=$(big_free_gib)
    echo "pancake memory check:"
    echo "  RAM available:                       $avail GiB"
    echo "  free in blocks of 2 MiB or more:     $big GiB   (the arena locks 33 GiB; low = fragmented)"
    echo "  vm.compact_unevictable_allowed:      $unev         (pancake needs 0: compaction must not move the locked arena)"
    echo "  vm.compaction_proactiveness:         $pro        (pancake recommends 50; the kernel default is 20)"
    echo "  vm.nr_hugepages:                     $hp ($hpf free)   (optional: $PAGES holds the IQ2_XS arena)"
    [ -f "$CONF" ] && echo "  $CONF: present" || echo "  $CONF: not installed"
    if [ "$unev" = 1 ] || { [ "$pro" != "?" ] && [ "$pro" -lt 50 ]; }; then
        echo "  -> recommended: sudo $0 install      (background compaction; prevents the oomd kill during the load)"
    fi
    if [ "$hp" -lt "$PAGES" ]; then
        echo "  -> optional:    sudo $0 install --hugepages   (reserves 33 GiB at boot for the arena; see the header)"
    fi
    if [ "$big" -lt 20 ] && [ "$hpf" -lt "$PAGES" ]; then
        echo "  -> RAM is fragmented now: sudo $0 compact   before starting the engine"
    fi
}

igpus() {   # the APU's own GPU (Renoir / Cezanne / Barcelo ids), never the discrete card
    for c in /sys/class/drm/card[0-9]*; do
        case "$(cat "$c/device/device" 2>/dev/null)" in 0x1636|0x1638|0x15e7) echo "$c/device" ;; esac
    done
}

iommu_check() {   # A53: the dGPU's IOMMU group type (identity = passthrough, DMA / DMA-FQ = translated); sysfs reads only
    found=0
    for d in /sys/bus/pci/devices/*; do
        case "$(cat "$d/class" 2>/dev/null)" in 0x03*) ;; *) continue ;; esac
        case "$(cat "$d/device" 2>/dev/null)" in 0x1636|0x1638|0x15e7) continue ;; esac   # not the APU's own graphics
        t=$(cat "$d/iommu_group/type" 2>/dev/null || echo "none")
        found=1
        echo "  IOMMU group of GPU $(basename "$d"): $t   (identity = passthrough, DMA / DMA-FQ = translated)"
        case "$t" in
            DMA|DMA-FQ) echo "  -> translated: PCIe DMA is address-translated and can cost throughput; 'iommu=pt' on the kernel command line avoids it (boot loader option + reboot; not for a PC that passes GPUs to VMs)" ;;
        esac
    done
    [ "$found" = 1 ] || echo "  IOMMU: no GPU device found in sysfs ($(ls /sys/class/iommu 2>/dev/null | wc -l) IOMMU units)"
}

cpu_state() {
    for g in $(igpus); do echo "  iGPU $(basename "$(readlink -f "$g")"): $(cat "$g/power_dpm_force_performance_level") (sclk now: $(grep '\*' "$g/pp_dpm_sclk" | awk '{print $2}'))"; done
    c3=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state3/disable 2>/dev/null || echo "?")
    echo "  core idle state C3 disabled: $c3   (pancake: 1 during benchmarks / daily decode)"
    [ -f "$UNIT" ] && echo "  $UNIT: present (applied at boot)" || echo "  $UNIT: not installed (cpu setting lasts until reboot)"
}

need_root() { [ "$(id -u)" = 0 ] || { echo "$0 $1: run it with sudo" >&2; exit 1; }; }

case "${1:-check}" in
    check) check; cpu_state; iommu_check ;;
    cpu)
        need_root cpu
        for g in $(igpus); do echo low > "$g/power_dpm_force_performance_level"; done
        for d in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state3/disable; do echo 1 > "$d"; done
        echo "cpu: iGPU at its low clock, C3 off (until reboot; 'cpu-off' undoes)"; cpu_state ;;
    cpu-off)
        need_root cpu-off
        for g in $(igpus); do echo auto > "$g/power_dpm_force_performance_level"; done
        for d in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state3/disable; do echo 0 > "$d"; done
        cpu_state ;;
    cpu-install)
        need_root cpu-install
        install -m 755 "$0" "$SBIN"
        cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=pancake-strata: iGPU low clock + core C3 off, so the CPU keeps its clocks during decode
After=systemd-modules-load.service systemd-udev-settle.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$SBIN cpu
ExecStop=$SBIN cpu-off

[Install]
WantedBy=multi-user.target
UNIT_EOF
        systemctl daemon-reload && systemctl enable --now pancake-cpu.service && echo "installed $UNIT (applied now and at every boot)"
        cpu_state ;;
    cpu-uninstall)
        need_root cpu-uninstall
        systemctl disable --now pancake-cpu.service 2>/dev/null
        rm -f "$UNIT" "$SBIN"; systemctl daemon-reload; echo "removed $UNIT"
        cpu_state ;;
    compact)
        need_root compact
        echo 1 > /proc/sys/vm/compact_memory && echo "compacted: $(big_free_gib) GiB free in blocks of 2 MiB or more" ;;
    install)
        need_root install
        {
            echo "# pancake-strata (tools/pancake/os_tune.sh): keep free RAM in large blocks for the 33 GiB expert arena"
            echo "vm.compact_unevictable_allowed = 0"
            echo "vm.compaction_proactiveness = 50"
            if [ "$2" = "--hugepages" ]; then
                echo "# optional: 2 MiB huge pages reserved at boot for the expert arena (taken out of ordinary RAM)"
                echo "vm.nr_hugepages = $PAGES"
            fi
        } > "$CONF"
        sysctl -q -w vm.compact_unevictable_allowed=0 vm.compaction_proactiveness=50
        echo 1 > /proc/sys/vm/compact_memory
        echo "installed $CONF (compaction_proactiveness=50 now$( [ "$2" = "--hugepages" ] && echo ", $PAGES huge pages from the next boot"))"
        check ;;
    uninstall)
        need_root uninstall
        rm -f "$CONF" && sysctl -q -w vm.compaction_proactiveness=20 vm.compact_unevictable_allowed=1 && echo "removed $CONF" ;;
    *) sed -n '2,35p' "$0"; exit 2 ;;
esac
