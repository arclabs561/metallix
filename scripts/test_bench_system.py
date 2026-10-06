# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CPU-only tests for the benchmark machine probes, on recorded command output."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_system

# Recorded on an M3 Max (128 GB), macOS 26.6.2, while other jobs were running.
VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  2328711.
Pages active:                                1784413.
Pages inactive:                              2092839.
Pages speculative:                           1838588.
Pages throttled:                                   0.
Pages wired down:                             221191.
Pages purgeable:                               20933.
"Translation faults":                      196312538.
Pages copy-on-write:                        21028741.
File-backed pages:                           4090924.
Anonymous pages:                             1624916.
Pages stored in compressor:                        0.
Pages occupied by compressor:                      0.
"""
IOREG = (
    '    | "PerformanceStatistics" = {"In use system memory (driver)"=0,'
    '"Alloc system memory"=4783898624,"Tiler Utilization %"=27,"recoveryCount"=0,'
    '"Renderer Utilization %"=27,"Device Utilization %"=29,"SplitSceneCount"=0,'
    '"In use system memory"=1154580480}\n'
)
SW_VERS = "ProductName:\t\tmacOS\nProductVersion:\t\t26.6.2\nBuildVersion:\t\t25G83\n"
LOCK = """\
[[package]]
name = "mlx-rs"
version = "0.32.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "mlx-sys"
version = "0.6.0"

[[package]]
name = "serde"
version = "1.0.0"
"""


class Parsers(unittest.TestCase):
    def test_vm_stat_used_is_app_plus_wired_plus_compressed(self) -> None:
        pages = 1624916 - 20933 + 221191 + 0
        self.assertEqual(bench_system.parse_vm_stat(VM_STAT), pages * 16384)
        self.assertIsNone(bench_system.parse_vm_stat(""))

    def test_gpu_stats_pick_device_utilization_not_the_driver_field(self) -> None:
        stats = bench_system.parse_gpu_stats(IOREG)
        self.assertEqual(stats, {"utilization_pct": 29, "in_use_bytes": 1154580480})
        self.assertEqual(
            bench_system.parse_gpu_stats(""),
            {"utilization_pct": None, "in_use_bytes": None},
        )

    def test_group_rss_sums_one_process_group(self) -> None:
        ps = "  101   100  2048\n  102   100  1024\n  200   200  9999\n  bad\n"
        self.assertEqual(bench_system.group_rss_bytes(ps, 100), 3072 * 1024)
        self.assertEqual(bench_system.group_rss_bytes(ps, 7), 0)

    def test_sw_vers_and_lock_versions(self) -> None:
        sw = bench_system.parse_sw_vers(SW_VERS)
        self.assertEqual(
            (sw["ProductVersion"], sw["BuildVersion"]), ("26.6.2", "25G83")
        )
        self.assertEqual(
            bench_system.lock_versions(LOCK, ("mlx-rs", "mlx-sys")),
            {"mlx-rs": "0.32.0", "mlx-sys": "0.6.0"},
        )


class SamplerPeaks(unittest.TestCase):
    def test_peaks_and_a_single_abort(self) -> None:
        loads = iter([1.0, 5.0, 3.0, 6.0])
        aborts = []
        sampler = bench_system.Sampler(
            abort_load=4.0,
            on_abort=aborts.append,
            probe=lambda pgid: {"load_1m": next(loads), "system_used_bytes": 7},
        )
        for _ in range(4):
            sampler.take()
        summary = sampler.summary()
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["peak_load_1m"], 6.0)
        self.assertEqual(summary["peak_system_used_bytes"], 7)
        self.assertIsNone(summary["peak_gpu_in_use_bytes"])
        self.assertEqual(len(aborts), 1)  # The second excursion does not re-fire.
        self.assertIn("5.00 rose above 4", summary["aborted"])

    def test_gpu_memory_above_the_cap_aborts_once(self) -> None:
        gpu = iter([10 * 2**30, 70 * 2**30, 80 * 2**30])
        aborts = []
        sampler = bench_system.Sampler(
            abort_gpu_bytes=64 * 2**30,
            on_abort=aborts.append,
            probe=lambda pgid: {"load_1m": 1.0, "gpu_in_use_bytes": next(gpu)},
        )
        sampler.take()
        self.assertIsNone(sampler.aborted)
        sampler.take()
        sampler.take()
        self.assertEqual(aborts, ["GPU memory 70.0 GiB rose above 64 GiB"])
        self.assertEqual(sampler.summary()["peak_gpu_in_use_bytes"], 80 * 2**30)

    def test_no_threshold_never_aborts(self) -> None:
        sampler = bench_system.Sampler(probe=lambda pgid: {"load_1m": 99.0})
        sampler.take()
        self.assertIsNone(sampler.aborted)


# `pmset -g therm` on a cool machine (recorded with VM_STAT above).
THERM_COOL = """\
Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
"""
# Throttled forms, written from pmset's output format, not recorded here.
THERM_HOT = """\
Note: No thermal warning level has been recorded
Performance warning level has been recorded: 1
CPU Power notify
\tCPU_Scheduler_Limit \t= 100
\tCPU_Available_CPUs \t= 16
\tCPU_Speed_Limit \t= 71
"""


class IdleGate(unittest.TestCase):
    def readings(self, **overrides):
        idle = {
            "load_1m": 0.8,
            "build_processes": [],
            "power_source": "AC Power",
            "thermal_warnings": [],
            "gpu_utilization_pct": [0, 1, 0, 2, 0],
        }
        return idle | overrides

    def test_power_thermal_and_process_parsers(self) -> None:
        battery = "Now drawing from 'Battery Power'\n -InternalBattery-0 46%;"
        self.assertEqual(bench_system.parse_power_source(battery), "Battery Power")
        self.assertEqual(
            bench_system.parse_power_source("Now drawing from 'AC Power'\n"),
            "AC Power",
        )
        self.assertEqual(bench_system.parse_thermal_warnings(THERM_COOL), [])
        self.assertEqual(
            bench_system.parse_thermal_warnings(THERM_HOT),
            [
                "Performance warning level has been recorded: 1",
                "CPU_Speed_Limit \t= 71",
            ],
        )
        ps = "/usr/bin/zsh\n/opt/rust/bin/rustc\ncargo\nrustc-wrapper\n"
        self.assertEqual(bench_system.build_processes(ps), ["cargo", "rustc"])

    def test_an_idle_machine_passes(self) -> None:
        self.assertEqual(bench_system.idle_failures(self.readings()), [])

    def test_each_condition_fails_on_its_own(self) -> None:
        cases = {
            "1-min load 2.0 >= 2": {"load_1m": 2.0},
            "running: cargo": {"build_processes": ["cargo"]},
            "power source 'Battery Power', not AC": {"power_source": "Battery Power"},
            "thermal: CPU_Speed_Limit = 71": {
                "thermal_warnings": ["CPU_Speed_Limit = 71"]
            },
            "GPU utilization up to 29% > 5%": {"gpu_utilization_pct": [0, 29, 0]},
            "GPU utilization unavailable": {"gpu_utilization_pct": [0, None]},
        }
        for expected, override in cases.items():
            with self.subTest(expected):
                self.assertEqual(
                    bench_system.idle_failures(self.readings(**override)), [expected]
                )


if __name__ == "__main__":
    unittest.main()
