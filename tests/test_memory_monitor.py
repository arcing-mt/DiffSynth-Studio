import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from diffsynth.core.memory_monitor import PeakMemoryMonitor


class FakeDevice:
    def __init__(self):
        self.allocated = 10 * 1024**2
        self.reserved = 20 * 1024**2
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved
        self.reset_count = 0
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1

    def reset_peak_memory_stats(self, device=None):
        self.reset_count += 1
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def memory_allocated(self, device=None):
        return self.allocated

    def memory_reserved(self, device=None):
        return self.reserved

    def max_memory_allocated(self, device=None):
        return self.peak_allocated

    def max_memory_reserved(self, device=None):
        return self.peak_reserved


class PeakMemoryMonitorTests(unittest.TestCase):
    def test_disabled_monitor_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            device = FakeDevice()
            monitor = PeakMemoryMonitor(directory, "fake:0", device_api=device)
            with monitor:
                monitor.before_step(1)
                monitor.after_step(1, 1, True)
            self.assertFalse(monitor.enabled)
            self.assertEqual(list(Path(directory).rglob("*")), [])
            self.assertEqual(device.sync_count, 0)

    def test_records_per_step_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MEMORY_MONITOR_ENABLE": "1", "MEMORY_MONITOR_INTERVAL": "2",
        }, clear=True):
            device = FakeDevice()
            monitor = PeakMemoryMonitor(directory, "fake:0", rank=3, device_api=device)
            with monitor:
                monitor.before_step(1)
                monitor.after_step(1, 0, False)
                monitor.before_step(2)
                device.allocated = 30 * 1024**2
                device.reserved = 40 * 1024**2
                device.peak_allocated = 50 * 1024**2
                device.peak_reserved = 60 * 1024**2
                monitor.after_step(2, 1, True)
                monitor.before_step(4)
                device.peak_allocated = 70 * 1024**2
                device.peak_reserved = 80 * 1024**2
                monitor.after_step(4, 2, True)
            output = Path(directory) / "peak_memory"
            with (output / "peak_memory_rank3.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["microstep"] for row in rows], ["2", "4"])
            self.assertEqual(rows[0]["peak_allocated_mb"], "50.0")
            self.assertEqual(rows[1]["peak_reserved_mb"], "80.0")
            summary = json.loads((output / "peak_memory_summary_rank3.json").read_text())
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["max_peak_allocated_mb"], 70.0)
            self.assertEqual(summary["max_peak_reserved_mb"], 80.0)
            self.assertEqual(device.reset_count, 2)
            self.assertGreaterEqual(device.sync_count, 4)

    def test_invalid_interval_fails_clearly(self):
        with patch.dict(os.environ, {"MEMORY_MONITOR_ENABLE": "1", "MEMORY_MONITOR_INTERVAL": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                PeakMemoryMonitor(tempfile.gettempdir(), "fake:0", device_api=FakeDevice())


if __name__ == "__main__":
    unittest.main()
