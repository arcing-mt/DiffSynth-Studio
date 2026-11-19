"""Optional per-step device memory peak monitoring for training."""

import csv
import json
import os
from pathlib import Path

import torch


def _env_bool(name, default=False):
    value = os.environ.get(name, "1" if default else "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (0/1, false/true, no/yes, off/on), got {value!r}.")


class PeakMemoryMonitor:
    """Record allocated and reserved device-memory peaks for selected steps.

    The monitor is disabled by default. When enabled, it synchronizes before
    resetting and after reading the backend counters so asynchronous kernels are
    included in the measured step. Each rank writes an independent CSV and JSON
    summary, avoiding distributed collectives in the training loop.
    """

    _FIELDS = (
        "microstep", "optimizer_step", "sync_gradients",
        "allocated_mb", "reserved_mb", "peak_allocated_mb", "peak_reserved_mb",
    )

    def __init__(self, output_path, device, rank=0, device_api=None):
        self.enabled = _env_bool("MEMORY_MONITOR_ENABLE")
        self.running = False
        self._active = False
        self._reset_supported = False
        self._records = []
        self._max_peak_allocated = 0
        self._max_peak_reserved = 0
        self._device = device
        self._device_api = device_api
        self._file = None
        self._writer = None
        self._csv_path = None
        self._summary_path = None
        if not self.enabled:
            return

        try:
            self.interval = int(os.environ.get("MEMORY_MONITOR_INTERVAL", "1"))
        except ValueError as error:
            raise ValueError("MEMORY_MONITOR_INTERVAL must be a positive integer.") from error
        if self.interval < 1:
            raise ValueError("MEMORY_MONITOR_INTERVAL must be a positive integer.")

        device_type = str(device).split(":", 1)[0]
        self._device_api = device_api or getattr(torch, device_type, None)
        if self._device_api is None or not callable(getattr(self._device_api, "memory_allocated", None)):
            # CPU and unsupported/private backends have no allocator counters.
            self.enabled = False
            return

        memory_dir = Path(os.environ.get("MEMORY_MONITOR_DIR", str(Path(output_path) / "peak_memory")))
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = memory_dir / f"peak_memory_rank{rank}.csv"
        self._summary_path = memory_dir / f"peak_memory_summary_rank{rank}.json"
        self._file = self._csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS, lineterminator="\n")
        self._writer.writeheader()
        self._file.flush()

    @staticmethod
    def _call(function, device):
        try:
            return function(device)
        except (TypeError, RuntimeError, ValueError):
            return function()

    def _synchronize(self):
        synchronize = getattr(self._device_api, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def _counter(self, name, fallback=0):
        function = getattr(self._device_api, name, None)
        if not callable(function):
            return fallback
        try:
            return int(self._call(function, self._device))
        except (TypeError, RuntimeError, ValueError):
            return fallback

    def __enter__(self):
        self.running = self.enabled
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def before_step(self, microstep):
        if not self.running or microstep % self.interval != 0:
            self._active = False
            return
        self._synchronize()
        reset = getattr(self._device_api, "reset_peak_memory_stats", None)
        if callable(reset):
            try:
                self._call(reset, self._device)
                self._reset_supported = True
            except (TypeError, RuntimeError):
                self._reset_supported = False
        self._active = True

    def after_step(self, microstep, optimizer_step, sync_gradients):
        if not self.running or not self._active:
            return
        self._synchronize()
        allocated = self._counter("memory_allocated")
        reserved = self._counter("memory_reserved")
        peak_allocated = self._counter("max_memory_allocated", allocated)
        peak_reserved = self._counter("max_memory_reserved", reserved)
        row = {
            "microstep": microstep,
            "optimizer_step": optimizer_step,
            "sync_gradients": int(bool(sync_gradients)),
            "allocated_mb": round(allocated / 1024**2, 3),
            "reserved_mb": round(reserved / 1024**2, 3),
            "peak_allocated_mb": round(peak_allocated / 1024**2, 3),
            "peak_reserved_mb": round(peak_reserved / 1024**2, 3),
        }
        self._writer.writerow(row)
        self._file.flush()
        self._records.append(row)
        self._max_peak_allocated = max(self._max_peak_allocated, peak_allocated)
        self._max_peak_reserved = max(self._max_peak_reserved, peak_reserved)
        self._active = False

    def close(self):
        if self._file is None:
            self.running = False
            return
        if not self._file.closed:
            self._file.close()
        summary = {
            "records": len(self._records),
            "reset_peak_memory_stats": self._reset_supported,
            "max_peak_allocated_mb": round(self._max_peak_allocated / 1024**2, 3),
            "max_peak_reserved_mb": round(self._max_peak_reserved / 1024**2, 3),
            "csv": str(self._csv_path),
        }
        self._summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.running = False
        self._file = None
