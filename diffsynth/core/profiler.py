"""Opt-in training traces, following VeOmni's one-shot profiler schedule."""

import os
import socket
import time
import warnings
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path

import torch


_current_profiler = ContextVar("diffsynth_profiler", default=None)


def _env_bool(name, default=False):
    value = os.environ.get(name, str(default)).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (0/1, false/true, no/yes, off/on), got {value!r}.")


def profile_scope(name):
    """Annotate a phase only while this process has a running training profiler."""
    profiler = _current_profiler.get()
    if profiler is not None and profiler.running:
        return torch.profiler.record_function(name)
    return nullcontext()


def _profiler_backend(device):
    device_type = str(device).split(":", 1)[0]
    if device_type == "npu":
        import torch_npu
        module = torch_npu.profiler
        return module, [module.ProfilerActivity.CPU, module.ProfilerActivity.NPU]

    module = torch.profiler
    activities = [module.ProfilerActivity.CPU]
    if device_type != "cpu":
        activity = getattr(module.ProfilerActivity, device_type.upper(), None)
        # MUSA builds expose either MUSA or the renamed PrivateUse1 backend.
        if activity is None and device_type == "musa":
            activity = getattr(module.ProfilerActivity, "PrivateUse1", None)
        if activity is None or activity not in module.supported_activities():
            warnings.warn(f"Profiler activity for {device_type} is unavailable; recording CPU events only.")
        else:
            activities.append(activity)
    return module, activities


class TrainingProfiler:
    """Profile microsteps in the 1-based interval [START_STEP, END_STEP).

    PROFILE_ENABLE is the only required switch. Other PROFILE_* variables are
    optional; disabled/unselected ranks never create a profiler or output files.
    Each training/data-processing invocation gets its own schedule and filename.
    """

    def __init__(self, output_path, device, rank=0, task="train"):
        self.enabled = _env_bool("PROFILE_ENABLE")
        self.running = False
        self.profiler = None
        self.steps = 0
        self._token = None
        if not self.enabled:
            return
        if _env_bool("PROFILE_RANK0_ONLY", True) and rank != 0:
            self.enabled = False
            return

        self.start_step = int(os.environ.get("PROFILE_START_STEP", "3"))
        self.end_step = int(os.environ.get("PROFILE_END_STEP", "6"))
        if not 1 <= self.start_step < self.end_step:
            raise ValueError("PROFILE_START_STEP/PROFILE_END_STEP must satisfy 1 <= START < END.")
        self.trace_dir = Path(os.environ.get("PROFILE_TRACE_DIR", str(Path(output_path) / "profiler")))
        self.worker_name = f"diffsynth_{task}_rank{rank}_{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"
        self.profiler_module, activities = _profiler_backend(device)
        self._npu_handler = None
        if str(device).split(":", 1)[0] == "npu":
            self._npu_handler = self.profiler_module.tensorboard_trace_handler(str(self.trace_dir / self.worker_name))
        warmup = 0 if self.start_step == 1 else 1
        self.profiler = self.profiler_module.profile(
            activities=activities,
            schedule=self.profiler_module.schedule(
                wait=self.start_step - warmup - 1,
                warmup=warmup,
                active=self.end_step - self.start_step,
                repeat=1,
            ),
            on_trace_ready=self._save_trace,
            record_shapes=_env_bool("PROFILE_RECORD_SHAPES", True),
            profile_memory=_env_bool("PROFILE_MEMORY"),
            with_stack=_env_bool("PROFILE_STACK", True),
        )

    def _save_trace(self, profiler):
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if self._npu_handler is not None:
            self._npu_handler(profiler)
            path = self.trace_dir / self.worker_name
        else:
            path = self.trace_dir / f"{self.worker_name}.pt.trace.json.gz"
            profiler.export_chrome_trace(str(path))
        print(f"Profiling trace saved to `{path}`.", flush=True)

    def __enter__(self):
        if self.enabled:
            self.profiler.start()
            self.running = True
            self._token = _current_profiler.set(self)
            print(
                f"Profiling enabled: microsteps [{self.start_step}, {self.end_step}), "
                f"trace directory `{self.trace_dir}`.", flush=True,
            )
        return self

    def step(self):
        if self.running:
            self.steps += 1
            self.profiler.step()
            if self.steps >= self.end_step - 1:
                self.stop()

    def stop(self):
        if self.running:
            self.running = False
            self.profiler.stop()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Also flush a partially recorded window on early exit/exception.
            self.stop()
        finally:
            if self._token is not None:
                _current_profiler.reset(self._token)
                self._token = None

    def iter_dataloader(self, dataloader):
        if not self.enabled:
            return dataloader
        return self._profiled_batches(dataloader)

    def _profiled_batches(self, dataloader):
        with profile_scope("diffsynth/data_loader_init"):
            iterator = iter(dataloader)
        while True:
            with profile_scope("diffsynth/data_loading"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            # Do not keep the data-loading range open across model execution.
            yield batch
