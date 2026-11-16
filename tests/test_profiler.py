import gzip
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import Accelerator

from diffsynth.core.profiler import TrainingProfiler, profile_scope
from diffsynth.diffusion.base_pipeline import PipelineUnit, PipelineUnitRunner
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_data_process_task, launch_training_task


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, cached=False):
        self.load_from_cache = cached

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([index + 1.0, 0.5])


class ToyUnit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("x",))

    def process(self, pipe, x):
        return {"x": x * 0.5}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
        self.pipe = SimpleNamespace(torch_dtype=torch.float32)
        self.unit = ToyUnit()
        self.unit_runner = PipelineUnitRunner()

    def forward(self, data, inputs=None):
        x = data if inputs is None else inputs
        shared, _, _ = self.unit_runner(self.unit, self.pipe, {"x": x}, {}, {})
        return (shared["x"] * self.weight).sum().square()

    def trainable_modules(self):
        return self.parameters()

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return state_dict


class ProfilerTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in list(os.environ):
            if key.startswith("PROFILE_"):
                del os.environ[key]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)

    def configure(self, **kwargs):
        os.environ.update({f"PROFILE_{key}": str(value) for key, value in kwargs.items()})

    def traces(self):
        return sorted(self.output.rglob("*.pt.trace.json.gz"))

    def names(self, path):
        with gzip.open(path, "rt") as stream:
            return {event.get("name") for event in json.load(stream)["traceEvents"]}

    def test_disabled_and_unselected_rank_have_no_profiler_or_files(self):
        self.configure(START_STEP="ignored while disabled")
        for enabled, rank in (("0", 0), ("1", 1)):
            self.configure(ENABLE=enabled)
            with patch("torch.profiler.profile", side_effect=AssertionError("profiler constructed")):
                with patch("torch.profiler.record_function", side_effect=AssertionError("range created")):
                    with TrainingProfiler(self.output, "cpu", rank) as profiler:
                        with profile_scope("disabled"):
                            torch.ones(2).add_(1)
                        profiler.step()
            self.assertFalse(profiler.running)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_schedule_records_only_requested_microsteps(self):
        self.configure(ENABLE="yes", START_STEP=3, END_STEP=6)
        with TrainingProfiler(self.output, "cpu") as profiler:
            for step in range(1, 9):
                with profile_scope(f"microstep_{step}"):
                    torch.ones(4).square()
                profiler.step()
        self.assertEqual(len(self.traces()), 1)
        names = self.names(self.traces()[0])
        self.assertEqual({f"microstep_{i}" for i in range(1, 9)} & names,
                         {"microstep_3", "microstep_4", "microstep_5"})
        self.assertFalse(profiler.running)

    def test_exception_flushes_partial_window_and_clears_scope(self):
        self.configure(ENABLE=1, START_STEP=1, END_STEP=6)
        with self.assertRaisesRegex(RuntimeError, "training failure"):
            with TrainingProfiler(self.output, "cpu"):
                with profile_scope("partial_step"):
                    torch.ones(2).square()
                raise RuntimeError("training failure")
        self.assertEqual(len(self.traces()), 1)
        self.assertIn("partial_step", self.names(self.traces()[0]))
        with patch("torch.profiler.record_function", side_effect=AssertionError("scope leaked")):
            with profile_scope("outside_training"):
                pass

    def test_exit_before_recording_produces_no_trace(self):
        self.configure(ENABLE=1, START_STEP=5, END_STEP=7)
        with TrainingProfiler(self.output, "cpu") as profiler:
            torch.ones(2).square()
            profiler.step()
        self.assertEqual(self.traces(), [])

    def test_ranks_and_invocations_have_unique_files(self):
        self.configure(ENABLE=1, START_STEP=1, END_STEP=2, RANK0_ONLY="false")
        for rank in (0, 1, 0):
            with TrainingProfiler(self.output, "cpu", rank) as profiler:
                torch.ones(2).square()
                profiler.step()
        self.assertEqual(len(self.traces()), 3)
        self.assertTrue(any("rank1_" in path.name for path in self.traces()))

    def test_invalid_configuration_fails_clearly(self):
        self.configure(ENABLE="typo")
        with self.assertRaisesRegex(ValueError, "PROFILE_ENABLE"):
            TrainingProfiler(self.output, "cpu")
        self.configure(ENABLE=1, START_STEP=3, END_STEP=3)
        with self.assertRaisesRegex(ValueError, "START < END"):
            TrainingProfiler(self.output, "cpu")

    def test_runner_preserves_updates_and_profiles_pipeline_units(self):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        for cached in (False, True):
            results = []
            for enabled in (0, 1):
                self.configure(ENABLE=enabled, START_STEP=3, END_STEP=6)
                torch.manual_seed(123)
                model = ToyModel()
                output = self.output / f"cached{cached}_profile{enabled}"
                logger = ModelLogger(str(output))
                launch_training_task(
                    accelerator, ToyDataset(cached), model, logger,
                    num_workers=0, num_epochs=2, save_steps=3,
                )
                self.assertEqual(logger.num_steps, 8)
                self.assertEqual(len(list(output.glob("*.safetensors"))), 3)
                results.append(model.weight.detach().clone())
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        self.assertEqual(len(self.traces()), 2)
        names = self.names(self.traces()[0])
        self.assertTrue({
            "diffsynth/data_loading", "diffsynth/train_step", "diffsynth/forward",
            "diffsynth/backward", "diffsynth/optimizer_step", "diffsynth/lr_scheduler_step",
            "diffsynth/zero_grad", "diffsynth/log_and_checkpoint", "diffsynth/unit/ToyUnit",
        } <= names)

    def test_data_processing_records_and_saves_each_batch(self):
        self.configure(ENABLE=1, START_STEP=2, END_STEP=4)
        launch_data_process_task(
            Accelerator(cpu=True), ToyDataset(), ToyModel(), ModelLogger(str(self.output)), num_workers=0,
        )
        self.assertEqual(len(list((self.output / "0").glob("*.pth"))), 4)
        self.assertEqual(len(self.traces()), 1)
        self.assertTrue({"diffsynth/data_process_step", "diffsynth/save_cache"} <= self.names(self.traces()[0]))


if __name__ == "__main__":
    # start_step=1 intentionally requests profiling without a warmup step.
    warnings.filterwarnings("ignore", message="Profiler won't be using warmup.*")
    unittest.main()
