import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch

from diffsynth.diffusion.logger import ModelLogger


class ModelLoggerTests(unittest.TestCase):
    def test_disabled_skips_all_save_paths_and_collectives_on_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            for main_process in (True, False):
                for save_steps in (None, 1, 2, 0, -1):
                    with self.subTest(main_process=main_process, save_steps=save_steps):
                        accelerator = Mock(is_main_process=main_process)
                        accelerator.get_state_dict.side_effect = AssertionError("gathered disabled checkpoint")
                        output = Path(directory) / "unused"
                        logger = ModelLogger(str(output), disable_model_save=True)
                        model = Mock()
                        for _ in range(3):
                            logger.on_step_end(accelerator, model, save_steps)
                        logger.on_epoch_end(accelerator, model, 0)
                        logger.on_training_end(accelerator, model, save_steps)
                        logger.save_model(accelerator, model, "manual.safetensors")
                        self.assertEqual(logger.num_steps, 3)
                        accelerator.wait_for_everyone.assert_not_called()
                        accelerator.get_state_dict.assert_not_called()
                        accelerator.unwrap_model.assert_not_called()
                        accelerator.save.assert_not_called()
                        self.assertFalse(output.exists())

    def test_disabled_saving_keeps_loss_logging_and_closes_loggers(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = ModelLogger(directory, enable_csv_log=True, disable_model_save=True)
            accelerator = Mock(is_main_process=True)
            model = Mock()
            for value in (0.5, 0.25):
                logger.on_step_end(accelerator, model, 1, loss=torch.tensor(value))
            logger.on_training_end(accelerator, model, 1)
            self.assertTrue(logger.loggers[0].file.closed)
            with open(Path(directory) / "loss.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["step"] for row in rows], ["1", "2"])
            self.assertEqual([float(row["value"]) for row in rows], [0.5, 0.25])
            accelerator.save.assert_not_called()

    def test_default_still_saves_periodic_final_and_epoch_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock()
            state = {"weight": torch.ones(1)}
            model.export_trainable_state_dict.return_value = state
            accelerator = Mock(is_main_process=True)
            accelerator.get_state_dict.return_value = state
            accelerator.unwrap_model.return_value = model
            logger = ModelLogger(directory)
            for _ in range(3):
                logger.on_step_end(accelerator, model, 2)
            logger.on_training_end(accelerator, model, 2)
            logger.on_epoch_end(accelerator, model, 0)
            paths = [Path(call.args[1]).name for call in accelerator.save.call_args_list]
            self.assertEqual(paths, ["step-2.safetensors", "step-3.safetensors", "epoch-0.safetensors"])
            self.assertEqual(accelerator.get_state_dict.call_count, 3)
            self.assertEqual(accelerator.wait_for_everyone.call_count, 3)


if __name__ == "__main__":
    unittest.main()
