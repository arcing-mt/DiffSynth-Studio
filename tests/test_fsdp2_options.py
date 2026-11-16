import argparse
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

from diffsynth.diffusion.fsdp2 import (
    FSDP2TrainingOptions,
    add_fsdp2_training_options,
    configure_fsdp2_training,
    fsdp2_accelerator_kwargs,
)


def parse_options(*tokens):
    return add_fsdp2_training_options(argparse.ArgumentParser()).parse_args(tokens)


def fake_accelerator(precision="bf16", fsdp2=True):
    return SimpleNamespace(
        is_fsdp2=fsdp2, mixed_precision=precision,
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(mixed_precision_policy=None)),
        gradient_state=SimpleNamespace(plugin_kwargs={"num_steps": 4}),
    )


class FSDP2OptionsTests(unittest.TestCase):
    def test_defaults_preserve_h3_precision_and_accumulation(self):
        args = parse_options()
        accelerator = fake_accelerator()
        self.assertEqual(fsdp2_accelerator_kwargs(args), {})
        configure_fsdp2_training(accelerator, args)
        policy = accelerator.state.fsdp_plugin.mixed_precision_policy
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.bfloat16)
        self.assertIsNone(policy.output_dtype)
        self.assertFalse(policy.cast_forward_inputs)
        self.assertEqual(accelerator.gradient_state.plugin_kwargs, {"num_steps": 4, "sync_each_batch": False})
        options = accelerator.diffsynth_fsdp2_options
        self.assertTrue(options.reshard_after_backward)
        self.assertFalse(options.hsdp_defer_all_reduce)
        self.assertEqual((options.forward_prefetch, options.backward_prefetch), (0, 0))

    def test_independent_precision_and_sync_controls(self):
        args = parse_options(
            "--fsdp2_sync_each_batch", "true", "--fsdp2_reshard_after_backward", "false",
            "--fsdp2_param_dtype", "float16", "--fsdp2_reduce_dtype", "float32",
            "--fsdp2_output_dtype", "float32", "--fsdp2_cast_forward_inputs", "true",
        )
        self.assertEqual(fsdp2_accelerator_kwargs(args)["mixed_precision"], "fp16")
        accelerator = fake_accelerator("fp16")
        configure_fsdp2_training(accelerator, args)
        policy = accelerator.state.fsdp_plugin.mixed_precision_policy
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertEqual(policy.output_dtype, torch.float32)
        self.assertTrue(policy.cast_forward_inputs)
        self.assertEqual(accelerator.gradient_state.plugin_kwargs, {"num_steps": 4, "sync_each_batch": True})
        self.assertFalse(accelerator.diffsynth_fsdp2_options.reshard_after_backward)
        self.assertTrue(accelerator.diffsynth_fsdp2_options.cast_forward_inputs)
        self.assertEqual(accelerator.diffsynth_fsdp2_options.computation_dtype, torch.float16)
        self.assertEqual(fsdp2_accelerator_kwargs(parse_options("--fsdp2_param_dtype", "float32"))["mixed_precision"], "no")

    def test_output_dtype_is_independent_of_input_casting(self):
        accelerator = fake_accelerator()
        configure_fsdp2_training(accelerator, parse_options("--fsdp2_output_dtype", "float32"))
        policy = accelerator.state.fsdp_plugin.mixed_precision_policy
        self.assertEqual(policy.output_dtype, torch.float32)
        self.assertFalse(policy.cast_forward_inputs)
        self.assertFalse(accelerator.diffsynth_fsdp2_options.cast_forward_inputs)

    def test_output_casts_preserve_h3_metadata_and_checkpoint_gradients(self):
        import copy
        from diffsynth.models import minimax_h3_dit as h3
        from test_minimax_h3_musa_patch import make_model, make_inputs, run_backward

        for checkpoint, offload in ((False, False), (True, False), (True, True)):
            with self.subTest(checkpoint=checkpoint, offload=offload):
                model = make_model().to(dtype=torch.bfloat16)
                reference = copy.deepcopy(model)
                inputs = make_inputs(dtype=torch.bfloat16)
                positions = inputs["img_position_ids"].clone()
                observed = []

                def check_input(module, args, kwargs):
                    observed.append((args[0].dtype, kwargs["rope_freqs"].dtype))

                for block in model.blocks:
                    block.register_forward_pre_hook(check_input, with_kwargs=True)
                    # Simulate FSDP's per-block output conversion on CPU.
                    block.register_forward_hook(lambda module, args, output: output.float())
                with patch.object(h3, "attention_forward", torch.nn.functional.scaled_dot_product_attention):
                    expected = run_backward(reference, inputs)
                    actual = run_backward(model, inputs, use_gradient_checkpointing=checkpoint,
                                          use_gradient_checkpointing_offload=offload)
                self.assertTrue(observed)
                self.assertTrue(all(pair == (torch.bfloat16, torch.float32) for pair in observed))
                torch.testing.assert_close(inputs["img_position_ids"], positions, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_other_backends_unchanged_and_explicit_options_rejected(self):
        accelerator = fake_accelerator(fsdp2=False)
        configure_fsdp2_training(accelerator, parse_options())
        self.assertFalse(hasattr(accelerator, "diffsynth_fsdp2_options"))
        with self.assertRaisesRegex(ValueError, "FSDP2 launch"):
            configure_fsdp2_training(accelerator, parse_options("--fsdp2_sync_each_batch"))

    def test_hsdp_mesh_inference_and_validation(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            for tokens in (
                ("--fsdp2_dp_replicate_size", "2"),
                ("--fsdp2_dp_shard_size", "4"),
                ("--fsdp2_dp_replicate_size", "2", "--fsdp2_dp_shard_size", "4"),
            ):
                config = fsdp2_accelerator_kwargs(parse_options(*tokens))["parallelism_config"]
                self.assertEqual((config.dp_replicate_size, config.dp_shard_size), (2, 4))
            for tokens in (
                ("--fsdp2_dp_replicate_size", "3"),
                ("--fsdp2_dp_shard_size", "0"),
                ("--fsdp2_dp_replicate_size", "2", "--fsdp2_dp_shard_size", "2"),
                ("--fsdp2_dp_replicate_size", "8"),
            ):
                with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                    fsdp2_accelerator_kwargs(parse_options(*tokens))

    def test_hsdp_deferral_requires_sync_and_resolved_hsdp_mesh(self):
        accelerator = fake_accelerator()
        accelerator.parallelism_config = SimpleNamespace(dp_replicate_size=2, dp_shard_size=2)
        args = parse_options("--fsdp2_hsdp_defer_all_reduce", "true")
        with self.assertRaisesRegex(ValueError, "sync_each_batch true"):
            configure_fsdp2_training(accelerator, args)
        self.assertEqual(accelerator.gradient_state.plugin_kwargs, {"num_steps": 4})
        self.assertIsNone(accelerator.state.fsdp_plugin.mixed_precision_policy)
        args.fsdp2_sync_each_batch = True
        for config in (None, SimpleNamespace(dp_replicate_size=1, dp_shard_size=4),
                       SimpleNamespace(dp_replicate_size=4, dp_shard_size=1)):
            accelerator.parallelism_config = config
            with self.assertRaisesRegex(ValueError, "requires HSDP"):
                configure_fsdp2_training(accelerator, args)
        accelerator.parallelism_config = SimpleNamespace(dp_replicate_size=2, dp_shard_size=2)
        configure_fsdp2_training(accelerator, args)
        self.assertTrue(accelerator.diffsynth_fsdp2_options.hsdp_defer_all_reduce)
        self.assertEqual(accelerator.gradient_state.plugin_kwargs, {"num_steps": 4, "sync_each_batch": True})

    def test_hsdp_deferral_is_independent_of_parameter_resharding(self):
        for reshard in (False, True):
            options = FSDP2TrainingOptions(torch.bfloat16, reshard_after_backward=reshard,
                                          hsdp_defer_all_reduce=True)
            model = Mock()
            accelerator = SimpleNamespace(sync_gradients=False)
            with options.training(model):
                for boundary in (False, False, False, True, True):
                    accelerator.sync_gradients = boundary
                    with options.microstep(accelerator, model):
                        self.assertEqual(options._all_reduce_enabled, boundary)
            self.assertEqual(model.set_requires_all_reduce.call_args_list, [call(False), call(True)])
            self.assertEqual(model.set_reshard_after_backward.call_args_list,
                             [] if reshard else [call(False), call(True)])
            model.set_requires_gradient_sync.assert_not_called()

    def test_hsdp_deferral_disabled_does_not_change_all_reduce(self):
        options = FSDP2TrainingOptions(torch.bfloat16, reshard_after_backward=False)
        model = Mock()
        with options.training(model):
            for boundary in (False, False, False, True):
                with options.microstep(SimpleNamespace(sync_gradients=boundary), model):
                    pass
        model.set_requires_all_reduce.assert_not_called()

    def test_hsdp_reduction_state_is_restored_on_early_or_exceptional_exit(self):
        for where in ("normal", "microstep", "dataloader"):
            options = FSDP2TrainingOptions(torch.bfloat16, reshard_after_backward=False,
                                          hsdp_defer_all_reduce=True)
            model = Mock()
            try:
                with options.training(model):
                    with options.microstep(SimpleNamespace(sync_gradients=False), model):
                        if where == "microstep":
                            raise RuntimeError("backward failed")
                    if where == "dataloader":
                        raise RuntimeError("data loading failed")
            except RuntimeError:
                pass
            self.assertEqual(model.set_requires_all_reduce.call_args_list, [call(False), call(True)])
            self.assertEqual(model.set_reshard_after_backward.call_args_list, [call(False), call(True)])

    def test_prefetch_order_and_default_behavior(self):
        class Block:
            def __init__(self):
                self.set_modules_to_forward_prefetch = Mock()
                self.set_modules_to_backward_prefetch = Mock()

        blocks = [Block() for _ in range(4)]
        model = SimpleNamespace(pipe=SimpleNamespace(dit=SimpleNamespace(blocks=blocks)))
        options = FSDP2TrainingOptions(torch.bfloat16)
        options.prepare_model(model)
        blocks[0].set_modules_to_forward_prefetch.assert_not_called()
        blocks[0].set_modules_to_backward_prefetch.assert_not_called()
        with patch("torch.distributed.fsdp.FSDPModule", Block):
            FSDP2TrainingOptions(torch.bfloat16, forward_prefetch=2, backward_prefetch=2).prepare_model(model)
        blocks[0].set_modules_to_forward_prefetch.assert_called_once_with(blocks[1:3])
        blocks[3].set_modules_to_forward_prefetch.assert_called_once_with([])
        blocks[3].set_modules_to_backward_prefetch.assert_called_once_with([blocks[2], blocks[1]])

    def test_resharding_follows_optimizer_boundary_and_resets_on_error(self):
        options = FSDP2TrainingOptions(torch.bfloat16, reshard_after_backward=False)
        model = Mock()
        accelerator = SimpleNamespace(sync_gradients=False)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with options.microstep(accelerator, model):
                model.set_reshard_after_backward.assert_called_with(False)
                raise RuntimeError("test failure")
        model.set_reshard_after_backward.assert_called_with(True)
        accelerator.sync_gradients = True
        with options.microstep(accelerator, model):
            model.set_reshard_after_backward.assert_called_with(True)

    def test_resharding_only_sets_at_accumulation_window_boundaries(self):
        options = FSDP2TrainingOptions(torch.bfloat16, reshard_after_backward=False)
        model = Mock()
        accelerator = SimpleNamespace(sync_gradients=False)
        with options.training(model):
            for boundary in (False, False, False, True, True):
                accelerator.sync_gradients = boundary
                with options.microstep(accelerator, model):
                    self.assertEqual(options._reshard_enabled, boundary)
        self.assertEqual(model.set_reshard_after_backward.call_args_list, [call(False), call(True)])

    def test_training_exit_restores_state_between_microsteps(self):
        for error in (False, True):
            options = FSDP2TrainingOptions(torch.bfloat16, reshard_after_backward=False)
            model = Mock()
            try:
                with options.training(model):
                    with options.microstep(SimpleNamespace(sync_gradients=False), model):
                        pass
                    if error:
                        raise RuntimeError("data loading failed")
            except RuntimeError:
                pass
            self.assertEqual(model.set_reshard_after_backward.call_args_list, [call(False), call(True)])


if __name__ == "__main__":
    unittest.main()
