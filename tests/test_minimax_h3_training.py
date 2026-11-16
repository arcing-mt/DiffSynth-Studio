import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from diffsynth.diffusion.loss import FlowMatchSFTMiniMaxH3AudioVideoLoss
from diffsynth.diffusion.minimax_h3_training import (
    FP32_MODULES, MiniMaxH3SyntheticDataset, configure_h3_fp32_modules,
    shard_h3_fp32_modules,
)
from diffsynth.models import minimax_h3_dit as h3
from test_minimax_h3_musa_patch import make_model, make_inputs


class H3TrainingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"MINIMAX_H3_MUSA_FUSED_ROPE": "0", "MINIMAX_H3_MUSA_FUSED_SWIGLU": "0"})
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(42)

    def test_synthetic_geometry_and_fixed_rank_samples(self):
        large = MiniMaxH3SyntheticDataset(720, 1280, 240, 1920, world_size=32, accumulation=2)
        self.assertEqual((large.grid, large.video_tokens, large.audio_tokens, large.seq_len),
                         ((60, 22, 40), 52800, 800, 53856))
        self.assertEqual(len(large) // 32 // 2, 30)
        data = MiniMaxH3SyntheticDataset(32, 32, 5, 16, world_size=2, accumulation=2)
        first, second = data[1], data[3]  # rank 1, microsteps 0 and 1
        self.assertIs(first, data[5])
        self.assertIs(second, data[7])
        shared, positive, _ = first
        expected = torch.randn((2, 96), generator=torch.Generator().manual_seed(271828 + 2 * 97 + 10007)).bfloat16().float()
        torch.testing.assert_close(h3.patchify_video(shared["input_latents"]), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(shared["input_latents"], second[0]["input_latents"]))
        self.assertEqual(positive["prompt_embeds"].dtype, torch.bfloat16)
        packed = shared["packed"]
        self.assertTrue((packed["token_tags"][packed["text_pos"]] == 1).all())
        self.assertTrue((packed["token_tags"][packed["audio_pos"]] == 2).all())
        torch.testing.assert_close(packed["cu_seqlens"], torch.tensor([0, data.seq_len], dtype=torch.int32))

    def test_loss_default_zero_and_disabled_flow_matching(self):
        video = torch.tensor([2., 4.], requires_grad=True)
        audio = torch.tensor([3., 5.], requires_grad=True)
        sched = Mock(timesteps=torch.tensor([500.]))
        sched.add_noise.return_value = torch.zeros(2)
        sched.training_target.return_value = torch.ones(2)
        sched.training_weight.return_value = 3.
        pipe = SimpleNamespace(device="cpu", in_iteration_models=[], scheduler=sched, scheduler_audio=sched,
                               model_fn=Mock(return_value=(video, audio)))
        inputs = dict(input_latents=torch.zeros(2), audio_input_latents=torch.zeros(2), synthetic_inputs=True)
        with patch.dict(os.environ):
            os.environ.pop("MINIMAX_H3_ZERO_TARGET_LOSS", None)
            loss = FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, audio_loss_weight=7., **inputs)
        self.assertEqual(loss.item(), 27.)  # No scheduler weighting or audio weighting.
        sched.add_noise.assert_not_called()
        sched.training_target.assert_not_called()
        self.assertIs(pipe.model_fn.call_args.kwargs["video_latents"], inputs["input_latents"])
        self.assertEqual(pipe.model_fn.call_args.kwargs["timestep_video"].item(), 500.)
        loss.backward()
        torch.testing.assert_close(video.grad, video.detach())
        with patch.dict(os.environ, {"MINIMAX_H3_ZERO_TARGET_LOSS": "0"}):
            loss = FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, audio_loss_weight=2., **inputs)
        expected = (video - 1).square().mean() * 3 + (audio - 1).square().mean() * 6
        torch.testing.assert_close(loss, expected)
        self.assertEqual(sched.add_noise.call_count, 2)

    def test_fp32_default_off_and_load_original_weights(self):
        original = make_model()
        model = make_model().bfloat16()
        with patch.dict(os.environ):
            os.environ.pop("MINIMAX_H3_FP32_MODULES", None)
            configure_h3_fp32_modules(model)
        self.assertFalse(hasattr(model, "_h3_fp32_modules"))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "model.safetensors")
            save_file(original.state_dict(), checkpoint)
            with patch.dict(os.environ, {"MINIMAX_H3_FP32_MODULES": "1"}):
                configure_h3_fp32_modules(model, [SimpleNamespace(path=checkpoint)])
        for name in FP32_MODULES:
            actual = model.get_submodule(name)
            self.assertTrue(all(p.dtype == torch.float32 for p in actual.parameters()))
            torch.testing.assert_close(actual.state_dict(), original.get_submodule(name).state_dict(), rtol=0, atol=0)
        self.assertEqual(model.blocks[0].attn.qkv_proj.weight.dtype, torch.bfloat16)

    def test_fp32_compute_and_backward_with_checkpointing_and_autocast(self):
        model = make_model().bfloat16()
        with patch.dict(os.environ, {"MINIMAX_H3_FP32_MODULES": "1"}):
            configure_h3_fp32_modules(model)
        dtypes = {}
        def hook(name):
            def record(module, inputs, output):
                dtypes[name] = output.dtype
            return record
        for name in (*FP32_MODULES, "blocks.0.attn.qkv_proj"):
            model.get_submodule(name).register_forward_hook(hook(name))
        inputs = make_inputs(dtype=torch.bfloat16)
        with patch.object(h3, "attention_forward", F.scaled_dot_product_attention):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs = model(**inputs, use_gradient_checkpointing=True)
                loss = sum(x.float().square().mean() for x in outputs)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name in FP32_MODULES:
            self.assertEqual(dtypes[name], torch.float32)
        self.assertEqual(dtypes["blocks.0.attn.qkv_proj"], torch.bfloat16)
        self.assertIsNotNone(model.blocks[0].attn.qkv_proj.weight.grad)
        self.assertTrue(torch.isfinite(model.final_layer.video_out.weight.grad).all())

    def test_protected_fsdp_policies_and_mesh(self):
        from torch.distributed.fsdp import MixedPrecisionPolicy

        dit = make_model()
        dit._h3_fp32_modules = True
        model = SimpleNamespace(pipe=SimpleNamespace(dit=dit), to=Mock())
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        plugin = SimpleNamespace(cpu_ram_efficient_loading=False, reshard_after_forward=True,
                                 cpu_offload=None, mixed_precision_policy=policy)
        accelerator = SimpleNamespace(is_fsdp2=True, state=SimpleNamespace(fsdp_plugin=plugin), device="cpu",
                                      torch_device_mesh=None)
        with patch("torch.distributed.fsdp.fully_shard") as shard:
            shard_h3_fp32_modules(model, accelerator)
        self.assertEqual(shard.call_count, len(FP32_MODULES))
        for call, name in zip(shard.call_args_list, FP32_MODULES):
            self.assertIs(call.args[0], dit.get_submodule(name))
            self.assertEqual(call.kwargs["mp_policy"].param_dtype, torch.float32)
            self.assertEqual(call.kwargs["mp_policy"].reduce_dtype, torch.float32)
            self.assertFalse(call.kwargs["mp_policy"].cast_forward_inputs)
        self.assertIs(plugin.mixed_precision_policy, policy)  # Root/block policy remains BF16.

    def test_synthetic_training_entry_both_switches(self):
        entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/train.py"))
        pipeline_class = entry["MiniMaxH3Pipeline"]
        dataset = MiniMaxH3SyntheticDataset(32, 32, 5, 2, use_gradient_checkpointing=True)
        for fp32 in ("0", "1"):
            for zero in ("0", "1"):
                with self.subTest(fp32=fp32, zero_target=zero), patch.dict(os.environ, {
                    "MINIMAX_H3_FP32_MODULES": fp32, "MINIMAX_H3_ZERO_TARGET_LOSS": zero,
                }):
                    pipe = pipeline_class(device="cpu", torch_dtype=torch.bfloat16)
                    pipe.dit = h3.MiniMaxH3DiT(
                        num_layers=1, token_refiner_num_layers=1, hidden_size=64,
                        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=128,
                        timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
                        adaln_out_features=18 * 64, final_adaln_out_features=2 * 64,
                    ).bfloat16()
                    with patch.object(pipeline_class, "from_pretrained", return_value=pipe) as loader:
                        model = entry["MiniMaxH3TrainingModule"](
                            synthetic_inputs=True, trainable_models="dit", task="sft:train",
                        )
                    self.assertIsNone(loader.call_args.kwargs["processor_config"])
                    with patch.object(h3, "attention_forward", F.scaled_dot_product_attention):
                        loss = model({}, inputs=dataset[0])
                        loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(pipe.dit.blocks[0].attn.qkv_proj.weight.grad).all())
                    self.assertTrue(dataset[0][0]["synthetic_inputs"])  # Forward must not mutate the cache.


if __name__ == "__main__":
    unittest.main()
