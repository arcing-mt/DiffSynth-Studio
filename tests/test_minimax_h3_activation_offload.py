import copy
import gc
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from diffsynth.core.gradient.minimax_h3_offload import H3ActivationOffload
from diffsynth.diffusion.minimax_h3_training import MiniMaxH3SyntheticDataset
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from test_minimax_h3_musa_patch import make_inputs, make_model, MUSA_AVAILABLE


class H3ActivationOffloadTests(unittest.TestCase):
    def test_training_options(self):
        entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/train.py"))
        parser = entry["minimax_h3_parser"]()
        self.assertEqual(parser.parse_args([]).activation_offload_layers, 0)
        args = parser.parse_args(["--activation_offload_layers", "1", "--activation_gpu_layers", "1",
                                  "--activation_offload_gib", "2"])
        cls = entry["MiniMaxH3Pipeline"]
        pipe = cls(device="cpu", torch_dtype=torch.bfloat16)
        pipe.dit = make_model()
        with patch.object(cls, "from_pretrained", return_value=pipe), \
                patch("diffsynth.core.attention.attention.ATTENTION_IMPLEMENTATION", "torch"):
            module = entry["MiniMaxH3TrainingModule"](
                synthetic_inputs=True, task="sft:train", trainable_models="dit",
                activation_offload_layers=args.activation_offload_layers,
                activation_gpu_layers=args.activation_gpu_layers,
                activation_offload_gib=args.activation_offload_gib,
            )
        policy = module.pipe.dit._activation_offload
        self.assertEqual((policy.cpu_layers, policy.gpu_start, policy.host_limit), (1, 1, 2 * 2**30))
        for cpu, gpu, gib in ((2, 1, 8), (-1, 1, 8), (1, 0, 0), (1, 0, float("nan"))):
            with self.subTest(cpu=cpu, gpu=gpu, gib=gib), self.assertRaises(ValueError):
                H3ActivationOffload(2, cpu, gpu, gib)

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_training_loss_with_cached_inputs(self):
        entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/train.py"))
        device = torch.device("musa", torch.musa.current_device())
        cls = entry["MiniMaxH3Pipeline"]
        pipe = cls(device=device, torch_dtype=torch.bfloat16)
        pipe.dit = MiniMaxH3DiT(
            num_layers=2, token_refiner_num_layers=1, hidden_size=64, num_attention_heads=2,
            ffn_hidden_size=128, timestep_input_dim=16, time_embed_hidden_size=64,
            time_embed_dim=32, adaln_out_features=18 * 64, final_adaln_out_features=2 * 64,
        ).to(device, torch.bfloat16)
        # Simulate cached inputs generated with checkpointing disabled.
        inputs = MiniMaxH3SyntheticDataset(32, 32, 5, 1, device=device)[0]
        with patch.object(cls, "from_pretrained", return_value=pipe), \
                patch("diffsynth.core.attention.attention.ATTENTION_IMPLEMENTATION", "torch"), \
                patch.dict("os.environ", {"MINIMAX_H3_ZERO_TARGET_LOSS": "0"}):
            model = entry["MiniMaxH3TrainingModule"](
                device=device, synthetic_inputs=True, trainable_models="dit", task="sft:train",
                activation_offload_layers=1, activation_gpu_layers=1,
            )
            loss = model({}, inputs=inputs)
            self.assertGreater(pipe.dit._activation_offload.host_bytes, 0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(pipe.dit.blocks[0].attn.qkv_proj.weight.grad).all())
            self.assertFalse(inputs[0]["use_gradient_checkpointing"])

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_packed_segments_retained_and_interleaved_graphs(self):
        device, dtype = torch.device("musa", torch.musa.current_device()), torch.bfloat16
        torch.manual_seed(401)
        reference = make_model().to(device, dtype)
        model = copy.deepcopy(reference)
        policy = model._activation_offload = H3ActivationOffload(2, 1, 1, 1)
        batches = [make_inputs(device, dtype) for _ in range(2)]
        def forward(net, i):
            return net(**batches[i], use_gradient_checkpointing=True)

        def gradients(outputs, net, retain=False):
            loss = sum(o.float().square().mean() for o in outputs)
            return torch.autograd.grad(loss, tuple(net.parameters()), allow_unused=True, retain_graph=retain)

        with patch("diffsynth.core.attention.attention.ATTENTION_IMPLEMENTATION", "torch"):
            expected = [forward(reference, i) for i in range(2)]
            actual = [forward(model, i) for i in range(2)]
            # Two outstanding graphs, two nonempty segments containing 8 tokens in total.
            self.assertEqual(policy.host_bytes, 2 * 8 * 2 * (128 * 2 + 4))
            for i in (1, 0):
                torch.testing.assert_close(actual[i], expected[i], atol=0, rtol=0)
                ref_grads = gradients(expected[i], reference)
                for retain in (True, False):
                    grads = gradients(actual[i], model, retain)
                    for ref, grad in zip(ref_grads, grads):
                        if ref is None:
                            self.assertIsNone(grad)
                        else:
                            torch.testing.assert_close(ref.float(), grad.float(), rtol=0.015, atol=5e-5)
            del actual
            gc.collect()
            self.assertEqual(policy.host_bytes, 0)

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_excess_segments_recompute_and_abandoned_graph_releases_budget(self):
        device, dtype = torch.device("musa", torch.musa.current_device()), torch.bfloat16
        model = make_model().to(device, dtype)
        # Only the length-2 segment fits. The length-6 segment must replay.
        policy = model._activation_offload = H3ActivationOffload(2, 1, 0, 1100 / 2**30)
        inputs = make_inputs(device, dtype)
        with patch("diffsynth.core.attention.attention.ATTENTION_IMPLEMENTATION", "torch"), \
                self.assertWarnsRegex(UserWarning, "limit reached"):
            out = model(**inputs, use_gradient_checkpointing=True)
            self.assertEqual(policy.host_bytes, 1040)
            del out
            gc.collect()
            self.assertEqual(policy.host_bytes, 0)
            out = model(**inputs, use_gradient_checkpointing=True)
            sum(t.float().sum() for t in out).backward()
            self.assertTrue(torch.isfinite(model.blocks[0].attn.qkv_proj.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
