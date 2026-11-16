import copy
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from diffsynth.core.musa_patch import apply_minimax_h3_musa_patch
from diffsynth.models import minimax_h3_dit as h3


ROPE = "MINIMAX_H3_MUSA_FUSED_ROPE"
SWIGLU = "MINIMAX_H3_MUSA_FUSED_SWIGLU"
MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()


def make_model():
    model = h3.MiniMaxH3DiT(
        num_layers=2, token_refiner_num_layers=2, hidden_size=64,
        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=128,
        latents_dim=2, audio_latents_dim=2, text_dim=32,
        timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
        adaln_out_features=18 * 64, final_adaln_out_features=2 * 64,
    )
    apply_minimax_h3_musa_patch(model)
    return model


def make_inputs(device="cpu", dtype=torch.float32):
    def index(values):
        return torch.tensor(values, device=device)

    return dict(
        x=torch.randn(1, 8, 8, device=device, dtype=dtype),
        audio_x=torch.randn(1, 8, 2, device=device, dtype=dtype),
        img_position_ids=torch.randn(1, 8, 3, device=device, dtype=torch.float64) * 500,
        unique_timesteps=torch.tensor([0.25, 0.5], device=device),
        inverse_indices=index([0, 0, 1, 1, 0, 0, 1, 1]),
        update_mask=index([1, 0, 1, 1]), update_audio_mask=index([1, 0]),
        token_tags=index([2, 2, 0, 0, 0, 0, 1, 1]),
        prompt_embeds=torch.randn(2, 32, device=device, dtype=dtype),
        img_pos_info={"position_ids": index([2, 3, 4, 5])},
        audio_pos_info={"position_ids": index([6, 7])},
        text_pos_info={"position_ids": index([0, 1])},
        img_pos_for_infer_output_info={"position_ids": index([2, 3, 4, 5])},
        # Includes an empty segment and a second non-empty segment.
        packed_seq_params={"cu_seqlens_q": index([0, 6, 6, 8]), "max_seqlen_q": 6},
        refiner_packed_seq_params={"cu_seqlens_q": index([0, 2]), "max_seqlen_q": 2},
    )


def reference_rope(self, x, freqs, rope_cos=None, rope_sin=None):
    """Original H3 expression: recompute cos/sin for every Q/K invocation."""
    dim = freqs.shape[-1]
    first, second = x[..., :dim].chunk(2, dim=-1)
    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    rotated = x[..., :dim] * cos + torch.cat((-second, first), -1) * sin
    return torch.cat((rotated, x[..., dim:]), -1)


def run_backward(model, inputs, **kwargs):
    output = model(**inputs, **kwargs)
    sum(value.float().square().mean() for value in output).backward()
    return output, {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


class H3TestCase(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {ROPE: "0", SWIGLU: "0"})
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(42)


class H3PatchTests(H3TestCase):
    def test_shared_values_preserve_outputs_and_gradients_with_checkpointing(self):
        for checkpoint, offload in ((False, False), (True, False), (True, True)):
            with self.subTest(checkpoint=checkpoint, offload=offload):
                model = make_model()
                reference = copy.deepcopy(model)
                inputs = make_inputs()
                with patch.object(h3, "attention_forward", F.scaled_dot_product_attention):
                    with patch.object(h3.MiniMaxH3Attention, "apply_rotary_emb", reference_rope):
                        expected, expected_grads = run_backward(reference, inputs)
                    tensor_bounds = []
                    original_bounds = h3._seqlen_bounds

                    def bounds(value):
                        if isinstance(value, torch.Tensor):
                            tensor_bounds.append(value)
                        return original_bounds(value)

                    with patch.object(h3, "_seqlen_bounds", side_effect=bounds):
                        with patch.object(h3, "_prepare_rope", wraps=h3._prepare_rope) as prepare:
                            actual, actual_grads = run_backward(
                                model, inputs, use_gradient_checkpointing=checkpoint,
                                use_gradient_checkpointing_offload=offload,
                            )
                    self.assertEqual(prepare.call_count, 1)
                    self.assertEqual(len(tensor_bounds), 2)  # DiT + token refiner, even on recompute.
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)

    def test_switches_are_independent_reversible_and_instance_local(self):
        model, other = make_model(), make_model()
        keys = set(model.state_dict())
        for rope, swiglu in ((1, 0), (0, 1), (1, 1), (0, 0)):
            with patch.dict(os.environ, {ROPE: str(rope), SWIGLU: str(swiglu)}):
                apply_minimax_h3_musa_patch(model)
            for block in list(model.blocks) + list(model.token_refiner.blocks):
                self.assertEqual("apply_rotary_emb" in block.attn.__dict__, bool(rope))
                self.assertEqual("activation" in block.mlp.__dict__, bool(swiglu))
            self.assertNotIn("apply_rotary_emb", other.blocks[0].attn.__dict__)
            self.assertEqual(set(model.state_dict()), keys)
            model.load_state_dict(other.state_dict(), strict=True)

    def test_cpu_fallback_with_both_switches_enabled(self):
        model = make_model()
        original = copy.deepcopy(model)
        with patch.dict(os.environ, {ROPE: "1", SWIGLU: "1"}):
            apply_minimax_h3_musa_patch(model)
        inputs = make_inputs()
        with patch.object(h3, "attention_forward", F.scaled_dot_product_attention):
            expected = run_backward(original, inputs)
            with patch("torch.rope", side_effect=AssertionError("CPU called a MUSA kernel"), create=True):
                actual = run_backward(model, inputs, use_gradient_checkpointing=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_entry_installs_patch_after_loading(self):
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline

        script = Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/train.py"
        training_module = runpy.run_path(str(script))["MiniMaxH3TrainingModule"]
        model = make_model()
        pipe = MiniMaxH3Pipeline(device="cpu")
        pipe.dit = model
        with patch.dict(os.environ, {ROPE: "1", SWIGLU: "1"}):
            with patch.object(MiniMaxH3Pipeline, "from_pretrained", return_value=pipe):
                training = training_module(trainable_models="dit", device="cpu")
        self.assertIs(training.pipe.dit, model)
        self.assertTrue(model.blocks[0].attn.musa_fused_rope)
        self.assertTrue(model.token_refiner.blocks[0].mlp.musa_fused_swiglu)


@unittest.skipUnless(MUSA_AVAILABLE, "requires a MUSA device")
class H3MUSAKernelTests(H3TestCase):
    def assert_relative_error(self, actual, expected, limit):
        delta = (actual.float() - expected.float()).norm()
        scale = expected.float().norm().clamp_min(1e-8)
        self.assertLessEqual((delta / scale).item(), limit)

    def test_native_kernels_and_partial_rope_backward(self):
        model = make_model().to(device="musa", dtype=torch.bfloat16)
        with patch.dict(os.environ, {ROPE: "1", SWIGLU: "1"}):
            apply_minimax_h3_musa_patch(model)
        attn, mlp = model.blocks[0].attn, model.blocks[0].mlp
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(16, 2, 128, device="musa", dtype=dtype, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            freqs = h3.MiniMaxH3Rope(16).to("musa")(torch.randn(1, 16, 3, device="musa") * 500)
            expected = reference_rope(attn, reference, freqs)
            with patch("torch.cos", side_effect=AssertionError("fused RoPE prepared cos")):
                with patch("torch.sin", side_effect=AssertionError("fused RoPE prepared sin")):
                    self.assertEqual(h3._prepare_rope(freqs, dtype, True), (None, None))
                    actual = attn.apply_rotary_emb(x, freqs)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            limit = 1e-5 if dtype == torch.float32 else 0.01
            self.assert_relative_error(actual, expected, limit)
            self.assert_relative_error(x.grad, reference.grad, limit)
            torch.testing.assert_close(actual[..., 96:], x[..., 96:], rtol=0, atol=0)
            torch.testing.assert_close(x.grad[..., 96:], gradient[..., 96:], rtol=0, atol=0)

            hidden = torch.randn(16, 256, device="musa", dtype=dtype, requires_grad=True)
            reference = hidden.detach().clone().requires_grad_()
            expected = h3.MiniMaxH3MLP.activation(mlp, reference)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                actual = mlp.activation(hidden)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
            expected.backward(gradient)
            self.assert_relative_error(actual, expected, limit)
            self.assert_relative_error(hidden.grad, reference.grad, limit)
            operations = {event.key for event in prof.key_averages()}
            self.assertIn("aten::swish_glu", operations)
            self.assertTrue(any("swiglu" in name.lower() and "backward" in name.lower() for name in operations))

    def test_both_switches_full_dit_forward_backward_and_checkpoint(self):
        original = make_model().to(device="musa", dtype=torch.bfloat16)
        inputs = make_inputs("musa", torch.bfloat16)
        expected, expected_grads = run_backward(original, inputs)
        for rope, swiglu in ((1, 0), (0, 1), (1, 1)):
            for checkpoint in (False, True):
                with self.subTest(rope=rope, swiglu=swiglu, checkpoint=checkpoint):
                    model = make_model().to(device="musa", dtype=torch.bfloat16)
                    model.load_state_dict(original.state_dict())
                    with patch.dict(os.environ, {ROPE: str(rope), SWIGLU: str(swiglu)}):
                        apply_minimax_h3_musa_patch(model)
                    actual, actual_grads = run_backward(model, inputs, use_gradient_checkpointing=checkpoint)
                    for a, b in zip(actual, expected):
                        self.assert_relative_error(a, b, 0.03)
                    self.assertEqual(actual_grads.keys(), expected_grads.keys())
                    for name, grad in actual_grads.items():
                        with self.subTest(parameter=name):
                            self.assert_relative_error(grad, expected_grads[name], 0.05)


if __name__ == "__main__":
    unittest.main()
