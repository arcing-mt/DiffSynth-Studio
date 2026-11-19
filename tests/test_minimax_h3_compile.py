import copy
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from diffsynth.diffusion.minimax_h3_training import configure_h3_compile, configure_h3_fp32_modules
from diffsynth.models import minimax_h3_dit as h3
from test_minimax_h3_musa_patch import make_model, make_inputs


class H3CompileTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "MINIMAX_H3_MUSA_FUSED_ROPE": "0", "MINIMAX_H3_MUSA_FUSED_SWIGLU": "0",
            "MINIMAX_H3_FP32_MODULES": "0",
        })
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(42)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_disabled_has_no_compiler_or_model_changes(self):
        model = make_model()
        before = {name: module.forward for name, module in model.named_modules()}
        with patch("torch.compile", side_effect=AssertionError("compiled while disabled")), \
                patch("torch.compiler.disable", side_effect=AssertionError("disabled eager function")):
            self.assertEqual(configure_h3_compile(model, safe_norm=True, safe_attention=True), 0)
        self.assertEqual(before, {name: module.forward for name, module in model.named_modules()})
        self.assertNotIn("modulate", vars(model.final_layer))

    def test_disabled_path_matches_precompile_forward_and_backward(self):
        current = make_model()
        baseline = copy.deepcopy(current)
        inputs = make_inputs()

        def precompile_attention(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None, rope_cos=None, rope_sin=None):
            total = x.shape[0]
            qkv = self.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim)
            q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
            q, k = self.q_norm(q), self.k_norm(k)
            if rope_freqs is not None:
                q = self.apply_rotary_emb(q, rope_freqs, rope_cos, rope_sin)
                k = self.apply_rotary_emb(k, rope_freqs, rope_cos, rope_sin)
            out = h3._sdpa_varlen_attention(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
            return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))

        def precompile_final(self, x, *, t_emb, inverse_indices):
            shift, scale = self.adaln_proj(t_emb)
            h = self.norm(x)
            h = h * (1.0 + scale.index_select(0, inverse_indices)) + shift.index_select(0, inverse_indices)
            return self.video_out(h), self.audio_out(h)

        with patch.object(h3, "attention_forward", F.scaled_dot_product_attention), \
                patch.object(h3.MiniMaxH3Attention, "forward", precompile_attention), \
                patch.object(h3.MiniMaxH3FinalLayer, "forward", precompile_final):
            expected, expected_grads = self._run_backward(baseline, inputs)
        with patch.object(h3, "attention_forward", F.scaled_dot_product_attention):
            actual, actual_grads = self._run_backward(current, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)

    @staticmethod
    def _run_backward(model, inputs):
        outputs = model(**inputs)
        loss = sum(value.float().square().mean() for value in outputs)
        loss.backward()
        return outputs, {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

    def test_regions_preserve_keys_parameters_and_instance_isolation(self):
        model, untouched = make_model(), make_model()
        parameters = dict(model.named_parameters())
        keys = list(model.state_dict())
        dispatcher = h3.attention_forward
        sdpa = F.scaled_dot_product_attention
        blocks = [*model.token_refiner.blocks, *model.blocks]
        targets = [block for block in blocks]
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as compile_fn:
            self.assertEqual(configure_h3_compile(model, True, True, True), len(targets) + 1)
            self.assertEqual(compile_fn.call_count, len(targets) + 1)
            self.assertTrue(all(call.kwargs == {"dynamic": False} for call in compile_fn.call_args_list))
            self.assertEqual(configure_h3_compile(model, True, True, True), 0)
            self.assertEqual(compile_fn.call_count, len(targets) + 1)
        self.assertEqual(list(model.state_dict()), keys)
        self.assertTrue(all(p is dict(model.named_parameters())[name] for name, p in parameters.items()))
        self.assertIs(h3.attention_forward, dispatcher)
        self.assertIs(F.scaled_dot_product_attention, sdpa)
        for module in untouched.modules():
            self.assertNotIn("forward", vars(module))
            self.assertNotIn("compute_attention", vars(module))
        for module in model.modules():
            if isinstance(module, torch.nn.RMSNorm):
                self.assertTrue(getattr(module.forward, "_torchdynamo_disable", False))
            if isinstance(module, h3.MiniMaxH3Attention):
                self.assertTrue(getattr(module.compute_attention, "_torchdynamo_disable", False))
        self.assertNotIn("forward", vars(model.final_layer.video_out))
        self.assertNotIn("forward", vars(model.final_layer.audio_out))
        self.assertNotIn("forward", vars(model))

    def test_compiled_forward_backward_accumulation_and_fp32_modules(self):
        # AOTAutograd actually traces/compiles both forward and backward on CPU.
        # Counting graphs also prevents an unnoticed all-eager fallback.
        from torch._dynamo.backends.registry import lookup_backend
        aot_eager = lookup_backend("aot_eager")
        original_compile = torch.compile
        for dtype, fp32 in ((torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)):
            with self.subTest(dtype=dtype, fp32_modules=fp32):
                torch._dynamo.reset()
                eager = make_model().to(dtype)
                if fp32:
                    with patch.dict(os.environ, {"MINIMAX_H3_FP32_MODULES": "1"}):
                        configure_h3_fp32_modules(eager)
                compiled = copy.deepcopy(eager)
                graphs = []
                def backend(graph, inputs):
                    graphs.append(graph)
                    return aot_eager(graph, inputs)
                def compile_fn(fn, **kwargs):
                    return original_compile(fn, backend=backend, **kwargs)
                with patch("torch.compile", side_effect=compile_fn):
                    configure_h3_compile(compiled, True, True, True)
                batches = [make_inputs(dtype=eager.condition_proj.weight.dtype) for _ in range(2)]
                optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in (eager, compiled)]
                with patch.object(h3, "attention_forward", F.scaled_dot_product_attention):
                    for checkpoint in (False, True):
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        for inputs in batches:
                            outputs = []
                            for model in (eager, compiled):
                                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                                    out = model(**inputs, use_gradient_checkpointing=checkpoint)
                                    loss = sum(value.float().square().mean() for value in out) / 2
                                loss.backward()
                                outputs.append(tuple(value.detach() for value in out))
                            torch.testing.assert_close(outputs[0], outputs[1])
                        for (name, p), (_, actual) in zip(eager.named_parameters(), compiled.named_parameters()):
                            self.assertEqual(p.grad is None, actual.grad is None, name)
                            if p.grad is not None:
                                if dtype == torch.bfloat16:
                                    # AOTAutograd can reorder BF16 gradient sums,
                                    # including gradients of protected FP32 weights.
                                    expected_grad, actual_grad = p.grad.float(), actual.grad.float()
                                    relative_error = (expected_grad - actual_grad).norm() / expected_grad.norm().clamp_min(1e-8)
                                    self.assertLess(relative_error.item(), 0.02, name)
                                else:
                                    torch.testing.assert_close(p.grad, actual.grad)
                        for optimizer in optimizers:
                            optimizer.step()
                        torch.testing.assert_close(eager.state_dict(), compiled.state_dict())
                self.assertGreater(len(graphs), 0)

    def test_training_entry_passes_compile_options(self):
        entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/train.py"))
        parser = entry["minimax_h3_parser"]()
        args = parser.parse_args([])
        self.assertFalse(args.compile_blocks)
        self.assertFalse(args.compile_safe_norm)
        self.assertFalse(args.compile_safe_attention)
        args = parser.parse_args(["--compile_blocks", "--compile_safe_norm", "--compile_safe_attention"])
        pipeline_class = entry["MiniMaxH3Pipeline"]
        pipe = pipeline_class(device="cpu", torch_dtype=torch.float32)
        pipe.dit = make_model()
        with patch.object(pipeline_class, "from_pretrained", return_value=pipe), \
                patch("torch.compile", side_effect=lambda fn, **kwargs: fn) as compile_fn:
            entry["MiniMaxH3TrainingModule"](
                synthetic_inputs=True, trainable_models="dit", task="sft:train",
                compile_blocks=args.compile_blocks, compile_safe_norm=args.compile_safe_norm,
                compile_safe_attention=args.compile_safe_attention,
            )
        self.assertEqual(compile_fn.call_count, 5)


if __name__ == "__main__":
    unittest.main()
