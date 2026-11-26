import copy
import gc
import runpy
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from diffsynth.core.gradient.minimax_h3_offload import H3ActivationOffload
from diffsynth.diffusion.minimax_h3_training import MiniMaxH3SyntheticDataset
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from test_minimax_h3_musa_patch import make_inputs, make_model, MUSA_AVAILABLE


class H3ActivationOffloadTests(unittest.TestCase):
    def allocation(self, ptr):
        for segment in torch.musa.memory_snapshot():
            for block in segment["blocks"]:
                if block["address"] <= ptr < block["address"] + block["size"]:
                    return segment["stream"], block["state"]
        self.fail(f"Allocation {ptr} missing from snapshot")

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_bidirectional_allocation_ownership_and_reuse(self):
        device = torch.device("musa", torch.musa.current_device())
        for producer in (torch.musa.current_stream(device), torch.musa.Stream(device=device)):
            with self.subTest(producer=producer), torch.musa.stream(producer):
                policy = H3ActivationOffload(2, 2, 0, 1)
                output = torch.full((1024, 1024), 7, device=device, dtype=torch.bfloat16).T
                lse = torch.full((1024,), 8.0, device=device)
                source_ptr = output.data_ptr()
                entry = policy.save_to_host((output, lse, torch.empty(0, device=device)))
                del output, lse
                self.assertEqual(self.allocation(source_ptr)[1], "active_allocated")
                entry.finish_forward()
                self.assertEqual(self.allocation(source_ptr)[1], "inactive")
                entry.prefetch()
                destination = entry.prefetched.tensors[0].data_ptr()
                self.assertEqual(self.allocation(destination)[0], producer.musa_stream)
                d2h, h2d = policy.streams[device]
                self.assertNotEqual(d2h, producer)
                self.assertNotEqual(h2d, producer)
                restored = entry.restore()
                # Leave the last consumer queued while releasing the tensors.
                # Safe immediate cache reuse depends on allocation ownership.
                torch.musa._sleep(20_000_000)
                actual = restored[0].clone()
                actual_lse = restored[1].clone()
                del restored
                self.assertEqual(self.allocation(destination)[1], "inactive")
                torch.empty((1024, 1024), device=device, dtype=torch.bfloat16).fill_(-1)
                torch.testing.assert_close(actual, torch.full_like(actual, 7), rtol=0, atol=0)
                torch.testing.assert_close(actual_lse, torch.full_like(actual_lse, 8), rtol=0, atol=0)

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_h2d_waits_for_previous_allocation_use(self):
        device = torch.device("musa", torch.musa.current_device())
        producer = torch.musa.Stream(device=device)
        policy = H3ActivationOffload(2, 2, 0, 1)
        with torch.musa.stream(producer):
            entry = policy.save_to_host((torch.full((256, 256), 7.0, device=device),
                                         torch.full((256,), 8.0, device=device),
                                         torch.empty(0, device=device)))
            entry.finish_forward()
            reused = [torch.empty(256 * 256, device=device), torch.empty(256, device=device)]
            torch.musa._sleep(50_000_000)
            for tensor in reused:
                tensor.fill_(-1)
            # Emulate the caching allocator returning storage whose previous
            # owner-stream use is still queued. DMA must wait before overwrite.
            with patch.object(torch, "empty_like", side_effect=reused):
                entry.prefetch()
            result = entry.restore()
            torch.testing.assert_close(result[0], torch.full_like(result[0], 7.0), rtol=0, atol=0)
            torch.testing.assert_close(result[1], torch.full_like(result[1], 8.0), rtol=0, atol=0)

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_prefetch_drop_and_changed_consumer_stream(self):
        device = torch.device("musa", torch.musa.current_device())
        producer, consumer = [torch.musa.Stream(device=device) for _ in range(2)]
        policy = H3ActivationOffload(2, 2, 0, 1)
        with torch.musa.stream(producer):
            entry = policy.save_to_host((torch.ones((256, 256), device=device),
                                         torch.ones(256, device=device), torch.empty(0, device=device)))
            entry.finish_forward()
            entry.prefetch()
            old_ptr = entry.prefetched.tensors[0].data_ptr()
        with torch.musa.stream(consumer):
            restored = entry.restore()
            self.assertEqual(self.allocation(restored[0].data_ptr())[0], consumer.musa_stream)
            self.assertEqual(self.allocation(old_ptr)[1], "inactive")
            torch.testing.assert_close(restored[0], torch.ones_like(restored[0]), rtol=0, atol=0)
        entry.prefetch()
        unused_ptr = entry.prefetched.tensors[0].data_ptr()
        del entry
        gc.collect()
        self.assertEqual(self.allocation(unused_ptr)[1], "inactive")
        self.assertEqual(policy.host_bytes, 0)

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_partial_copy_failure_both_directions(self):
        device = torch.device("musa", torch.musa.current_device())
        producer = torch.musa.Stream(device=device)
        real_copy = torch.Tensor.copy_
        for direction in ("d2h", "h2d"):
            with self.subTest(direction=direction), torch.musa.stream(producer):
                policy = H3ActivationOffload(2, 2, 0, 1)
                output = torch.full((256, 256), 7.0, device=device)
                lse = torch.full((256,), 8.0, device=device)
                tensors = (output, lse, torch.empty(0, device=device))
                entry = None
                if direction == "h2d":
                    entry = policy.save_to_host(tensors)
                    entry.finish_forward()
                copies = []

                def copy_then_fail(dst, src, **kwargs):
                    result = real_copy(dst, src, **kwargs)
                    copies.append(dst)
                    if len(copies) == 2:
                        raise RuntimeError("injected copy failure")
                    return result

                with patch.object(torch.Tensor, "copy_", copy_then_fail), \
                        self.assertRaisesRegex(RuntimeError, "injected copy failure"):
                    if entry is None:
                        policy.save_to_host(tensors)
                    else:
                        entry.prefetch()
                output.fill_(-1)
                lse.fill_(-1)
                producer.synchronize()
                torch.testing.assert_close(copies[0], torch.full_like(copies[0], 7.0), rtol=0, atol=0)
                torch.testing.assert_close(copies[1], torch.full_like(copies[1], 8.0), rtol=0, atol=0)
                if entry is not None:
                    self.assertIsNone(entry.prefetched)
                del entry
                gc.collect()
                self.assertEqual(policy.host_bytes, 0)

    @unittest.skipUnless(MUSA_AVAILABLE, "requires native MUSA Flash SDPA")
    def test_block_cleanup_partial_gradients_and_backward_failure(self):
        device, dtype = torch.device("musa", torch.musa.current_device()), torch.bfloat16
        producer = torch.musa.Stream(device=device)
        with torch.musa.stream(producer):
            model = make_model().to(device, dtype)
            reference = copy.deepcopy(model)
            policy = model._activation_offload = H3ActivationOffload(2, 2, 0, 1)
            inputs = make_inputs(device, dtype)
            entries = []
            save = policy.save_to_host

            def capture(tensors):
                entry = save(tensors)
                entries.append(weakref.ref(entry))
                return entry

            def loss(net):
                return sum(t.float().square().mean() for t in net(**inputs, use_gradient_checkpointing=True))

            replay_checks = []

            def check_first_restore(module, args):
                # Last block's original forward has only the first block's two
                # entries. On replay all four exist, and the first backward
                # restore must already be prefetched before replay's qkv/MLP.
                if len(entries) == 4:
                    replay_checks.append(True)
                    self.assertTrue(all(ref().prefetched is not None for ref in entries[2:]))

            bootstrap_hook = model.blocks[1].register_forward_pre_hook(check_first_restore)
            with patch.object(policy, "save_to_host", capture), \
                    patch("diffsynth.core.attention.attention.ATTENTION_IMPLEMENTATION", "torch"):
                expected, actual = loss(reference), loss(model)
                self.assertTrue(all(not ref().source.tensors for ref in entries))
                params = [model.blocks[1].mlp.fc2.weight]
                ref_params = [reference.blocks[1].mlp.fc2.weight]
                ref_grads = torch.autograd.grad(expected, ref_params)
                grads = torch.autograd.grad(actual, params, retain_graph=True)
                self.assertTrue(replay_checks)
                torch.testing.assert_close(grads[0], ref_grads[0], rtol=0.015, atol=5e-5)
                self.assertTrue(all(ref().prefetched is None for ref in entries))

                def fail(*args):
                    raise RuntimeError("injected backward failure")

                hook = model.blocks[1].mlp.fc2.weight.register_hook(fail)
                try:
                    with self.assertRaisesRegex(RuntimeError, "injected backward failure"):
                        actual.backward(retain_graph=True)
                finally:
                    hook.remove()
                bootstrap_hook.remove()
                gc.collect()
                self.assertTrue(all(ref().prefetched is None for ref in entries))

                # Checkpoint forward failures must retire sources too, even if
                # Python retains the exception traceback and its graph objects.
                entries.clear()
                hook = model.blocks[0].register_forward_hook(fail)
                try:
                    try:
                        loss(model)
                    except RuntimeError as error:
                        self.assertIn("injected backward failure", str(error))
                        self.assertTrue(entries)
                        self.assertTrue(all(ref() is None or not ref().source.tensors for ref in entries))
                    else:
                        self.fail("forward hook should raise")
                finally:
                    hook.remove()

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
