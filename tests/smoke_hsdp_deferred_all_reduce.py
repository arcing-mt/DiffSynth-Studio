"""Distributed HSDP communication counts and numerical gradient/update reference.

Run with torchrun, ACCELERATE_USE_FSDP=true, and --fsdp2_dp_replicate_size 2
--fsdp2_dp_shard_size 2. Compare the three valid sync/deferral combinations.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsynth.diffusion.fsdp2 import add_fsdp2_training_options, configure_fsdp2_training, fsdp2_accelerator_kwargs


def sample(index):
    # Binary fractions keep numerical comparisons sensitive to scaling errors.
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8 + index / 32
    return x, torch.full_like(x, -index / 64)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, world):
        self.world = world

    def __len__(self):
        return self.world * 5

    def __getitem__(self, index):
        return sample(index)


if __name__ == "__main__":
    args = add_fsdp2_training_options(argparse.ArgumentParser()).parse_args()
    kwargs = fsdp2_accelerator_kwargs(args)
    kwargs.setdefault("mixed_precision", "no")
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        fsdp_plugin=FullyShardedDataParallelPlugin(fsdp_version=2, auto_wrap_policy="no_wrap", cpu_ram_efficient_loading=False),
        **kwargs,
    )
    configure_fsdp2_training(accelerator, args)
    options = accelerator.diffsynth_fsdp2_options
    assert options.computation_dtype == torch.float32
    assert accelerator.parallelism_config.dp_replicate_size > 1

    reference = (torch.arange(16, dtype=torch.float32).reshape(4, 4) - 8) / 16
    model = torch.nn.Linear(4, 4, bias=False).to(accelerator.device)
    with torch.no_grad():
        model.weight.copy_(reference)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
    dataloader = torch.utils.data.DataLoader(Dataset(accelerator.num_processes), collate_fn=lambda batch: batch[0])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    counts = Counter(reduce_scatter=0, all_reduce=0)
    observations = []
    reference_grad = torch.zeros_like(reference)
    numerical_errors = []
    replica_group = None

    class CountReduceScatter:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def __call__(self, *args, **kwargs):
            counts["reduce_scatter"] += 1
            return self.original(*args, **kwargs)

    def install_counters(module, inputs):
        global replica_group
        # Backends can select a new communication object in each pre-forward.
        group = module._get_fsdp_state()._fsdp_param_group
        assert group._is_hsdp
        replica_group = group._all_reduce_process_group
        if not isinstance(group._reduce_scatter_comm, CountReduceScatter):
            group._reduce_scatter_comm = CountReduceScatter(group._reduce_scatter_comm)

    model.register_forward_pre_hook(install_counters)
    original_all_reduce = dist.all_reduce

    def count_all_reduce(tensor, *args, **kwargs):
        if kwargs.get("group") is replica_group and replica_group is not None:
            counts["all_reduce"] += 1
        return original_all_reduce(tensor, *args, **kwargs)

    with patch.object(dist, "all_reduce", side_effect=count_all_reduce), options.training(model):
        for step, (x, target) in enumerate(dataloader):
            for rank in range(accelerator.num_processes):
                ref_x, ref_target = sample(step * accelerator.num_processes + rank)
                diff = ref_x @ reference.T - ref_target
                reference_grad += (2 / diff.numel()) * diff.T @ ref_x / (accelerator.num_processes * 4)
            previous = counts.copy()
            with accelerator.accumulate(model), options.microstep(accelerator, model):
                loss = torch.nn.functional.mse_loss(model(x), target)
                accelerator.backward(loss)
                delta = counts - previous
                expected_sync = bool(args.fsdp2_sync_each_batch) or accelerator.sync_gradients
                expected_all_reduce = accelerator.sync_gradients if options.hsdp_defer_all_reduce else expected_sync
                assert delta["reduce_scatter"] == int(expected_sync), (step, counts)
                assert delta["all_reduce"] == int(expected_all_reduce), (step, counts)
                observations.append(dict(microstep=step + 1, reduce_scatter=delta["reduce_scatter"],
                                         all_reduce=delta["all_reduce"], update=accelerator.sync_gradients))
                if accelerator.sync_gradients:
                    actual_grad = model.weight.grad.full_tensor().detach().cpu()
                    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-6)
                    numerical_errors.append(float((actual_grad - reference_grad).abs().max()))
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    reference -= 0.125 * reference_grad
                    reference_grad.zero_()
                    actual_weight = model.weight.full_tensor().detach().cpu()
                    torch.testing.assert_close(actual_weight, reference, rtol=1e-5, atol=1e-6)
    assert len(observations) == 5 and len(numerical_errors) == 2
    assert model._get_fsdp_state()._fsdp_param_group.all_reduce_grads
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("PASS HSDP communication and numerics: " + json.dumps({
            "sync_each_batch": bool(args.fsdp2_sync_each_batch),
            "defer_all_reduce": options.hsdp_defer_all_reduce,
            "steps": observations, "gradient_max_abs_errors": numerical_errors,
        }), flush=True)
    dist.destroy_process_group()
