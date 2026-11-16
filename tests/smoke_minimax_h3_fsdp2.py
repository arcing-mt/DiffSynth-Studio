"""Run with torchrun and an Accelerate FSDP2 environment; accepts H3 CLI options."""

import json
from pathlib import Path
import runpy
import sys
import tempfile

import torch
from torch.distributed.fsdp import FSDPModule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffsynth.diffusion.base_pipeline import BasePipeline
from diffsynth.diffusion.runner import launch_training_task
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from test_minimax_h3_musa_patch import make_inputs


class CountingSGD(torch.optim.SGD):
    updates = 0

    def step(self, closure=None):
        CountingSGD.updates += 1
        return super().step(closure)


class Dataset(torch.utils.data.Dataset):
    load_from_cache = True

    def __init__(self, world_size):
        self.world_size = world_size

    def __len__(self):
        # Five microsteps/rank tests both a full and a partial accumulation window.
        return self.world_size * 5

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(index)
            return make_inputs()


class TinyTrainingModule(DiffusionTrainingModule):
    def __init__(self, dtype, device):
        super().__init__()
        self.pipe = BasePipeline(torch_dtype=dtype, device=device)
        self.pipe.dit = MiniMaxH3DiT(
            num_layers=3, token_refiner_num_layers=1, hidden_size=64,
            num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=128,
            latents_dim=2, audio_latents_dim=2, text_dim=32,
            timestep_input_dim=16, time_embed_hidden_size=64, time_embed_dim=32,
            adaln_out_features=18 * 64, final_adaln_out_features=2 * 64,
        ).to(dtype=dtype)
        self.pipe.dit._share_attention_inputs = True
        self.pipe.dit.token_refiner._share_attention_inputs = True

    def forward(self, data, inputs):
        assert self.pipe.torch_dtype == expected_dtype, (self.pipe.torch_dtype, expected_dtype)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        outputs = self.pipe.dit(**inputs, use_gradient_checkpointing=True)
        assert torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all()
        # Keep this synthetic loss within the initial FP16 GradScaler range.
        return sum(value.float().square().mean() for value in outputs) * 1e-3


class Logger:
    def __init__(self, path):
        self.output_path = path
        self.observations = []

    def on_step_end(self, accelerator, model, save_steps, loss):
        root = model._get_fsdp_state()._fsdp_param_group
        self.observations.append((accelerator.sync_gradients, root.reshard_after_backward, root.reduce_grads,
                                  CountingSGD.updates, root.all_reduce_grads))
        for module in model.modules():
            if isinstance(module, FSDPModule):
                group = module._get_fsdp_state()._fsdp_param_group
                if group is not None:
                    assert group.all_reduce_grads == root.all_reduce_grads

    def on_epoch_end(self, *args):
        pass

    def on_training_end(self, *args):
        pass


if __name__ == "__main__":
    entry = runpy.run_path(str(ROOT / "examples/minimax_h3/model_training/train.py"))
    args = entry["minimax_h3_parser"]().parse_args(["--dataset_base_path", ".", *sys.argv[1:]])
    args.gradient_accumulation_steps = 4
    args.dataset_num_workers = 0
    args.num_epochs = 1
    args.learning_rate = 1e-4
    args.customized_optimizer = "__main__.CountingSGD"
    args.warm_up_steps = 0
    args.output_path = tempfile.mkdtemp(prefix="h3_fsdp2_options_")
    accelerator = entry["minimax_h3_accelerator"](args)
    assert accelerator.is_fsdp2
    options = accelerator.diffsynth_fsdp2_options
    expected_dtype = options.computation_dtype
    reshard_calls = []
    all_reduce_calls = []
    prepared_model = options.prepare_model

    def prepare_and_check(model):
        prepared_model(model)
        original_set_reshard = model.set_reshard_after_backward

        def set_reshard(enabled):
            reshard_calls.append(enabled)
            original_set_reshard(enabled)

        model.set_reshard_after_backward = set_reshard
        original_set_all_reduce = model.set_requires_all_reduce

        def set_all_reduce(enabled):
            all_reduce_calls.append(enabled)
            original_set_all_reduce(enabled)

        model.set_requires_all_reduce = set_all_reduce

        def check_positions(module, positional, kwargs):
            expected = expected_dtype if options.cast_forward_inputs else torch.float64
            assert kwargs["img_position_ids"].dtype == expected

        def check_block_inputs(module, positional, kwargs):
            assert positional[0].dtype == expected_dtype
            expected = expected_dtype if options.cast_forward_inputs else torch.float32
            assert kwargs["rope_freqs"].dtype == expected

        model.pipe.dit.register_forward_pre_hook(check_positions, with_kwargs=True)
        for block in model.pipe.dit.blocks:
            block.register_forward_pre_hook(check_block_inputs, with_kwargs=True)

    options.prepare_model = prepare_and_check
    torch.manual_seed(123)
    model = TinyTrainingModule(expected_dtype, accelerator.device)
    logger = Logger(args.output_path)
    launch_training_task(accelerator, Dataset(accelerator.num_processes), model, logger, args=args)
    assert CountingSGD.updates == 2, logger.observations
    boundaries = [False, False, False, True, True]
    assert [row[0] for row in logger.observations] == boundaries
    assert [row[1] for row in logger.observations] == ([True] * 5 if options.reshard_after_backward else boundaries)
    assert [row[2] for row in logger.observations] == ([True] * 5 if args.fsdp2_sync_each_batch else boundaries)
    assert [row[3] for row in logger.observations] == [0, 0, 0, 1, 2]
    expected_all_reduce = boundaries if options.hsdp_defer_all_reduce else [row[2] for row in logger.observations]
    assert [row[4] for row in logger.observations] == expected_all_reduce
    assert reshard_calls == ([] if options.reshard_after_backward else [False, True]), reshard_calls
    assert all_reduce_calls == ([False, True] if options.hsdp_defer_all_reduce else []), all_reduce_calls
    blocks = list(model.pipe.dit.blocks)
    for index, block in enumerate(blocks):
        assert isinstance(block, FSDPModule)
        state = block._get_fsdp_state()
        if options.forward_prefetch:
            expected = blocks[index + 1:index + 1 + options.forward_prefetch]
            assert state._states_to_forward_prefetch == [m._get_fsdp_state() for m in expected]
        if options.backward_prefetch:
            expected = list(reversed(blocks[max(0, index - options.backward_prefetch):index]))
            assert state._states_to_backward_prefetch == [m._get_fsdp_state() for m in expected]
    if accelerator.parallelism_config is not None:
        config = accelerator.parallelism_config
        mesh = next(model.parameters()).device_mesh
        sizes = (config.dp_replicate_size, config.dp_shard_size)
        if config.dp_replicate_size > 1:
            assert tuple(mesh.shape) == sizes, (mesh, sizes)
    else:
        sizes = (1, accelerator.num_processes)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("PASS FSDP2 runner: " + json.dumps({
            "dtype": str(expected_dtype), "reduce_dtype": str(accelerator.state.fsdp_plugin.mixed_precision_policy.reduce_dtype),
            "mesh": sizes, "microsteps": logger.observations,
            "prefetch": [options.forward_prefetch, options.backward_prefetch],
            "reshard_calls": reshard_calls,
            "all_reduce_calls": all_reduce_calls,
        }), flush=True)
    torch.distributed.destroy_process_group()
