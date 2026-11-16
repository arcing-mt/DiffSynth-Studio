"""Instance-local MUSA kernels for the standard MiniMax-H3 training model."""

import os
import types

import torch


def _enabled(name):
    value = os.environ.get(name, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def can_use_musa_rope(freqs, dtype):
    return (
        freqs.device.type == "musa"
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not freqs.requires_grad
    )


def rotary_emb_musa(self, x, freqs, rope_cos=None, rope_sin=None):
    if x.device.type != "musa" or not can_use_musa_rope(freqs, x.dtype):
        return type(self).apply_rotary_emb(self, x, freqs, rope_cos, rope_sin)
    if not hasattr(torch, "rope"):
        raise RuntimeError("MINIMAX_H3_MUSA_FUSED_ROPE requires a torch-musa build with torch.rope.")
    rot_dim = freqs.shape[-1]
    # H3 uses half-split partial RoPE (normally 96 of 128 head dimensions).
    # muDNN expects [sequence, batch, heads, rotary_dim] and FP32 angles.
    # It computes cos/sin internally; never reconstruct angles from BF16 cos/sin.
    rotated = torch.rope(
        x[..., :rot_dim].unsqueeze(1).contiguous(),
        freqs.float().contiguous(),
        False, False, False,
    ).squeeze(1)
    return torch.cat((rotated, x[..., rot_dim:]), dim=-1)


def swiglu_musa(self, hidden):
    if hidden.device.type != "musa" or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return type(self).activation(self, hidden)
    if not hasattr(torch.ops.aten, "swish_glu"):
        raise RuntimeError("MINIMAX_H3_MUSA_FUSED_SWIGLU requires a torch-musa build with aten.swish_glu.")
    # The registered autograd formula dispatches the fused SwiGLU backward.
    return torch.ops.aten.swish_glu.default(hidden)


def apply_minimax_h3_musa_patch(model):
    """Read independent env switches and patch only this model's H3 modules.

    Installation also works on CPU-initialized/offloaded models: the actual
    tensor device is checked at execution. Reapplying with disabled switches
    restores the class implementations. No parameters or state-dict keys change.
    """
    if model is None:
        return
    from ...models.minimax_h3_dit import MiniMaxH3Attention, MiniMaxH3DiT, MiniMaxH3MLP

    if type(model) is not MiniMaxH3DiT or getattr(model, "vdn_enabled", False):
        return

    rope = _enabled("MINIMAX_H3_MUSA_FUSED_ROPE")
    swiglu = _enabled("MINIMAX_H3_MUSA_FUSED_SWIGLU")
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            if rope:
                module.apply_rotary_emb = types.MethodType(rotary_emb_musa, module)
            elif getattr(module, "musa_fused_rope", False):
                del module.apply_rotary_emb
            module.musa_fused_rope = rope
        if isinstance(module, MiniMaxH3MLP):
            if swiglu:
                module.activation = types.MethodType(swiglu_musa, module)
            elif getattr(module, "musa_fused_swiglu", False):
                del module.activation
            module.musa_fused_swiglu = swiglu
    model.musa_fused_rope = rope
    model._share_attention_inputs = True
    model.token_refiner._share_attention_inputs = True
    if rope or swiglu:
        print(f"MiniMax-H3 MUSA patch ({type(model).__name__}): fused_rope={rope}, fused_swiglu={swiglu}.")
