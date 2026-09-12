"""Nunchaku MXFP4 implementation of the Diffusers Wan transformer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel
from huggingface_hub import utils
from torch import nn

from ...utils import get_precision_from_quantization_config
from ..linear import SVDQW4A4Linear
from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key, resolve_pretrained_onefile

WAN_QUANTIZED_PROJECTIONS = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)


def _get_child(module: nn.Module, path: str) -> nn.Module:
    child = module
    for part in path.split("."):
        child = child[int(part)] if part.isdigit() else getattr(child, part)
    return child


def _set_child(module: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _convert_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        key = key.replace(".lora_down", ".proj_down").replace(".lora_up", ".proj_up")
        if ".smooth_orig" in key:
            key = key.replace(".smooth_orig", ".smooth_factor_orig")
        elif ".smooth" in key:
            key = key.replace(".smooth", ".smooth_factor")
        converted[key] = value
    return converted


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    """Wan transformer retaining Diffusers attention while using Nunchaku W4A4 linears."""

    def _patch_model(self, *, precision: str, rank: int) -> None:
        for block in self.blocks:
            for path in WAN_QUANTIZED_PROJECTIONS:
                linear = _get_child(block, path)
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"Wan projection {path!r} is not nn.Linear: {type(linear).__name__}")
                quantized = SVDQW4A4Linear.from_linear(linear, precision=precision, rank=rank)
                _set_child(block, path, quantized)

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """Load an AutoRound SVDQuant Nunchaku Wan onefile or component directory."""

        device = torch.device(kwargs.pop("device", "cpu"))
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        offload = kwargs.pop("offload", False)
        if offload:
            raise NotImplementedError("Nunchaku Wan currently supports component placement, not internal offload")

        onefile = resolve_pretrained_onefile(pretrained_model_name_or_path)
        if not onefile.is_file():
            raise ValueError(f"Nunchaku Wan requires a local onefile safetensors checkpoint, got {onefile}")

        transformer, state_dict, metadata = cls._build_model(Path(onefile), torch_dtype=torch_dtype)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = int(quantization_config.get("rank", 32))
        precision = get_precision_from_quantization_config(quantization_config)
        if precision == "fp4":
            precision = "nvfp4"
        if precision != "mxfp4":
            raise ValueError(f"Nunchaku Wan currently expects MXFP4, got {precision!r}")

        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)
        # RoPE tables are non-persistent buffers: to_empty() discards their
        # initialization, and load_state_dict() cannot restore them.
        with torch.device("cpu"):
            rope = type(transformer.rope)(
                attention_head_dim=transformer.config.attention_head_dim,
                patch_size=transformer.config.patch_size,
                max_seq_len=transformer.config.rope_max_seq_len,
            )
        transformer.rope = rope.to(device=device, dtype=torch_dtype)
        state_dict = _convert_state_dict_keys(state_dict)
        # Wan declares numerically sensitive modules in FP32. The generic meta
        # constructor casts everything to torch_dtype, so restore these tensors
        # before load_state_dict copies checkpoint values into their destinations.
        fp32_names = transformer._keep_in_fp32_modules or []
        for name, tensor in [*transformer.named_parameters(), *transformer.named_buffers()]:
            if tensor.is_floating_point() and any(part in name for part in fp32_names):
                tensor.data = tensor.data.to(torch.float32)
                if name in state_dict:
                    state_dict[name] = state_dict[name].to(torch.float32)
        patch_scale_key(transformer, state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, state_dict)
        transformer.load_state_dict(state_dict, strict=True)
        return transformer


__all__ = [
    "NunchakuWanTransformer3DModel",
    "WAN_QUANTIZED_PROJECTIONS",
]
