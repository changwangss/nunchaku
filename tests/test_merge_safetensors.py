import json

import torch
from safetensors.torch import save_file

from nunchaku.merge_safetensors import merge_safetensors


def test_merge_safetensors_detects_mxfp4_metadata(tmp_path):
    save_file(
        {
            "transformer_blocks.0.qkv_proj.qweight": torch.zeros((4, 2), dtype=torch.int8),
            "transformer_blocks.0.qkv_proj.wscales": torch.zeros((1, 4), dtype=torch.uint8),
            "transformer_blocks.0.qkv_proj.lora_down": torch.zeros((4, 32), dtype=torch.bfloat16),
        },
        tmp_path / "transformer_blocks.safetensors",
    )
    save_file({"x_embedder.weight": torch.zeros((1, 1))}, tmp_path / "unquantized_layers.safetensors")
    (tmp_path / "config.json").write_text('{"num_layers": 1}', encoding="utf-8")

    _, metadata = merge_safetensors(tmp_path, "NunchakuFluxTransformer2dModel")

    quantization_config = json.loads(metadata["quantization_config"])
    assert quantization_config["weight"] == {
        "dtype": "fp4_e2m1_all",
        "scale_dtype": "ue8m0",
        "group_size": 32,
    }
    assert quantization_config["activation"] == {
        "dtype": "fp4_e2m1_all",
        "scale_dtype": "ue8m0",
        "group_size": 32,
    }
    assert quantization_config["rank"] == 32
    assert metadata["config"] == '{"num_layers": 1}'
    assert metadata["comfy_config"] == "{}"
    assert metadata["format"] == "pt"
