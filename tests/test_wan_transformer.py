import torch

from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.transformers.transformer_wan import (
    WAN_QUANTIZED_PROJECTIONS,
    NunchakuWanTransformer3DModel,
    _convert_state_dict_keys,
    _get_child,
)


def _tiny_wan_model():
    config = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 64,
        "freq_dim": 32,
        "ffn_dim": 128,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 64,
    }
    with torch.device("meta"):
        return NunchakuWanTransformer3DModel(**config)


def test_wan_mxfp4_patch_replaces_all_exported_projection_paths():
    model = _tiny_wan_model()
    original_shapes = {
        path: (_get_child(model.blocks[0], path).in_features, _get_child(model.blocks[0], path).out_features)
        for path in WAN_QUANTIZED_PROJECTIONS
    }

    model._patch_model(precision="mxfp4", rank=16)

    assert len(WAN_QUANTIZED_PROJECTIONS) == 10
    for path, (in_features, out_features) in original_shapes.items():
        layer = _get_child(model.blocks[0], path)
        assert isinstance(layer, SVDQW4A4Linear)
        assert (layer.in_features, layer.out_features) == (in_features, out_features)
        assert layer.precision == "mxfp4"
        assert layer.group_size == 32
        assert layer.rank == 16
        assert layer.qweight.shape == (out_features, in_features // 2)
        assert layer.wscales.shape == (in_features // 32, out_features)


def test_wan_projection_dimensions_match_runtime_mapping():
    model = _tiny_wan_model()
    block = model.blocks[0]

    for path in WAN_QUANTIZED_PROJECTIONS[:8]:
        layer = _get_child(block, path)
        assert (layer.in_features, layer.out_features) == (64, 64)
    assert (_get_child(block, "ffn.net.0.proj").in_features, _get_child(block, "ffn.net.0.proj").out_features) == (
        64,
        128,
    )
    assert (_get_child(block, "ffn.net.2").in_features, _get_child(block, "ffn.net.2").out_features) == (128, 64)


def test_wan_export_keys_are_converted_to_nunchaku_linear_names():
    tensors = {
        "blocks.0.attn1.to_q.lora_down": torch.empty(1),
        "blocks.0.attn1.to_q.lora_up": torch.empty(1),
        "blocks.0.attn1.to_q.smooth": torch.empty(1),
        "blocks.0.attn1.to_q.smooth_orig": torch.empty(1),
        "blocks.0.attn1.to_q.qweight": torch.empty(1),
    }

    converted = _convert_state_dict_keys(tensors)

    assert set(converted) == {
        "blocks.0.attn1.to_q.proj_down",
        "blocks.0.attn1.to_q.proj_up",
        "blocks.0.attn1.to_q.smooth_factor",
        "blocks.0.attn1.to_q.smooth_factor_orig",
        "blocks.0.attn1.to_q.qweight",
    }


def test_wan_onefile_loader_restores_nonpersistent_rope_buffers(tmp_path):
    import json

    from safetensors.torch import save_file

    model = _tiny_wan_model()
    model._patch_model(precision="mxfp4", rank=16)
    model = model.to_empty(device="cpu").to(torch.bfloat16)
    state = {
        key.replace(".smooth_factor_orig", ".smooth_orig")
        .replace(".smooth_factor", ".smooth")
        .replace(".proj_down", ".lora_down")
        .replace(".proj_up", ".lora_up"): torch.zeros_like(value)
        for key, value in model.state_dict().items()
    }
    protected_key = "condition_embedder.time_embedder.linear_1.weight"
    state[protected_key] = torch.full_like(state[protected_key], 1.001, dtype=torch.float32)
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        state,
        checkpoint,
        metadata={
            "config": json.dumps(dict(model.config)),
            "quantization_config": json.dumps({"rank": 16, "weight": {"dtype": "mxfp4", "group_size": 32}}),
        },
    )

    loaded = NunchakuWanTransformer3DModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    assert loaded.condition_embedder.time_embedder.linear_1.weight.dtype == torch.float32
    assert loaded.scale_shift_table.dtype == torch.float32
    torch.testing.assert_close(loaded.state_dict()[protected_key], state[protected_key], rtol=0, atol=0)
    assert loaded.blocks[0].attn1.to_q.proj_down.dtype == torch.bfloat16
    expected = type(loaded.rope)(
        loaded.config.attention_head_dim, loaded.config.patch_size, loaded.config.rope_max_seq_len
    ).to(dtype=torch.bfloat16)

    assert "rope.freqs_cos" not in state
    assert "rope.freqs_sin" not in state
    torch.testing.assert_close(loaded.rope.freqs_cos, expected.freqs_cos, rtol=0, atol=0)
    torch.testing.assert_close(loaded.rope.freqs_sin, expected.freqs_sin, rtol=0, atol=0)
