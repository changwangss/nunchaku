import pytest
import torch
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_utils_module():
    spec = importlib.util.spec_from_file_location("nunchaku_utils_under_test", ROOT / "nunchaku" / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_linear_module(monkeypatch, calls=None):
    if calls is None:
        calls = []

    package = types.ModuleType("nunchaku")
    package.__path__ = [str(ROOT / "nunchaku")]
    models_package = types.ModuleType("nunchaku.models")
    models_package.__path__ = [str(ROOT / "nunchaku" / "models")]
    ops_package = types.ModuleType("nunchaku.ops")
    ops_package.__path__ = [str(ROOT / "nunchaku" / "ops")]

    gemm_module = types.ModuleType("nunchaku.ops.gemm")
    def fake_gemm(*args, **kwargs):
        calls.append(("gemm", kwargs))

    gemm_module.svdq_gemm_w4a4_cuda = fake_gemm
    gemv_module = types.ModuleType("nunchaku.ops.gemv")
    gemv_module.awq_gemv_w4a16_cuda = lambda *args, **kwargs: None
    quantize_module = types.ModuleType("nunchaku.ops.quantize")

    def fake_quantize(input, **kwargs):
        calls.append(("quantize", kwargs))
        batch_size, channels = input.shape
        rank = kwargs["lora_down"].shape[1]
        return (
            torch.empty(batch_size, channels // 2, dtype=torch.uint8),
            torch.empty(channels // 32, batch_size, dtype=torch.uint8),
            torch.empty(batch_size, rank, dtype=torch.float32),
        )

    quantize_module.svdq_quantize_w4a4_act_fuse_lora_cuda = fake_quantize

    monkeypatch.setitem(sys.modules, "nunchaku", package)
    monkeypatch.setitem(sys.modules, "nunchaku.models", models_package)
    monkeypatch.setitem(sys.modules, "nunchaku.ops", ops_package)
    monkeypatch.setitem(sys.modules, "nunchaku.ops.gemm", gemm_module)
    monkeypatch.setitem(sys.modules, "nunchaku.ops.gemv", gemv_module)
    monkeypatch.setitem(sys.modules, "nunchaku.ops.quantize", quantize_module)

    spec = importlib.util.spec_from_file_location(
        "nunchaku.models.linear", ROOT / "nunchaku" / "models" / "linear.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_precision_from_quantization_config_accepts_mxfp4_group32():
    get_precision_from_quantization_config = load_utils_module().get_precision_from_quantization_config
    config = {"weight": {"dtype": "mxfp4", "group_size": 32}}

    assert get_precision_from_quantization_config(config) == "mxfp4"


def test_svdq_linear_mxfp4_has_group32_and_dispatches_mxfp4_kernel(monkeypatch):
    calls = []
    SVDQW4A4Linear = load_linear_module(monkeypatch, calls).SVDQW4A4Linear
    layer = SVDQW4A4Linear(64, 32, rank=8, precision="mxfp4")

    assert layer.precision == "mxfp4"
    assert layer.group_size == 32
    assert layer.wscales.shape == (2, 32)

    x = torch.empty(1, 64, dtype=torch.bfloat16)
    quantized_x, ascales, lora_act = layer.quantize(x)
    layer.forward_quant(quantized_x, ascales, lora_act)

    assert calls[0][0] == "quantize"
    assert calls[0][1]["fp4"] is False
    assert calls[0][1]["mxfp4"] is True
    assert calls[1][0] == "gemm"
    assert calls[1][1]["fp4"] is False
    assert calls[1][1]["mxfp4"] is True
