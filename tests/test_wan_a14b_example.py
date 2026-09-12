import importlib.util
from pathlib import Path

import pytest
import torch
from diffusers import UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

spec = importlib.util.spec_from_file_location(
    "wan_a14b_example", Path(__file__).resolve().parents[1] / "examples" / "wan22_t2v_a14b.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_trajectory_audit_observes_real_wan_dual_expert_switch():
    def expert():
        return WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=16,
            ffn_dim=64,
            num_layers=1,
            rope_max_seq_len=32,
        )

    pipe = WanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=expert(),
        transformer_2=expert(),
        scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0),
        boundary_ratio=0.875,
    )
    audit = example.TrajectoryAudit(pipe)
    try:
        result = pipe(
            prompt_embeds=torch.randn(1, 4, 32),
            negative_prompt_embeds=torch.zeros(1, 4, 32),
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=4,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            output_type="latent",
            generator=torch.Generator().manual_seed(0),
            callback_on_step_end=audit.callback,
        ).frames
        report = audit.finish()
        assert report["expert_calls"] == {"transformer": 4, "transformer_2": 4}
        assert len(report["steps"]) == 4
        assert torch.isfinite(result).all()
    finally:
        audit.close()
    assert not pipe.transformer._forward_pre_hooks
    assert not pipe.transformer_2._forward_pre_hooks


def test_trajectory_audit_rejects_nans_and_missing_expert():
    from types import SimpleNamespace

    pipe = SimpleNamespace(transformer=torch.nn.Identity(), transformer_2=torch.nn.Identity())
    audit = example.TrajectoryAudit(pipe)
    try:
        with pytest.raises(FloatingPointError, match="step 0"):
            audit.callback(pipe, 0, 1000, {"latents": torch.tensor([float("nan")])})
        with pytest.raises(RuntimeError, match="Both experts"):
            audit.finish()
    finally:
        audit.close()


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0)])
def test_mxfp4_rejects_unsupported_native_kernel_architecture(monkeypatch, capability):
    from nunchaku.utils import check_hardware_compatibility

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(ValueError, match="requires SM120/SM121"):
        check_hardware_compatibility({"weight": {"dtype": "mxfp4", "group_size": 32}})


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_mxfp4_accepts_supported_native_kernel_architecture(monkeypatch, capability):
    from nunchaku.utils import check_hardware_compatibility

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    check_hardware_compatibility({"weight": {"dtype": "mxfp4", "group_size": 32}})
