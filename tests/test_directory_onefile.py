import json
from pathlib import Path

import diffusers
import nunchaku
import torch
from diffusers import DiffusionPipeline, ModelMixin

from nunchaku.models.transformers.utils import resolve_pretrained_onefile


def test_resolve_pretrained_onefile_prefers_diffusers_weight_name(tmp_path):
    onefile = tmp_path / "diffusion_pytorch_model.safetensors"
    onefile.touch()

    assert resolve_pretrained_onefile(tmp_path) == onefile


def test_resolve_pretrained_onefile_supports_legacy_model_name(tmp_path):
    onefile = tmp_path / "model.safetensors"
    onefile.touch()

    assert resolve_pretrained_onefile(tmp_path) == onefile


def test_resolve_pretrained_onefile_preserves_legacy_directory(tmp_path):
    assert resolve_pretrained_onefile(tmp_path) == Path(tmp_path)


def test_resolve_pretrained_onefile_preserves_explicit_file(tmp_path):
    onefile = tmp_path / "custom.safetensors"
    onefile.touch()

    assert resolve_pretrained_onefile(onefile) == onefile


def test_nunchaku_exposes_diffusers_model_mixin_for_pipeline_loading():
    assert nunchaku.ModelMixin is ModelMixin


def test_diffusers_pipeline_routes_component_directory_to_nunchaku(monkeypatch, tmp_path):
    class TinyNunchakuPipeline(DiffusionPipeline):
        def __init__(self, transformer):
            super().__init__()
            self.register_modules(transformer=transformer)

    captured = {}

    def fake_from_pretrained(cls, path, **kwargs):
        captured["path"] = Path(path)
        captured["torch_dtype"] = kwargs["torch_dtype"]
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(diffusers, "TinyNunchakuPipeline", TinyNunchakuPipeline, raising=False)
    monkeypatch.setattr(
        nunchaku.NunchakuFluxTransformer2dModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )
    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "TinyNunchakuPipeline",
                "_diffusers_version": diffusers.__version__,
                "transformer": ["nunchaku", "NunchakuFluxTransformer2dModel"],
            }
        ),
        encoding="utf-8",
    )

    pipeline = DiffusionPipeline.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)

    assert isinstance(pipeline.transformer, torch.nn.Linear)
    assert captured == {"path": tmp_path / "transformer", "torch_dtype": torch.bfloat16}
