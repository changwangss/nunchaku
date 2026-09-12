"""Smoke-test both Nunchaku Wan2.2 T2V A14B experts and optionally export a video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local AutoRound exported pipeline directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--latent-only", action="store_true")
    parser.add_argument(
        "--prompt", default="An orange cat walks through a sunlit garden, flowers swaying in the breeze."
    )
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted, static, oversaturated")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA index")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error("height and width must be positive multiples of 16")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("num-frames must be 4k+1")
    if args.steps < 4 or args.fps < 1:
        parser.error("steps must be at least 4 to check both experts; fps must be positive")
    return args


class TrajectoryAudit:
    """Observe actual expert forwards and reject non-finite denoising states."""

    def __init__(self, pipe):
        self.calls = dict(transformer=0, transformer_2=0)
        self.steps = []
        self.handles = []
        for name in self.calls:

            def record(module, inputs, component=name):
                self.calls[component] += 1

            self.handles.append(getattr(pipe, name).register_forward_pre_hook(record))

    def callback(self, pipe, step, timestep, kwargs):
        latents = kwargs["latents"]
        finite = bool(torch.isfinite(latents).all().item())
        entry = dict(step=step, timestep=float(timestep), finite=finite, expert_calls=dict(self.calls))
        self.steps.append(entry)
        print(json.dumps(entry), flush=True)
        if not finite:
            raise FloatingPointError(f"Non-finite latents at step {step}, timestep {float(timestep)}")
        return kwargs

    def finish(self):
        if not self.steps or not all(self.calls.values()):
            raise RuntimeError(f"Both experts must execute; observed calls: {self.calls}")
        return dict(expert_calls=self.calls, steps=self.steps)

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_pipeline(model_path):
    from diffusers import AutoencoderKLWan, WanPipeline

    from nunchaku import NunchakuWanTransformer3DModel
    from nunchaku.models.linear import SVDQW4A4Linear

    experts = {}
    for name in ("transformer", "transformer_2"):
        model = NunchakuWanTransformer3DModel.from_pretrained(model_path / name, torch_dtype=torch.bfloat16)
        if model.config.num_layers != 40 or model.config.in_channels != 16:
            raise ValueError(f"{name} is not a Wan2.2 T2V A14B expert")
        if sum(isinstance(module, SVDQW4A4Linear) for module in model.modules()) != 400:
            raise ValueError(f"{name} does not contain 400 Nunchaku projections")
        experts[name] = model
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanPipeline.from_pretrained(
        model_path, **experts, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True
    )
    if pipe.config.boundary_ratio != 0.875:
        raise ValueError("Expected official A14B boundary_ratio=0.875")
    # Keep the model's scheduler and temporal/latent configuration unchanged.
    pipe.vae.enable_tiling()
    return pipe


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("Nunchaku MXFP4 inference requires a supported CUDA GPU")

    from nunchaku.utils import check_hardware_compatibility

    device = torch.device("cuda", args.device)
    check_hardware_compatibility({"weight": {"dtype": "mxfp4", "group_size": 32}}, device)
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pipe = load_pipeline(args.model)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    audit = TrajectoryAudit(pipe)
    start = time.monotonic()
    try:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            output_type="latent" if args.latent_only else "np",
            callback_on_step_end=audit.callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames
        report = audit.finish()
        if args.latent_only:
            torch.save(result.cpu(), args.output / "latents.pt")
        else:
            import numpy as np
            from diffusers.utils import export_to_video

            if not np.isfinite(result).all():
                raise FloatingPointError("VAE output contains non-finite pixels")
            export_to_video(result[0], str(args.output / "video.mp4"), fps=args.fps)
        report.update(
            seconds=time.monotonic() - start, peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30
        )
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        audit.close()


if __name__ == "__main__":
    main()
