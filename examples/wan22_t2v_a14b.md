# Wan2.2 T2V A14B with AutoRound SVDQuant

Use this branch (`wangchang/wan-mxfp4-runtime`) with the AutoRound fork branch `wangchang/wan-svdquant-nunchaku`.
The AutoRound branch's `B200_WAN_SVDQUANT_HANDOFF.md` and `_CN.md` contain the calibrated export commands.
The target source is `Wan-AI/Wan2.2-T2V-A14B-Diffusers`.

## Hardware boundary

This native MXFP4 backend requires SM120/SM121, such as RTX 5090, with a matching compiled extension.
B200/SM100 and H100/SM90 are unsupported. The SM120 `mma.sync` block-scale instruction cannot be compiled for SM100;
a separate SM100 `tcgen05` kernel is required. The example checks this before loading model tensors.
AutoRound can produce Nunchaku-format exports without installing this runtime.

## Run

From the Nunchaku checkout, with a compatible CUDA PyTorch environment and toolkit:

```bash
git submodule update --init --recursive
python -m pip install ninja wheel setuptools imageio imageio-ffmpeg
CUDA_VISIBLE_DEVICES=0 NUNCHAKU_INSTALL_MODE=FAST python -m pip install --no-build-isolation -e .
python -m pip install 'diffusers==0.39.0' 'transformers==5.12.1'
```

SM120 requires CUDA toolkit 12.8 or later; SM121 requires 13.0 or later.
The development environment used Python 3.12, PyTorch 2.13.0+cu130, Diffusers 0.39.0 and Transformers 5.12.1.

Use a complete local AutoRound pipeline export with both expert directories:

```bash
CUDA_VISIBLE_DEVICES=0 python -u examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-latent-smoke \
  --latent-only --height 256 --width 256 --num-frames 9 --steps 4
```

Then generate a short video in another output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -u examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-video-smoke \
  --height 384 --width 640 --num-frames 33 --steps 30 --seed 0
```

The example loads both 40-block, 400-projection experts explicitly, retains the source scheduler and boundary ratio,
uses FP32 VAE with tiling, and enables pipeline CPU model offload by default. `--no-cpu-offload` requires enough
VRAM for the whole pipeline. All output paths must be new.

Every denoising step must be finite, and forward hooks must observe both experts. Decoded pixels are also checked.
`audit.json` records steps, actual expert forward counts, timing and peak allocated CUDA memory.
Results are `latents.pt` or `video.mp4` (16 FPS by default). A four-step smoke checks execution only; inspect a longer
video to evaluate structure and motion.

## Validation limits and loader fix

The Wan loader now reconstructs non-persistent RoPE buffers after `to_empty`; they are not present in safetensors.
Without this repair, loaded models can produce blurred frames or non-finite trajectories despite successful weight loading.
The same local 5B checkpoint produced recognizable motion after the repair. No full A14B weights were downloaded or
run on the development machine; A14B memory usage, output quality and complete target-model execution remain unverified.

The loader also preserves Wan-declared FP32 time embeddings, normalization parameters and scale-shift tables when loading BF16 residual/low-rank weights.
