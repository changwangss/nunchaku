# Wan2.2 T2V A14B：AutoRound SVDQuant 推理

使用 Nunchaku 分支 `wangchang/wan-mxfp4-runtime`，搭配 AutoRound fork 的 `wangchang/wan-svdquant-nunchaku`。
完整校准导出命令见 AutoRound 分支的 `B200_WAN_SVDQUANT_HANDOFF_CN.md`。
目标源模型是 `Wan-AI/Wan2.2-T2V-A14B-Diffusers`。

## 硬件边界

此原生 MXFP4 内核要求 SM120/SM121（例如 RTX 5090）及匹配的 CUDA 扩展。
B200/SM100、H100/SM90 尚不支持；SM120 的 `mma.sync` block-scale 指令无法针对 SM100 编译，
需单独实现 SM100 `tcgen05` 内核。示例在加载权重前检查硬件。
AutoRound 导出 Nunchaku 格式无需安装该推理库。

## 运行

在 Nunchaku 仓库目录中，先准备匹配的 CUDA PyTorch 环境及 toolkit：

```bash
git submodule update --init --recursive
python -m pip install ninja wheel setuptools imageio imageio-ffmpeg
CUDA_VISIBLE_DEVICES=0 NUNCHAKU_INSTALL_MODE=FAST python -m pip install --no-build-isolation -e .
python -m pip install 'diffusers==0.39.0' 'transformers==5.12.1'
```

SM120 要求 CUDA toolkit 12.8 或以上；SM121 要求 13.0 或以上。
开发环境为 Python 3.12、PyTorch 2.13.0+cu130、Diffusers 0.39.0、Transformers 5.12.1。

传入包含两个专家目录的完整 AutoRound 本地导出：

```bash
CUDA_VISIBLE_DEVICES=0 python -u examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-latent-smoke \
  --latent-only --height 256 --width 256 --num-frames 9 --steps 4
```

再用另一输出目录生成短视频：

```bash
CUDA_VISIBLE_DEVICES=0 python -u examples/wan22_t2v_a14b.py \
  --model /data/wan/a14b-svdquant-smoke --output /data/wan/a14b-video-smoke \
  --height 384 --width 640 --num-frames 33 --steps 30 --seed 0
```

示例显式加载两个专家，每个 40 层、400 个量化投影；保留源 scheduler 和切换阈值，
使用 FP32 VAE、tiling，默认启用 pipeline CPU model offload。
`--no-cpu-offload` 需要足够显存放下完整 pipeline。输出路径必须不存在。

每个去噪步的 latent 必须有限，forward hook 必须观察到两个专家实际执行；解码像素也要有限。
`audit.json` 记录逐步结果、专家调用数、耗时、CUDA 峰值分配内存。
输出为 `latents.pt` 或 `video.mp4`（默认 16 FPS）。四步 smoke 只检查执行链路，画面结构和运动需查看更长视频。

## 验证范围与 loader 修复

Wan loader 在 `to_empty` 后重建非持久化 RoPE buffer，因为这些 buffer 不在 safetensors 中。
不修复时，即使权重加载成功也可能产生模糊画面或非有限轨迹。
修复后，同一个本地 5B checkpoint 能产生可辨识运动。
开发机器没有下载或运行完整 A14B 权重，A14B 显存、画质与完整目标模型执行仍待目标机器验证。

加载 BF16 残差及低秩权重时，loader 也保留 Wan 声明的 FP32 时间嵌入、归一化参数和 scale-shift table。
