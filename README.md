# ASRwin — 可复现的低延迟流式 ASR 基准

> 在 Windows 11 + RTX 5060 Ti 8GB (Blackwell) 上，用 **sherpa-onnx + streaming Zipformer (RNN-T)** 跑通论文级精度复现 + 流式延迟实测。**CPU 推理**，无须 GPU。

| 指标 | ASRwin 实测 | 论文 / 官方 | 误差 |
|---|---|---|---|
| LibriSpeech test-clean **WER** | **3.24%** | 3.30% | −0.06 |
| AISHELL-1 test **CER** | **4.16%** | 4.60% | −0.44 |
| **RTF** (CPU 4 线程) | **0.05** | ~0.3 | 6× 优于预期 |
| 流式首字延迟 | **~1.1 s** | <2 s | ✅ |

![Accuracy vs paper](plots/accuracy_vs_paper.png)

---

## 目录

- [背景与目标](#背景与目标)
- [硬件 / 软件环境](#硬件--软件环境)
- [方案选型与权衡](#方案选型与权衡)
- [复现步骤](#复现步骤)
- [实测结果](#实测结果)
- [项目结构](#项目结构)
- [可扩展方向](#可扩展方向)
- [致谢](#致谢)

---

## 背景与目标

需求是"在 Windows 上能复现一个低延迟流式 ASR，开源、可重跑、能给出硬指标"。MoChA (2017) 已被现代 streaming Transducer / chunked Conformer / Zipformer 全面取代，所以选当前 SOTA 的 **Zipformer streaming RNN-T**。

**三条硬指标**：CER/WER（精度）、RTF（吞吐）、首字延迟（用户感知）。每条都必须能跟公开论文/官方数字对照。

## 硬件 / 软件环境

| 项 | 配置 | 备注 |
|---|---|---|
| GPU | **RTX 5060 Ti 8GB (Blackwell, sm_120)** | 需 CUDA ≥ 12.8 / PyTorch ≥ 2.7 — 但本项目走 ONNX，**用不上 CUDA** |
| OS | Windows 11 26100 (24H2) | |
| 运行环境 | WSL2 Ubuntu 22.04 + conda env `asr` (Python 3.10) | |
| 推理框架 | sherpa-onnx 1.13.1 (CPU onnxruntime) | |
| 评测库 | jiwer (WER/CER)，matplotlib (plots) | |
| 模型 | `csukuangfj/sherpa-onnx-streaming-zipformer-*` (HuggingFace, hf-mirror.com 中转) | |

## 方案选型与权衡

| 决策 | 替代方案 | 选 sherpa-onnx 的原因 |
|---|---|---|
| sherpa-onnx 而非 icefall + PyTorch | icefall + k2 + PyTorch nightly | 绕开 Blackwell sm_120 上 k2 / 自定义 CUDA kernel 的编译坑 |
| Zipformer streaming 而非 MoChA | MoChA / Conformer-CTC | MoChA 2017 已淘汰；Zipformer 2024 是当前 SOTA |
| CPU 推理而非 GPU | onnxruntime-gpu + CUDA EP | CPU RTF=0.05 已 20× 实时，GPU 边际收益小 |
| WSL2 而非 Windows 原生 | Windows 原生 Python | 工具链统一，apt 装 sox/ffmpeg/build 方便 |

## 复现步骤

### 1. 装环境（WSL2 Ubuntu）

```bash
# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
bash mc.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash && source ~/.bashrc

# asr env
conda create -n asr python=3.10 -y && conda activate asr
pip install sherpa-onnx onnxruntime soundfile numpy jiwer tqdm matplotlib pandas huggingface_hub
sudo apt install -y build-essential cmake git ffmpeg sox libsox-fmt-all
```

### 2. 下模型（国内走 hf-mirror）

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p models && cd models
python -c "
from huggingface_hub import snapshot_download
for repo, name in [
  ('csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20',
   'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'),
  ('csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26',
   'sherpa-onnx-streaming-zipformer-en-2023-06-26'),
]:
    snapshot_download(repo_id=repo, local_dir=name)
"
```

### 3. 下数据集

```bash
mkdir -p data && cd data
# LibriSpeech test-clean (~346MB, 2620 utt)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz && rm test-clean.tar.gz

# AISHELL-1 (~15GB tar, 含 train/dev/test 全部 400 speakers)
wget https://www.openslr.org/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz
# 只解 test 集 20 speakers
bash ../scripts/extract_aishell_test.sh
```

### 4. 跑基准

```bash
cd /path/to/ASRwin

# 烟雾测试 (5 条 wav, ~10 秒)
python scripts/smoke_decode.py

# 延迟基准 (test_wavs, 模拟实时)
python scripts/bench_latency.py --model zipformer-bi --chunk-ms 100

# 精度基准 (英文 WER, 17 min)
python scripts/bench_accuracy.py --model zipformer-en --dataset librispeech \
    --data data/LibriSpeech/test-clean --threads 4

# 精度基准 (中文 CER, 30 min)
python scripts/bench_accuracy.py --model zipformer-bi --dataset aishell \
    --data data/data_aishell/wav/test \
    --aishell-trans data/data_aishell/transcript/aishell_transcript_v0.8.txt \
    --threads 4

# 流式实时 demo (彩色 partial 假设逐字成长)
python scripts/demo_streaming.py \
    models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav

# 生成 4 张图
python scripts/make_plots.py
```

## 实测结果

### 精度：贴合论文

![Accuracy vs paper](plots/accuracy_vs_paper.png)

两项均落在 ±0.5% 论文复现目标内。AISHELL CER 实际略好于官方（−0.44）属归一化差异范围内。

### 逐句误差分布

LibriSpeech test-clean：**>50% 句子 0% WER**，长尾集中在 10% 以内。

![LibriSpeech per-utt WER](plots/wer_hist_librispeech.png)

AISHELL-1 test：**~40% 句子 0% CER**，分布更紧凑（中文逐字评测，单字错惩罚轻）。

![AISHELL per-utt CER](plots/cer_hist_aishell.png)

### 吞吐：CPU 上 20× 实时

200 句采样下，RTF 几乎不随音频长度变化，AISHELL 上稍快（中文 token 短）。

![RTF vs audio length](plots/rtf_vs_audio_len.png)

### 流式行为（demo_streaming 截录）

```
[ 1.10s] 昨                                           ← 首字 1.1s
[ 1.40s] 昨天
[ 2.40s] 昨天是 MON
[ 2.70s] 昨天是 MONDAY
[ 5.60s] 昨天是 MONDAY TODAY IS
[ 8.50s] 昨天是 MONDAY TODAY IS LIBR THE DAY AFTER TOMORROW
[10.05s] 昨天是 MONDAY TODAY IS LIBR THE DAY AFTER TOMORROW是星期三  ← final
```

墙钟 ≈ 音频时长，**chunk 提交后响应 ~18 ms**，首字 ~1.1 s（含模型算法 look-ahead）。

## 项目结构

```
ASRwin/
├── README.md                     ← 本文件
├── PLAN.md                       ← 原始 7 步里程碑规划
├── RESULTS.md                    ← 详细结果与指标定义
├── scripts/
│   ├── smoke_decode.py           ← 单 wav 流式解码烟雾测试
│   ├── bench_latency.py          ← 延迟基准 (含实时模拟)
│   ├── bench_accuracy.py         ← LibriSpeech WER / AISHELL CER
│   ├── demo_streaming.py         ← 彩色实时流式演示
│   ├── make_plots.py             ← 生成 4 张图
│   └── extract_aishell_test.sh   ← AISHELL test 20 speaker 解包
├── plots/                        ← README 引用的 PNG
├── results/                      ← 评测 CSV + 汇总 txt
├── models/                       ← (gitignored) 下载的 ONNX 模型
└── data/                         ← (gitignored) LibriSpeech / AISHELL
```

`models/` 与 `data/` 加起来约 ~1.8 GB，按上述脚本下载即可。

## 可扩展方向

- **更短 chunk**：测 `chunk-16-left-64` 变体的首字延迟，比 `left-128` 应更低
- **GPU 推理**：换 `onnxruntime-gpu` 配 CUDA 12.8 EP，对照 CPU 速度增量
- **FunASR Paraformer / WeNet U2++** 横向对比（本项目 paraformer 仓库在 hf-mirror 401，需备选源）
- **端到端 speech LLM**：Qwen2-Audio-7B 4-bit 量化（8GB VRAM 边界，复现难度更高）
- **真实麦克风**：`sounddevice` 接 `OnlineRecognizer`（建议跑 Windows 原生 Python，WSL 麦克风转发烦）

## 致谢

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 推理框架与预训练模型
- [icefall](https://github.com/k2-fsa/icefall) — Zipformer streaming 模型训练
- [OpenSLR](https://www.openslr.org/) — LibriSpeech / AISHELL-1 数据
- [hf-mirror.com](https://hf-mirror.com) — HuggingFace 国内镜像

## License

代码 MIT。模型/数据各自的许可证以原仓库为准。
