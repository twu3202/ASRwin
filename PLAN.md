# ASRwin — 流式 ASR 复现与基准测试方案

> 路线：sherpa-onnx + streaming Zipformer (RNN-T) on WSL2 Ubuntu
> 目标：在 RTX 5060 Ti 8GB / Windows 11 上跑通 **可复现的低延迟流式 ASR**，给出首包延迟、RTF、CER/WER 三项硬指标
> 日期：2026-05-13

---

## 0. 硬件 / 软件约束

| 项 | 现状 | 注意 |
|---|---|---|
| GPU | RTX 5060 Ti 8GB (Blackwell, sm_120) | 必须 CUDA ≥ 12.8 / PyTorch ≥ 2.7；老 wheel 全部不兼容 |
| OS | Windows 11 Pro 26100 | WSL2 需管理员 + 重启 |
| 当前已装 | nvidia 驱动 595.76 | 满足 CUDA 12.8 要求 (driver ≥ 555) |
| 未装 | WSL、Python、conda、CUDA toolkit | 见 §1 |

> **为什么选 sherpa-onnx**：推理走 onnxruntime，绕开 Blackwell 上 PyTorch / k2 自定义 kernel 的编译坑；预训练 streaming Zipformer 直接下载即用；C++/Python/CLI 三套接口，麦克风 demo 现成。

---

## 1. 环境准备 (一次性)

### 1.1 装 WSL2 (管理员 PowerShell)
```powershell
wsl --install -d Ubuntu-22.04
# 重启后首次进 Ubuntu 设置用户名/密码
```

### 1.2 验证 GPU 透传
```bash
nvidia-smi   # WSL 内应看到 RTX 5060 Ti
```

### 1.3 装 Miniconda + 工具链
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda create -n asr python=3.10 -y && conda activate asr
sudo apt update && sudo apt install -y build-essential cmake git sox libsox-fmt-all ffmpeg portaudio19-dev
```

### 1.4 工作目录映射
WSL 内访问：`/mnt/d/Winprojects/ASRwin/`
所有代码与下载产物都放这里，Windows 侧也可见。

---

## 2. 安装 sherpa-onnx

```bash
# CPU 版即可（ONNX 推理 CPU 已足够流式实时；GPU 版收益有限且依赖额外 CUDA EP）
pip install sherpa-onnx onnxruntime soundfile sounddevice numpy pyyaml jiwer tqdm

# 验证
python -c "import sherpa_onnx; print(sherpa_onnx.__version__)"
sherpa-onnx --help
```

GPU 版（可选，做对比）：`pip install sherpa-onnx onnxruntime-gpu`

---

## 3. 下载预训练模型 (3 个对比)

放到 `/mnt/d/Winprojects/ASRwin/models/`

| 模型 | 语言 | 大小 | 用途 |
|---|---|---|---|
| `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | 中英双语 | ~150MB | **主基线** |
| `sherpa-onnx-streaming-zipformer-en-2023-06-26` | 英文 LibriSpeech | ~80MB | WER 对照 |
| `sherpa-onnx-streaming-paraformer-bilingual-zh-en-2023-11-09` | 中英 (Paraformer) | ~250MB | 架构对照 |

```bash
cd /mnt/d/Winprojects/ASRwin && mkdir -p models && cd models
# 三个模型 HuggingFace 链接 (sherpa 官方):
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en-2023-11-09
```

---

## 4. 跑通基线 (烟雾测试)

```bash
cd /mnt/d/Winprojects/ASRwin

# 单条 wav 流式解码
sherpa-onnx \
  --tokens=models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --encoder=models/.../encoder-epoch-99-avg-1.onnx \
  --decoder=models/.../decoder-epoch-99-avg-1.onnx \
  --joiner=models/.../joiner-epoch-99-avg-1.onnx \
  test.wav

# 麦克风实时（在 Windows 跑更方便，WSL 拿麦克风要折腾 PulseAudio）
```

---

## 5. 测试集准备

| 数据集 | 语种 | 时长 | 下载 |
|---|---|---|---|
| **AISHELL-1 test** | 中文 | 5h, 7176 utt | OpenSLR-33 |
| **LibriSpeech test-clean** | 英文 | 5.4h | OpenSLR-12 |
| **LibriSpeech test-other** | 英文(噪) | 5.1h | OpenSLR-12 |

```bash
mkdir -p data && cd data
wget https://www.openslr.org/resources/33/data_aishell.tgz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz
# 解包略
```

---

## 6. 基准测试脚本设计 (要写的代码)

放 `scripts/`：

### 6.1 `bench_latency.py` — 首包 / 末包延迟
- 输入一批 wav，按 **真实采样率** 喂 frame chunk (默认 320 sample = 20ms)
- 记录：
  - **First-token latency**：从音频开始到第一个非空假设
  - **Last-token latency**：音频结束到 final hypothesis
  - **RTF** = 解码耗时 / 音频时长
- 输出 CSV：`utt_id, audio_sec, first_ms, last_ms, rtf`
- 工具：`sherpa_onnx.OnlineRecognizer` + `time.perf_counter()`

### 6.2 `bench_accuracy.py` — CER/WER
- 解码整个 test set，文本归一化后跟 jiwer 比对
- 中文按字符 (CER)，英文按词 (WER)
- 输出每模型一份报告

### 6.3 `bench_realtime_mic.py` — 麦克风实时演示
- `sounddevice` 抓 16kHz mono，每 20ms 推一次
- 实时打印 partial / final
- 在 Windows 原生 Python 跑（不进 WSL，避开音频驱动问题）

> Windows 原生只装一个轻量 venv：`python -m venv .venv-win && .venv-win\Scripts\pip install sherpa-onnx sounddevice`

---

## 7. 评估目标（论文公开指标对照）

| 模型 | 数据集 | 论文/官方 | 目标 (我们) |
|---|---|---|---|
| Zipformer streaming bilingual | AISHELL-1 test | CER ~4.6% | 复现 ±0.3 |
| Zipformer streaming en | test-clean | WER ~3.3% | 复现 ±0.3 |
| Zipformer streaming en | test-other | WER ~8.7% | 复现 ±0.5 |
| 首包延迟 (chunk=320ms) | — | <400ms | <500ms (CPU) |
| RTF (CPU 单线程) | — | ~0.3 | <0.5 |

---

## 8. 里程碑

1. **M1 (今天)**: §1 环境就绪，`nvidia-smi` 在 WSL 看到卡
2. **M2**: §2-4 sherpa-onnx 装好，单条 wav 解码出正确结果
3. **M3**: §5 三个数据集本地齐
4. **M4**: §6.1 延迟脚本跑出第一张表
5. **M5**: §6.2 CER/WER 复现到论文 ±0.5
6. **M6**: §6.3 Windows 端麦克风 demo
7. **M7 (可选)**: 加入 FunASR Paraformer-streaming + WeNet U2++ 横向对比；如显存富裕，再上 Zipformer GPU 推理

---

## 9. 风险与备选

| 风险 | 触发条件 | 备选 |
|---|---|---|
| WSL GPU 透传失败 | 旧 Windows 内核 | `wsl --update`；或全部在 Windows 原生 Python 跑 (sherpa-onnx CPU 即可) |
| HuggingFace clone 慢 | 国内网络 | 用 `HF_ENDPOINT=https://hf-mirror.com` |
| ONNX CPU 不够实时 | RTF > 1 | 切 `onnxruntime-gpu` + CUDA EP，或减少 chunk size |
| 想训练 / 微调 | 论文级复现 | 转 icefall + k2（GPU 训练，要解决 Blackwell + k2 编译，难度↑） |

---

## 10. 下一步立即动作

1. 管理员 PowerShell 跑 `wsl --install -d Ubuntu-22.04` 然后重启
2. 重启回来告诉我，继续 §1.3 之后的步骤（我会一步步带你跑）
