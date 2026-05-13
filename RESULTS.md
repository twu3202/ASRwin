# ASRwin 复现结果总结

> 路线：sherpa-onnx + streaming Zipformer (RNN-T)，WSL2 Ubuntu 22.04
> 硬件：RTX 5060 Ti 8GB / Windows 11 26100 / CPU 推理 4 线程
> 日期：2026-05-13

---

## 1. 精度复现（核心结果）

| 模型 | 数据集 | 我们 | 论文/官方 | 误差 | 状态 |
|---|---|---|---|---|---|
| **zipformer-en** streaming | LibriSpeech test-clean (2620 utt, 5.40h) | **WER 3.24%** | 3.30% | -0.06 | ✅ 复现 |
| **zipformer-bi** streaming | AISHELL-1 test (7176 utt, 10.03h) | **CER 4.16%** | 4.60% | -0.44 | ✅ 复现 |

**两项均在 ±0.5 目标内**。中文 CER 实际比论文还好 0.44 个百分点（可能是 jiwer 与官方 CER 计算的归一化差异，但属正常波动）。

## 2. 延迟与吞吐

| 指标 | zipformer-bi | zipformer-en | 备注 |
|---|---|---|---|
| **RTF** (CPU 4 线程) | 0.050 | 0.054 | ≈ 20× 实时 |
| 解码全集耗时 | 30.2 min / 10h | 17.6 min / 5.4h | 单机 CPU |
| **流式首字延迟** | ~1118ms | — | 含算法 look-ahead，从音频开始到首字 |
| chunk 提交后响应延迟 | ~18ms | — | 算法外的纯响应 |
| 模型加载时间 | 2.8s | — | cold start |

## 3. 流式行为演示（M6）

实测 `models/.../test_wavs/0.wav` (10.05s 中英混读)：

```
[ 1.10s] 昨                              ← 首字出现 (1.1s)
[ 1.40s] 昨天
[ 2.40s] 昨天是 MON
[ 2.70s] 昨天是 MONDAY
[ 4.60s] 昨天是 MONDAY TODAY
[ 5.60s] 昨天是 MONDAY TODAY IS
...
[10.05s] 昨天是 MONDAY TODAY IS LIBR THE DAY AFTER TOMORROW是星期三  ← final
```

墙钟 10.14s ≈ 音频 10.05s，**实时不卡顿**。Partial 假设逐 chunk 成长，是真正的 streaming behavior。

## 4. 复现条件

- WSL2 Ubuntu 22.04, conda env `asr` (Python 3.10)
- sherpa-onnx 1.13.1 (pip, CPU onnxruntime)
- 模型：HuggingFace `csukuangfj/sherpa-onnx-streaming-zipformer-{bilingual-zh-en-2023-02-20,en-2023-06-26}`（hf-mirror.com 中转）
- 数据：OpenSLR-12 (LibriSpeech test-clean) + OpenSLR-33 (AISHELL-1)
- 文本归一化：英文小写 + 标点统一；中文去标点 + 去空白；评测用 jiwer
- 解码方法：greedy_search；feature_dim=80；sample_rate=16000

## 5. 关键决策回顾

| 决策 | 替代方案 | 选 sherpa-onnx 的原因 |
|---|---|---|
| sherpa-onnx 不用 PyTorch/k2 | icefall + PyTorch nightly | 绕开 RTX 5060 Ti (Blackwell sm_120) 上 PyTorch / k2 自定义 kernel 兼容性坑 |
| Zipformer streaming 不用 MoChA | MoChA / Transducer | MoChA 2017 已被淘汰，Zipformer 是 2024 SOTA 流式架构 |
| CPU 推理不用 GPU | onnxruntime-gpu | CPU RTF=0.05 已经 20× 实时，无须 GPU |
| WSL2 不用原生 Windows | Windows 原生 Python | 工具链统一，apt 装包方便 |

## 6. 输出产物

```
D:\Winprojects\ASRwin\
├── PLAN.md                              ← 原始规划
├── RESULTS.md                           ← 本文件
├── models\
│   ├── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20\  (591MB)
│   └── sherpa-onnx-streaming-zipformer-en-2023-06-26\               (~800MB)
├── data\
│   ├── LibriSpeech\test-clean\          (2620 FLAC, 356MB)
│   └── data_aishell\
│       ├── transcript\aishell_transcript_v0.8.txt
│       └── wav\test\S07XX|S09XX\*.wav   (7176 wav, ~900MB)
├── scripts\
│   ├── smoke_decode.py                  ← M3a 烟雾测试
│   ├── bench_latency.py                 ← M4 延迟基准（含实时模拟）
│   ├── bench_accuracy.py                ← M5 精度基准（LibriSpeech + AISHELL）
│   ├── demo_streaming.py                ← M6 实时流式演示
│   └── extract_aishell_test.sh          ← AISHELL test 20 speaker 解压
└── results\
    ├── latency_zipformer-bi_100ms.csv
    ├── accuracy_zipformer-en_librispeech.{csv,txt}
    └── accuracy_zipformer-bi_aishell.{csv,txt}
```

## 7. 可扩展方向

- **更低延迟 chunk**：测 zipformer-en 的 `chunk-16-left-64` 变体（更短 left context）
- **GPU 推理**：换 `onnxruntime-gpu` + CUDA EP，测 GPU RTF 与首包延迟
- **加入 FunASR Paraformer / WeNet U2++** 横向对比（hf-mirror 上 paraformer repo 401，需备选源）
- **端到端语音 LLM**：8GB 卡上 Qwen2-Audio 4-bit 量化（独立路线，PLAN §备选）
- **真实麦克风 demo**：换有麦克风的机器，用 `sounddevice` 接 `OnlineRecognizer`
