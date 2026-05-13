"""
M4 延迟基准：首包延迟 / 末包延迟 / RTF。

用法:
    python scripts/bench_latency.py [--model MODEL_KEY] [--chunk-ms 100] [--wavs DIR] [--n N]

输出 CSV: results/latency_<model>_<chunk>ms.csv
列: utt_id, audio_sec, first_token_ms, last_token_ms, rtf, decode_sec
"""
from __future__ import annotations

import argparse
import csv
import time
import wave
from pathlib import Path
from statistics import mean, median

import numpy as np
import sherpa_onnx as so

ROOT = Path(__file__).resolve().parent.parent

# 模型注册表 — 新增模型只在这里加一项
MODELS = {
    "zipformer-bi": {
        "dir": ROOT / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "encoder": "encoder-epoch-99-avg-1.onnx",
        "decoder": "decoder-epoch-99-avg-1.onnx",
        "joiner": "joiner-epoch-99-avg-1.onnx",
        "type": "transducer",
    },
    "zipformer-en": {
        "dir": ROOT / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "type": "transducer",
    },
    "paraformer-bi": {
        "dir": ROOT / "models" / "sherpa-onnx-streaming-paraformer-bilingual-zh-en-2023-11-09",
        "encoder": "encoder.onnx",
        "decoder": "decoder.onnx",
        "type": "paraformer",
    },
}


def build_recognizer(key: str, num_threads: int = 2) -> so.OnlineRecognizer:
    cfg = MODELS[key]
    d = cfg["dir"]
    if cfg["type"] == "transducer":
        return so.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / cfg["encoder"]),
            decoder=str(d / cfg["decoder"]),
            joiner=str(d / cfg["joiner"]),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
    elif cfg["type"] == "paraformer":
        return so.OnlineRecognizer.from_paraformer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / cfg["encoder"]),
            decoder=str(d / cfg["decoder"]),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu",
        )
    else:
        raise ValueError(cfg["type"])


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, f"{path} 需单声道"
        assert wf.getsampwidth() == 2, f"{path} 需 16-bit PCM"
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def bench_one(rec: so.OnlineRecognizer, samples: np.ndarray, sr: int,
              chunk_ms: int = 100, realtime: bool = True) -> dict:
    """
    流式延迟测量。

    指标定义（realtime=True 时）:
      first_ms : 第一个非空假设出现的"真实流式延迟"。
                 = 处理完产生该假设的 chunk 的墙钟时刻 - 该 chunk 末尾音频在实时输入下的到达时刻。
                 在实时输入下永远 >= 0；含算法延迟 + 计算耗时。
      last_ms  : 最后一段音频喂完到 final 假设出来的耗时（追尾延迟）。
      rtf      : 纯计算耗时 / 音频时长（不含 sleep）。

    realtime=False 时不 sleep，纯吞吐测试，first_ms / last_ms 仅反映计算耗时下限。
    """
    stream = rec.create_stream()
    chunk = max(1, int(sr * chunk_ms / 1000))
    audio_sec = len(samples) / sr

    first_ms: float | None = None
    compute_sec = 0.0
    t_start = time.perf_counter()

    n_chunks = (len(samples) + chunk - 1) // chunk
    for k in range(n_chunks):
        seg = samples[k * chunk:(k + 1) * chunk]
        # 实时模式：等到该 chunk 末尾音频"已经到达"的时刻才喂
        if realtime:
            chunk_end_audio_time = (k + 1) * chunk / sr  # 该 chunk 末尾对应的音频时间
            now = time.perf_counter() - t_start
            wait = chunk_end_audio_time - now
            if wait > 0:
                time.sleep(wait)

        c0 = time.perf_counter()
        stream.accept_waveform(sr, seg)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        compute_sec += time.perf_counter() - c0

        if first_ms is None:
            partial = rec.get_result(stream).strip()
            if partial:
                # 真实流式首包延迟 = 当前墙钟 - 该 chunk 末尾音频应到达的时刻
                now = time.perf_counter() - t_start
                chunk_end_audio_time = (k + 1) * chunk / sr
                first_ms = (now - chunk_end_audio_time) * 1000

    # 末尾 padding 触发 final
    audio_end_wall = time.perf_counter()
    tail = np.zeros(int(sr * 0.5), dtype=np.float32)
    c0 = time.perf_counter()
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    compute_sec += time.perf_counter() - c0
    last_ms = (time.perf_counter() - audio_end_wall) * 1000

    text = rec.get_result(stream).strip()
    rtf = compute_sec / audio_sec if audio_sec > 0 else float("nan")

    return {
        "audio_sec": audio_sec,
        "first_ms": first_ms if first_ms is not None else float("nan"),
        "last_ms": last_ms,
        "rtf": rtf,
        "compute_sec": compute_sec,
        "text": text,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="zipformer-bi", choices=list(MODELS))
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--wavs", type=str, default=None,
                   help="wav 目录；不传则用模型自带 test_wavs/")
    p.add_argument("--n", type=int, default=None, help="只跑前 N 条")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-realtime", action="store_true",
                   help="不模拟实时音频到达 (纯吞吐测试)")
    args = p.parse_args()

    print(f"[*] 构建识别器: {args.model} (threads={args.threads})")
    t = time.perf_counter()
    rec = build_recognizer(args.model, num_threads=args.threads)
    print(f"[+] 加载耗时 {time.perf_counter()-t:.2f}s")

    if args.wavs:
        wav_dir = Path(args.wavs)
    else:
        wav_dir = MODELS[args.model]["dir"] / "test_wavs"
    wavs = sorted(wav_dir.glob("*.wav"))
    if args.n:
        wavs = wavs[:args.n]
    if not wavs:
        raise SystemExit(f"找不到 wav 于 {wav_dir}")
    print(f"[*] 共 {len(wavs)} 条 wav，chunk={args.chunk_ms}ms")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"latency_{args.model}_{args.chunk_ms}ms.csv"

    rows = []
    realtime = not args.no_realtime
    print(f"[*] realtime simulation: {realtime}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "audio_sec", "first_ms", "last_ms", "rtf", "compute_sec", "text"])
        for i, wav in enumerate(wavs, 1):
            samples, sr = read_wave(wav)
            r = bench_one(rec, samples, sr, args.chunk_ms, realtime=realtime)
            w.writerow([wav.stem, f"{r['audio_sec']:.3f}", f"{r['first_ms']:.1f}",
                        f"{r['last_ms']:.1f}", f"{r['rtf']:.4f}",
                        f"{r['compute_sec']:.4f}", r["text"]])
            if i > 1:  # 跳过 warm-up
                rows.append(r)
            print(f"  [{i}/{len(wavs)}] {wav.name} {r['audio_sec']:.2f}s "
                  f"first={r['first_ms']:.0f}ms last={r['last_ms']:.0f}ms RTF={r['rtf']:.3f}")

    if rows:
        print("\n=== 汇总 (跳过首条 warm-up) ===")
        print(f"  N         = {len(rows)}")
        print(f"  audio total = {sum(r['audio_sec'] for r in rows):.1f}s")
        print(f"  first_ms  mean={mean(r['first_ms'] for r in rows):.1f} "
              f"med={median(r['first_ms'] for r in rows):.1f}")
        print(f"  last_ms   mean={mean(r['last_ms'] for r in rows):.1f} "
              f"med={median(r['last_ms'] for r in rows):.1f}")
        print(f"  RTF       mean={mean(r['rtf'] for r in rows):.3f} "
              f"med={median(r['rtf'] for r in rows):.3f}")
    print(f"\n[+] CSV: {out_csv}")


if __name__ == "__main__":
    main()
