"""
M6 流式 ASR 实时演示（无需麦克风）。

读一个 wav 文件，按真实采样率喂给流式识别器，逐 chunk 打印 partial 假设，
最后打印 final。直观看到 streaming ASR 的"边听边出字"。

用法:
    python scripts/demo_streaming.py path/to.wav --model zipformer-bi [--chunk-ms 100]

按 Ctrl+C 提前结束。
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_latency import MODELS, build_recognizer  # noqa: E402

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() in (".flac", ".ogg"):
        if sf is None:
            raise SystemExit("需要 soundfile: pip install soundfile")
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        return data, sr
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, sr


# ANSI 颜色
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=str, help="wav/flac 文件路径")
    p.add_argument("--model", default="zipformer-bi", choices=list(MODELS))
    p.add_argument("--chunk-ms", type=int, default=100,
                   help="模拟麦克风一次回调的长度 (ms)")
    p.add_argument("--no-realtime", action="store_true",
                   help="不模拟实时；尽可能快地解码")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise SystemExit(f"找不到 {wav_path}")

    print(f"{DIM}加载模型 {args.model}...{RESET}", flush=True)
    rec = build_recognizer(args.model, num_threads=args.threads)

    samples, sr = load_audio(wav_path)
    duration = len(samples) / sr
    print(f"{DIM}音频 {wav_path.name}: {duration:.2f}s @ {sr}Hz, chunk={args.chunk_ms}ms, "
          f"realtime={not args.no_realtime}{RESET}\n", flush=True)
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}流式识别开始 (青色=partial, 绿色=final){RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n", flush=True)

    stream = rec.create_stream()
    chunk = max(1, int(sr * args.chunk_ms / 1000))
    n_chunks = (len(samples) + chunk - 1) // chunk

    last_partial = ""
    first_text_t: float | None = None
    t_start = time.perf_counter()

    try:
        for k in range(n_chunks):
            seg = samples[k * chunk:(k + 1) * chunk]
            # 实时模式：等到该 chunk 末尾"应到达"的时刻
            if not args.no_realtime:
                target = (k + 1) * chunk / sr
                now = time.perf_counter() - t_start
                wait = target - now
                if wait > 0:
                    time.sleep(wait)

            stream.accept_waveform(sr, seg)
            while rec.is_ready(stream):
                rec.decode_stream(stream)

            partial = rec.get_result(stream).strip()
            audio_time = (k + 1) * chunk / sr

            if partial != last_partial:
                if first_text_t is None and partial:
                    first_text_t = time.perf_counter() - t_start
                # 同一行刷新：[音频时间] partial
                sys.stdout.write(
                    f"{CLEAR_LINE}{DIM}[{audio_time:5.2f}s]{RESET} "
                    f"{CYAN}{partial}{RESET}")
                sys.stdout.flush()
                last_partial = partial

        # 收尾
        tail = np.zeros(int(sr * 0.5), dtype=np.float32)
        stream.accept_waveform(sr, tail)
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        final = rec.get_result(stream).strip()

        total_t = time.perf_counter() - t_start
        sys.stdout.write(f"{CLEAR_LINE}{DIM}[{duration:5.2f}s]{RESET} "
                         f"{GREEN}{final}{RESET}\n\n")
        print(f"{CYAN}{'='*70}{RESET}")
        print(f"  音频时长   : {duration:.2f}s")
        print(f"  墙钟耗时   : {total_t:.2f}s")
        if first_text_t is not None:
            print(f"  首字出现   : {first_text_t*1000:.0f}ms (墙钟)")
        print(f"  RTF (~纯算): {(total_t - max(0, duration if not args.no_realtime else 0))/duration:.3f}")
        print(f"{CYAN}{'='*70}{RESET}")

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[中断]{RESET} 当前: {rec.get_result(stream)}")


if __name__ == "__main__":
    main()
