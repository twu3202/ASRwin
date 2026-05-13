"""
M3a 烟雾测试：用主基线模型对一条 wav 做流式解码。
用法:
    python scripts/smoke_decode.py [wav_path]
不传参数时自动用模型仓库自带的 test_wavs 第一条。
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx as so

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"


def load_recognizer() -> so.OnlineRecognizer:
    return so.OnlineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=str(MODEL_DIR / "encoder-epoch-99-avg-1.onnx"),
        decoder=str(MODEL_DIR / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(MODEL_DIR / "joiner-epoch-99-avg-1.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "需要单声道 wav"
        assert wf.getsampwidth() == 2, "需要 16-bit PCM wav"
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def stream_decode(rec: so.OnlineRecognizer, samples: np.ndarray, sr: int,
                  chunk_ms: int = 100) -> tuple[str, float, float]:
    stream = rec.create_stream()
    chunk = int(sr * chunk_ms / 1000)
    audio_sec = len(samples) / sr

    t0 = time.perf_counter()
    for i in range(0, len(samples), chunk):
        stream.accept_waveform(sr, samples[i:i + chunk])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    # 尾部 padding 触发最终 token
    tail = np.zeros(int(sr * 0.5), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    elapsed = time.perf_counter() - t0

    text = rec.get_result(stream)
    rtf = elapsed / audio_sec
    return text, audio_sec, rtf


def main():
    print(f"[*] 加载模型 from {MODEL_DIR}")
    t = time.perf_counter()
    rec = load_recognizer()
    print(f"[+] 模型加载耗时 {time.perf_counter()-t:.2f}s")

    if len(sys.argv) > 1:
        wavs = [Path(sys.argv[1])]
    else:
        wavs = sorted((MODEL_DIR / "test_wavs").glob("*.wav"))
        if not wavs:
            print("[!] 找不到 test_wavs/，请手动传一条 wav 路径")
            sys.exit(1)

    for wav in wavs[:5]:  # 只跑前 5 条
        samples, sr = read_wave(wav)
        text, dur, rtf = stream_decode(rec, samples, sr)
        print(f"\n--- {wav.name} ({dur:.2f}s, RTF={rtf:.3f}) ---")
        print(f"识别: {text}")


if __name__ == "__main__":
    main()
