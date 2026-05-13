"""
Task C: Streaming behavior — Qwen2-Audio (growing-window) vs sherpa-onnx (true streaming)
       on the same audio. Show how each evolving hypothesis looks at every 1s checkpoint.

Qwen2-Audio is *not* a streaming model: each query needs a full forward pass over the
audio. We simulate "streaming" by feeding audio[0:t] for t = 1s, 2s, ..., T and
re-generating each time. This is the upper-bound semantic accuracy a non-streaming
model could ever achieve in a streaming setting (at huge compute cost).

Compared against sherpa-onnx OnlineRecognizer which is a true streaming model:
emits partial hypotheses incrementally with 100ms chunks, no re-decode.

Output:
  results/streaming_compare_<wav>.json   per-checkpoint hypotheses + timing
  plots/streaming_compare_<wav>.png      side-by-side timeline
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import librosa
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request

# 中文字体: 按需下载 Noto Sans SC, 装到本仓库 plots/.fonts/
def _ensure_cjk_font() -> None:
    font_dir = Path(__file__).resolve().parent.parent / "plots" / ".fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / "NotoSansSC-Regular.otf"
    if not font_path.exists():
        url = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        try:
            print(f"[*] downloading CJK font...")
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            print(f"[!] CJK font download failed: {e}")
            return
    fm.fontManager.addfont(str(font_path))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

_ensure_cjk_font()

from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import sherpa_onnx as so

ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = ROOT / "models" / "Qwen2-Audio-7B-Instruct"
SHERPA_DIR = ROOT / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"


def build_qwen():
    proc = AutoProcessor.from_pretrained(str(QWEN_DIR))
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(QWEN_DIR), quantization_config=bnb_cfg,
        device_map={"": 0}, torch_dtype=torch.float16,
        max_memory={0: "7GiB"},
    )
    model.eval()
    return proc, model


def build_sherpa():
    return so.OnlineRecognizer.from_transducer(
        tokens=str(SHERPA_DIR / "tokens.txt"),
        encoder=str(SHERPA_DIR / "encoder-epoch-99-avg-1.onnx"),
        decoder=str(SHERPA_DIR / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(SHERPA_DIR / "joiner-epoch-99-avg-1.onnx"),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu",
    )


def qwen_transcribe(proc, model, audio: np.ndarray, sr: int) -> tuple[str, float]:
    conv = [
        {"role": "system", "content": "Transcribe the audio. Output only the transcription text."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "(streamed audio)"},
            {"type": "text", "text": "Transcribe this audio."},
        ]},
    ]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.perf_counter() - t0
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip(), elapsed


def sherpa_stream(rec: so.OnlineRecognizer, samples: np.ndarray, sr: int,
                  checkpoints_sec: list[float]) -> list[dict]:
    """Feed audio at real-time pace, sample partial hypothesis at given checkpoints."""
    stream = rec.create_stream()
    chunk = int(sr * 0.1)  # 100ms chunks
    audio_sec = len(samples) / sr
    out = []
    next_cp = 0
    fed_samples = 0
    t_start = time.perf_counter()

    for k in range(0, len(samples), chunk):
        seg = samples[k:k+chunk]
        # realtime
        chunk_end = (k + len(seg)) / sr
        wait = chunk_end - (time.perf_counter() - t_start)
        if wait > 0:
            time.sleep(wait)
        stream.accept_waveform(sr, seg)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        fed_samples = k + len(seg)
        audio_time = fed_samples / sr

        # hit any pending checkpoints
        while next_cp < len(checkpoints_sec) and audio_time >= checkpoints_sec[next_cp]:
            partial = rec.get_result(stream).strip()
            wall = time.perf_counter() - t_start
            out.append({
                "audio_t": float(checkpoints_sec[next_cp]),
                "wall_t": float(wall),
                "hyp": partial,
            })
            next_cp += 1

    tail = np.zeros(int(sr * 0.5), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    final = rec.get_result(stream).strip()
    # remaining checkpoints get final
    while next_cp < len(checkpoints_sec):
        out.append({
            "audio_t": float(checkpoints_sec[next_cp]),
            "wall_t": float(time.perf_counter() - t_start),
            "hyp": final,
        })
        next_cp += 1
    out.append({"audio_t": float(audio_sec), "wall_t": float(time.perf_counter() - t_start), "hyp": final, "kind": "final"})
    return out


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() in (".flac", ".ogg"):
        a, sr = librosa.load(str(path), sr=None, mono=True)
        return a.astype(np.float32), sr
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, sr


def plot(record: dict, wav_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    qwen = record["qwen"]
    sherpa = record["sherpa"]
    audio_sec = record["audio_sec"]
    # 两个 timeline
    for i, ev in enumerate(sherpa):
        ax.scatter(ev["audio_t"], 1, s=22, color="#2f80ed", alpha=0.7)
        if ev.get("kind") == "final":
            ax.annotate(f"FINAL: {ev['hyp']}", xy=(ev["audio_t"], 1),
                        xytext=(ev["audio_t"], 1.15),
                        fontsize=8, ha="right", color="#2f80ed",
                        wrap=True)
        elif ev["hyp"]:
            ax.annotate(ev["hyp"], xy=(ev["audio_t"], 1),
                        xytext=(ev["audio_t"], 1.05),
                        fontsize=7, ha="left", color="#2f80ed", rotation=20)
    for ev in qwen:
        ax.scatter(ev["audio_t"], 0, s=22, color="#eb5757", alpha=0.7)
        if ev["hyp"]:
            ax.annotate(ev["hyp"], xy=(ev["audio_t"], 0),
                        xytext=(ev["audio_t"], -0.18),
                        fontsize=7, ha="left", color="#eb5757", rotation=-20)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Qwen2-Audio\n(growing window)", "sherpa-onnx\n(true streaming)"])
    ax.set_xlabel("Audio time (s)")
    ax.set_xlim(-0.5, audio_sec + 1.5)
    ax.set_ylim(-1.2, 1.8)
    ax.set_title(f"Streaming behavior on {wav_name} — Qwen (red) vs sherpa-onnx (blue)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=str)
    ap.add_argument("--step", type=float, default=1.0, help="checkpoint 间隔 (秒)")
    ap.add_argument("--qwen-min", type=float, default=1.0,
                    help="Qwen 起始扫描时刻（< 1s 时上下文太少，意义不大）")
    args = ap.parse_args()

    wav = Path(args.wav)
    samples, sr = load_audio(wav)
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio_sec = len(samples) / sr
    print(f"[*] {wav.name}: {audio_sec:.2f}s @ 16kHz")

    checkpoints = list(np.arange(args.qwen_min, audio_sec + 0.001, args.step))
    # 加上 final
    if checkpoints and checkpoints[-1] < audio_sec - 0.1:
        checkpoints.append(audio_sec)

    print(f"[*] checkpoints: {checkpoints}")

    # 1) sherpa-onnx streaming (realtime sim) — 一遍跑完，定时抽 partial
    print("[*] sherpa-onnx streaming...")
    rec = build_sherpa()
    sherpa_events = sherpa_stream(rec, samples, sr, checkpoints)

    # 2) Qwen growing window — 每个 checkpoint 重新 generate
    print("[*] Qwen2-Audio loading (4-bit)...")
    proc, model = build_qwen()
    qwen_events = []
    for cp in checkpoints:
        sub = samples[:int(cp * sr)]
        hyp, gen_sec = qwen_transcribe(proc, model, sub, sr)
        qwen_events.append({"audio_t": float(cp), "gen_sec": float(gen_sec), "hyp": hyp})
        print(f"  Qwen @ {cp:.1f}s ({gen_sec:.1f}s gen): {hyp}")
    # final
    hyp, gen_sec = qwen_transcribe(proc, model, samples, sr)
    qwen_events.append({"audio_t": float(audio_sec), "gen_sec": float(gen_sec), "hyp": hyp, "kind": "final"})
    print(f"  Qwen FINAL ({gen_sec:.1f}s gen): {hyp}")

    record = {
        "wav": wav.name,
        "audio_sec": float(audio_sec),
        "checkpoints": [float(x) for x in checkpoints],
        "qwen": qwen_events,
        "sherpa": sherpa_events,
    }
    out_json = ROOT / "results" / f"streaming_compare_{wav.stem}.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] {out_json}")

    out_png = ROOT / "plots" / f"streaming_compare_{wav.stem}.png"
    plot(record, wav.name, out_png)
    print(f"[+] {out_png}")


if __name__ == "__main__":
    main()
