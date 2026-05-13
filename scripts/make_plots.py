"""
生成 README 用的 4 张图，输出到 plots/:
  1. accuracy_vs_paper.png   - WER/CER 柱状对比
  2. wer_hist_librispeech.png - LibriSpeech 逐句 WER 直方图
  3. cer_hist_aishell.png    - AISHELL 逐句 CER 直方图
  4. rtf_vs_audio_len.png    - RTF 与音频长度散点
运行： python scripts/make_plots.py
"""
from __future__ import annotations
import csv
import time
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_latency import MODELS, build_recognizer  # noqa: E402

try:
    import jiwer
except ImportError:
    raise SystemExit("需要 jiwer: pip install jiwer")

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS = ROOT / "results"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})


# ---------- 1. accuracy bar chart ----------

def plot_accuracy_bar():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["LibriSpeech\ntest-clean (WER)", "AISHELL-1\ntest (CER)"]
    ours   = [3.24, 4.16]
    paper  = [3.30, 4.60]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, paper, w, label="Paper / Official",
                color="#9aa5b1", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, ours, w, label="ASRwin (ours)",
                color="#2f80ed", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Streaming Zipformer  —  ours vs paper")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(paper + ours) * 1.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                    f"{h:.2f}%", ha="center", va="bottom", fontsize=10)
    out = PLOTS_DIR / "accuracy_vs_paper.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


# ---------- 2 + 3. per-utt histograms ----------

def _per_utt_scores(csv_path: Path, metric_fn):
    scores = []
    with csv_path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ref, hyp = row["ref_norm"], row["hyp_norm"]
            if not ref:
                continue
            try:
                s = metric_fn(ref, hyp) * 100
            except Exception:
                continue
            scores.append(min(s, 100.0))  # clip outliers
    return np.array(scores)


def plot_per_utt_hist(csv_name: str, metric_fn, metric_label: str, color: str, title: str, out_name: str):
    csv_path = RESULTS / csv_name
    if not csv_path.exists():
        print(f"[!] skip {csv_name}: not found")
        return
    scores = _per_utt_scores(csv_path, metric_fn)
    if len(scores) == 0:
        print(f"[!] {csv_name}: no rows")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, max(20, np.percentile(scores, 99)), 40)
    ax.hist(scores, bins=bins, color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
    median = float(np.median(scores))
    mean = float(np.mean(scores))
    ax.axvline(mean, color="crimson", linestyle="--", linewidth=1.2, label=f"mean {mean:.2f}%")
    ax.axvline(median, color="black", linestyle=":", linewidth=1.2, label=f"median {median:.2f}%")
    ax.set_xlabel(f"{metric_label} per utterance (%)")
    ax.set_ylabel("Utterance count")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # 0% 占大头时单独标注
    zero_frac = float((scores == 0).mean()) * 100
    ax.text(0.98, 0.96, f"N={len(scores)}\n0% rate: {zero_frac:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    out = PLOTS_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


# ---------- 4. RTF scatter (mini bench) ----------

def _read_wave_or_flac(p: Path):
    if p.suffix.lower() == ".flac":
        import soundfile as sf
        d, sr = sf.read(str(p), dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1).astype(np.float32)
        return d, sr
    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def plot_rtf_scatter(n_per_set: int = 200):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sets = []
    libri_dir = ROOT / "data" / "LibriSpeech" / "test-clean"
    if libri_dir.exists():
        flacs = list(libri_dir.rglob("*.flac"))[:n_per_set]
        sets.append(("LibriSpeech", flacs, "zipformer-en", "#2f80ed"))
    aishell_dir = ROOT / "data" / "data_aishell" / "wav" / "test"
    if aishell_dir.exists():
        wavs = list(aishell_dir.rglob("*.wav"))[:n_per_set]
        sets.append(("AISHELL", wavs, "zipformer-bi", "#eb5757"))

    overall = []
    for label, files, model_key, color in sets:
        if not files:
            continue
        print(f"[*] RTF mini-bench {label}: {len(files)} utt with {model_key}")
        rec = build_recognizer(model_key, num_threads=4)
        durs, rtfs = [], []
        for f in files:
            samples, sr = _read_wave_or_flac(f)
            dur = len(samples) / sr
            stream = rec.create_stream()
            t0 = time.perf_counter()
            stream.accept_waveform(sr, samples)
            tail = np.zeros(int(sr * 0.5), dtype=np.float32)
            stream.accept_waveform(sr, tail)
            stream.input_finished()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            elapsed = time.perf_counter() - t0
            durs.append(dur)
            rtfs.append(elapsed / dur if dur > 0 else 0)
        durs = np.array(durs); rtfs = np.array(rtfs)
        ax.scatter(durs, rtfs, s=14, alpha=0.55, color=color,
                   edgecolor="white", linewidth=0.3,
                   label=f"{label} ({model_key}) mean RTF={rtfs.mean():.3f}")
        overall.append(rtfs.mean())
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="RTF = 1 (realtime)")
    ax.set_xlabel("Audio length (s)")
    ax.set_ylabel("RTF  =  decode time / audio time")
    ax.set_title("Per-utterance decoding RTF (CPU, 4 threads)")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    out = PLOTS_DIR / "rtf_vs_audio_len.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


# ---------- main ----------

if __name__ == "__main__":
    plot_accuracy_bar()
    plot_per_utt_hist(
        "accuracy_zipformer-en_librispeech.csv",
        jiwer.wer, "WER", "#2f80ed",
        "LibriSpeech test-clean  —  per-utterance WER (2620 utt)",
        "wer_hist_librispeech.png",
    )
    plot_per_utt_hist(
        "accuracy_zipformer-bi_aishell.csv",
        jiwer.cer, "CER", "#eb5757",
        "AISHELL-1 test  —  per-utterance CER (7176 utt)",
        "cer_hist_aishell.png",
    )
    plot_rtf_scatter(n_per_set=200)
    print("\n[done]")
