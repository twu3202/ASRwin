"""
v2 plots: speech LLM (Qwen2-Audio / Qwen2.5-Omni) vs streaming ASR (sherpa-onnx).
输出:
  plots/llm_vs_streaming_cer.png       same-utt CER bar
  plots/llm_vs_streaming_speed.png     RTF / latency dual axis
  plots/llm_vs_streaming_per_utt.png   per-utt CER scatter (qwen vs sherpa)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jiwer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "savefig.dpi": 140, "savefig.bbox": "tight",
})


# 1) same-utt CER bar
def plot_cer_bar():
    cmp_csv = RESULTS / "compare_aishell_qwen_vs_sherpa.csv"
    if not cmp_csv.exists():
        print("[!] skip cer bar: compare csv missing")
        return
    refs, qwen, sherpa = [], [], []
    with cmp_csv.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["sherpa_norm"] and row["ref_norm"]:
                refs.append(row["ref_norm"])
                qwen.append(row["qwen_norm"])
                sherpa.append(row["sherpa_norm"])
    cer_q = jiwer.cer(refs, qwen) * 100
    cer_s = jiwer.cer(refs, sherpa) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    names = [f"sherpa-onnx\nZipformer (CPU, 234M)",
             f"Qwen2-Audio 4-bit\n(GPU, 7B)"]
    vals = [cer_s, cer_q]
    colors = ["#2f80ed", "#eb5757"]
    b = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.7)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.12, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("CER (%)  — lower is better")
    ax.set_ylim(0, max(vals)*1.4)
    ax.set_title(f"AISHELL-1 test (same {len(refs)} utterances)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(PLOTS / "llm_vs_streaming_cer.png")
    plt.close(fig)
    print(f"[+] llm_vs_streaming_cer.png  CER sherpa={cer_s:.2f} qwen={cer_q:.2f}")


# 2) speed comparison: RTF + latency dual axis
def plot_speed():
    # 已知数据：
    # sherpa-onnx: RTF 0.05, streaming first-token latency 18ms (compute) / ~1.1s (alg)
    # Qwen2-Audio:  RTF 0.13 (compute), 13 tok/s, non-streaming so latency = full pass
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cats = ["sherpa-onnx\n(streaming)", "Qwen2-Audio 4-bit\n(non-streaming)"]
    rtf  = [0.05, 0.13]
    # 流式 first-text latency in second (algorithmic look-ahead included)
    latency_s = [1.1, 5.0]  # qwen on 10s wav takes ~5s before answering
    x = np.arange(len(cats))
    w = 0.35
    a1 = ax.bar(x-w/2, rtf, w, color="#2f80ed", label="RTF (lower=faster)")
    ax.set_ylabel("RTF", color="#2f80ed")
    ax.tick_params(axis='y', labelcolor="#2f80ed")
    ax.set_ylim(0, 0.3)
    for bar, v in zip(a1, rtf):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.2f}",
                ha="center", va="bottom", color="#2f80ed")

    ax2 = ax.twinx()
    a2 = ax2.bar(x+w/2, latency_s, w, color="#eb5757",
                 label="time-to-first-text (s)")
    ax2.set_ylabel("Time to first text (s)", color="#eb5757")
    ax2.tick_params(axis='y', labelcolor="#eb5757")
    ax2.set_ylim(0, 6)
    for bar, v in zip(a2, latency_s):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.1f}s",
                 ha="center", va="bottom", color="#eb5757")

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("Throughput & latency  —  on a 10s utterance")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.savefig(PLOTS / "llm_vs_streaming_speed.png")
    plt.close(fig)
    print("[+] llm_vs_streaming_speed.png")


# 3) per-utt CER scatter
def plot_per_utt_scatter():
    cmp_csv = RESULTS / "compare_aishell_qwen_vs_sherpa.csv"
    if not cmp_csv.exists():
        print("[!] skip scatter: compare csv missing")
        return
    qs, ss = [], []
    with cmp_csv.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row["ref_norm"] or not row["sherpa_norm"]:
                continue
            try:
                cer_q = jiwer.cer(row["ref_norm"], row["qwen_norm"]) * 100
                cer_s = jiwer.cer(row["ref_norm"], row["sherpa_norm"]) * 100
            except Exception:
                continue
            qs.append(min(cer_q, 60))
            ss.append(min(cer_s, 60))
    qs = np.array(qs); ss = np.array(ss)
    if len(qs) == 0:
        print("[!] no scatter data")
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(ss, qs, s=14, alpha=0.4, color="#444",
               edgecolor="white", linewidth=0.3)
    lim = 60
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, label="y=x")
    n_qwen_better = int((qs < ss).sum())
    n_sherpa_better = int((ss < qs).sum())
    n_tie = int((qs == ss).sum())
    ax.set_xlabel("sherpa-onnx CER per utt (%)")
    ax.set_ylabel("Qwen2-Audio CER per utt (%)")
    ax.set_xlim(-2, lim); ax.set_ylim(-2, lim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(0.02, 0.98,
            f"N={len(qs)}\nQwen better: {n_qwen_better}\nsherpa better: {n_sherpa_better}\ntie: {n_tie}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaa"))
    ax.set_title("Per-utterance CER  —  Qwen2-Audio vs sherpa-onnx (clipped at 60%)")
    fig.savefig(PLOTS / "llm_vs_streaming_per_utt.png")
    plt.close(fig)
    print("[+] llm_vs_streaming_per_utt.png")


if __name__ == "__main__":
    plot_cer_bar()
    plot_speed()
    plot_per_utt_scatter()
