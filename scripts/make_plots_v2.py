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


# 1) same-utt CER bar — 三方对比 (sherpa / qwen2audio / omni, 取 Omni 与 Qwen2-Audio 各自对照所用的 utt 子集)
def plot_cer_bar():
    cmp_q = RESULTS / "compare_aishell_qwen2audio_vs_sherpa.csv"
    if not cmp_q.exists():
        cmp_q = RESULTS / "compare_aishell_qwen_vs_sherpa.csv"  # 旧文件名兼容
    cmp_o = RESULTS / "compare_aishell_omni_vs_sherpa.csv"

    def read_pair(path):
        if not path.exists():
            return None
        refs, hyp, sh = [], [], []
        with path.open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["sherpa_norm"] and row["ref_norm"]:
                    refs.append(row["ref_norm"])
                    hyp.append(row["qwen_norm"])
                    sh.append(row["sherpa_norm"])
        return refs, hyp, sh

    pq = read_pair(cmp_q); po = read_pair(cmp_o)
    rows = []
    if pq:
        rows.append(("sherpa-onnx\nZipformer\n(CPU, 234M)",
                     jiwer.cer(pq[0], pq[2]) * 100, len(pq[0]), "#2f80ed"))
        rows.append(("Qwen2-Audio\n4-bit (GPU, 7B)",
                     jiwer.cer(pq[0], pq[1]) * 100, len(pq[0]), "#eb5757"))
    if po:
        rows.append(("Qwen2.5-Omni\n4-bit (GPU, 7B)",
                     jiwer.cer(po[0], po[1]) * 100, len(po[0]), "#27ae60"))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names = [r[0] for r in rows]
    vals  = [r[1] for r in rows]
    cols  = [r[3] for r in rows]
    b = ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.7)
    for bar, v, r in zip(b, vals, rows):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.12,
                f"{v:.2f}%\nN={r[2]}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("CER (%)  — lower is better")
    ax.set_ylim(0, max(vals)*1.5)
    ax.set_title("AISHELL-1 test — sherpa-onnx vs Qwen2-Audio vs Qwen2.5-Omni")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.savefig(PLOTS / "llm_vs_streaming_cer.png")
    plt.close(fig)
    print(f"[+] llm_vs_streaming_cer.png  values={[round(v,2) for v in vals]}")


# 2) speed comparison: RTF + latency dual axis
def plot_speed():
    # 已知数据：
    # sherpa-onnx: RTF 0.05, streaming first-token latency 18ms (compute) / ~1.1s (alg)
    # Qwen2-Audio:  RTF 0.13 (compute), 13 tok/s, non-streaming so latency = full pass
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    cats = ["sherpa-onnx\n(streaming)",
            "Qwen2-Audio 4-bit\n(non-streaming)",
            "Qwen2.5-Omni 4-bit\n(non-streaming)"]
    rtf  = [0.05, 0.13, 0.27]
    latency_s = [1.1, 5.0, 6.0]  # full-pass time on a 10s utterance
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


# 3) per-utt CER scatter — 两张：Qwen2-Audio vs sherpa, Omni vs sherpa
def _scatter_pair(cmp_path: Path, label_y: str, color: str, out: Path):
    if not cmp_path.exists():
        print(f"[!] skip {out.name}: {cmp_path.name} missing")
        return
    qs, ss = [], []
    with cmp_path.open(encoding="utf-8") as f:
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
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(ss, qs, s=14, alpha=0.4, color=color,
               edgecolor="white", linewidth=0.3)
    lim = 60
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, label="y=x")
    n_better = int((qs < ss).sum())
    n_worse = int((ss < qs).sum())
    n_tie = int((qs == ss).sum())
    ax.set_xlabel("sherpa-onnx CER per utt (%)")
    ax.set_ylabel(f"{label_y} CER per utt (%)")
    ax.set_xlim(-2, lim); ax.set_ylim(-2, lim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(0.02, 0.98,
            f"N={len(qs)}\n{label_y} better: {n_better}\n"
            f"sherpa better: {n_worse}\ntie: {n_tie}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaa"))
    ax.set_title(f"Per-utterance CER  —  {label_y} vs sherpa-onnx (clipped at 60%)")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out.name}")


def plot_per_utt_scatter():
    for cmp_name, label, color, out_name in [
        ("compare_aishell_qwen2audio_vs_sherpa.csv", "Qwen2-Audio 4-bit", "#eb5757", "llm_vs_streaming_per_utt.png"),
        ("compare_aishell_qwen_vs_sherpa.csv",      "Qwen2-Audio 4-bit", "#eb5757", "llm_vs_streaming_per_utt.png"),
        ("compare_aishell_omni_vs_sherpa.csv",      "Qwen2.5-Omni 4-bit", "#27ae60", "llm_vs_streaming_per_utt_omni.png"),
    ]:
        cmp_path = RESULTS / cmp_name
        if cmp_path.exists():
            _scatter_pair(cmp_path, label, color, PLOTS / out_name)


if __name__ == "__main__":
    plot_cer_bar()
    plot_speed()
    plot_per_utt_scatter()
