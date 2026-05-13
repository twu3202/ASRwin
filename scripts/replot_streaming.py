"""从 results/streaming_compare_*.json 重画图，自带 CJK 字体下载。"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent.parent


def ensure_cjk_font():
    font_dir = ROOT / "plots" / ".fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / "NotoSansCJKsc-Regular.otf"
    if not font_path.exists():
        url = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        print(f"[*] downloading CJK font from {url}")
        urllib.request.urlretrieve(url, font_path)
    fm.fontManager.addfont(str(font_path))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def render(record: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    audio_sec = record["audio_sec"]

    # Y=1 sherpa (true streaming), Y=0 qwen (growing window)
    for ev in record["sherpa"]:
        ax.scatter(ev["audio_t"], 1, s=20, color="#2f80ed", alpha=0.7, zorder=3)
        h = ev["hyp"]
        if not h:
            continue
        ax.annotate(h, xy=(ev["audio_t"], 1), xytext=(ev["audio_t"], 1.08),
                    fontsize=7, ha="left", color="#2f80ed", rotation=18,
                    annotation_clip=False)

    for ev in record["qwen"]:
        ax.scatter(ev["audio_t"], 0, s=20, color="#eb5757", alpha=0.7, zorder=3)
        h = ev["hyp"]
        if not h:
            continue
        # 去掉模型的固定前缀让标注更短
        h = h.replace("The original content of this audio is: ", "")
        h = h.replace("The transcription of the audio is: ", "")
        h = h.strip(" '.")
        ax.annotate(h, xy=(ev["audio_t"], 0), xytext=(ev["audio_t"], -0.18),
                    fontsize=7, ha="left", color="#eb5757", rotation=-18,
                    annotation_clip=False)

    ax.axhline(0, color="#888", linewidth=0.5)
    ax.axhline(1, color="#888", linewidth=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Qwen2-Audio 4-bit\n(growing window,\nnon-streaming)",
                        "sherpa-onnx Zipformer\n(true streaming)"])
    ax.set_xlabel("Audio time (s) — each dot is a hypothesis sampled at that audio time")
    ax.set_xlim(-0.5, audio_sec + 2.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_title(f"Streaming behaviour on {record['wav']} ({audio_sec:.1f}s)  "
                 "—  sherpa monotone vs Qwen unstable", fontsize=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_cjk_font()
    for jpath in (ROOT / "results").glob("streaming_compare_*.json"):
        record = json.loads(jpath.read_text(encoding="utf-8"))
        out = ROOT / "plots" / f"streaming_compare_{jpath.stem.split('_')[-1]}.png"
        render(record, out)
        print(f"[+] {out}")


if __name__ == "__main__":
    main()
