"""
M5 精度基准：CER (中文) / WER (英文)。

支持数据集:
  --dataset librispeech --data DIR    DIR 指向 LibriSpeech/test-clean 等
  --dataset aishell     --data DIR    DIR 指向 data_aishell/wav/test/  (需 transcript)
                                       transcript 路径用 --aishell-trans 指定 (默认自动找)

用法示例:
  python scripts/bench_accuracy.py \\
      --model zipformer-en --dataset librispeech \\
      --data data/LibriSpeech/test-clean

  python scripts/bench_accuracy.py \\
      --model zipformer-bi --dataset aishell \\
      --data data/data_aishell/wav/test \\
      --aishell-trans data/data_aishell/transcript/aishell_transcript_v0.8.txt

输出:
  results/accuracy_<model>_<dataset>.csv  逐句结果
  results/accuracy_<model>_<dataset>.txt  汇总
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx as so

# 复用 bench_latency.py 的模型注册表与构建函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_latency import MODELS, build_recognizer  # noqa: E402

try:
    import soundfile as sf  # FLAC 读取
except ImportError:
    sf = None

try:
    import jiwer
except ImportError:
    raise SystemExit("缺 jiwer，请: pip install jiwer")

ROOT = Path(__file__).resolve().parent.parent


# -------------- 文本归一化 --------------

_EN_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_en(s: str) -> str:
    s = s.lower()
    s = _EN_PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def normalize_zh(s: str) -> str:
    # 去掉所有空白、英文标点、中文标点，只保留中英文字符与数字
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？、；：""''（）《》【】,.!?;:\"'()<>\[\]]", "", s)
    return s.strip().lower()


def cer(ref: str, hyp: str) -> float:
    # 中文按字符切，英文单词保留为单字符占位 (常见做法)
    def split_chars(t: str):
        return list(t)
    return jiwer.cer(ref, hyp)


def wer(ref: str, hyp: str) -> float:
    return jiwer.wer(ref, hyp)


# -------------- 数据加载 --------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() in (".flac", ".ogg"):
        if sf is None:
            raise RuntimeError("需要 soundfile 读 flac: pip install soundfile")
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        return data, sr
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, f"{path} 非单声道"
        assert wf.getsampwidth() == 2, f"{path} 非 16-bit"
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def iter_librispeech(root: Path):
    """yield (utt_id, audio_path, transcript)"""
    for trans_file in root.rglob("*.trans.txt"):
        with trans_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                flac = trans_file.parent / f"{utt_id}.flac"
                if flac.exists():
                    yield utt_id, flac, text


# AISHELL-1 官方 test 划分: 20 speakers, ~7176 utt
AISHELL_TEST_SPEAKERS = {
    "S0764", "S0765", "S0766", "S0767", "S0768", "S0769", "S0770",
    "S0901", "S0902", "S0903", "S0904", "S0905", "S0906", "S0907", "S0908",
    "S0912", "S0913", "S0914", "S0915", "S0916",
}


def iter_aishell(wav_root: Path, trans_file: Path, only_test: bool = True):
    # 加载转录到字典
    trans = {}
    with trans_file.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                trans[parts[0]] = parts[1]
    for wav in wav_root.rglob("*.wav"):
        utt = wav.stem
        if utt not in trans:
            continue
        if only_test:
            # speaker id 是 wav 父目录名 (e.g. S0764)
            speaker = wav.parent.name
            if speaker not in AISHELL_TEST_SPEAKERS:
                continue
        yield utt, wav, trans[utt]


# -------------- 解码 --------------

def decode_one(rec: so.OnlineRecognizer, samples: np.ndarray, sr: int) -> str:
    stream = rec.create_stream()
    stream.accept_waveform(sr, samples)
    tail = np.zeros(int(sr * 0.5), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return rec.get_result(stream).strip()


# -------------- 主流程 --------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--dataset", required=True, choices=["librispeech", "aishell"])
    p.add_argument("--data", required=True, help="测试集根目录")
    p.add_argument("--aishell-trans", default=None,
                   help="AISHELL transcript 路径；默认自动找")
    p.add_argument("--n", type=int, default=None, help="只跑前 N 条 (调试)")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    print(f"[*] 构建识别器 {args.model}")
    rec = build_recognizer(args.model, num_threads=args.threads)

    data_root = Path(args.data)
    if args.dataset == "librispeech":
        items = list(iter_librispeech(data_root))
        norm = normalize_en
        metric_fn = wer
        metric_name = "WER"
    else:  # aishell
        trans = args.aishell_trans
        if not trans:
            # 自动找
            cands = list(data_root.parent.parent.rglob("aishell_transcript*.txt"))
            if not cands:
                raise SystemExit("找不到 aishell_transcript_v0.8.txt，请用 --aishell-trans 指定")
            trans = cands[0]
        items = list(iter_aishell(data_root, Path(trans)))
        norm = normalize_zh
        metric_fn = cer
        metric_name = "CER"

    if args.n:
        items = items[:args.n]
    print(f"[*] 共 {len(items)} utt")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"accuracy_{args.model}_{args.dataset}.csv"
    txt_path = out_dir / f"accuracy_{args.model}_{args.dataset}.txt"

    refs, hyps = [], []
    t0 = time.perf_counter()
    audio_total = 0.0
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["utt_id", "ref", "hyp", "ref_norm", "hyp_norm"])
        for i, (utt, audio, ref) in enumerate(items, 1):
            samples, sr = load_audio(audio)
            audio_total += len(samples) / sr
            hyp_raw = decode_one(rec, samples, sr)
            ref_n, hyp_n = norm(ref), norm(hyp_raw)
            refs.append(ref_n)
            hyps.append(hyp_n)
            w.writerow([utt, ref, hyp_raw, ref_n, hyp_n])
            if i % 50 == 0 or i == len(items):
                elapsed = time.perf_counter() - t0
                rtf = elapsed / audio_total if audio_total else 0
                print(f"  [{i}/{len(items)}] elapsed={elapsed:.0f}s audio={audio_total:.0f}s RTF={rtf:.3f}")

    elapsed = time.perf_counter() - t0
    score = metric_fn(refs, hyps) * 100
    rtf = elapsed / audio_total if audio_total else 0

    summary = (
        f"=== {args.model} on {args.dataset} ===\n"
        f"  utt count   : {len(items)}\n"
        f"  audio total : {audio_total:.1f}s ({audio_total/3600:.2f}h)\n"
        f"  decode time : {elapsed:.1f}s\n"
        f"  RTF         : {rtf:.3f}\n"
        f"  {metric_name:<11}: {score:.2f}%\n"
    )
    print("\n" + summary)
    txt_path.write_text(summary, encoding="utf-8")
    print(f"[+] CSV : {csv_path}")
    print(f"[+] 汇总: {txt_path}")


if __name__ == "__main__":
    main()
