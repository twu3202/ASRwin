"""
Task A: Qwen2-Audio-7B 4-bit 在 AISHELL-1 test 子集上的 CER 评测，
       与 sherpa-onnx Zipformer 的结果在同一批 utt 上对比。

用法:
    python scripts/bench_qwen_aishell.py [--n 500] [--max-new-tokens 64] [--seed 0]

输出:
    results/accuracy_qwen2audio_aishell.csv     逐句结果
    results/accuracy_qwen2audio_aishell.txt     汇总
    results/compare_aishell_qwen_vs_sherpa.csv  同样 N 条上的对照
"""
from __future__ import annotations
import argparse
import csv
import random
import re
import time
from pathlib import Path

import librosa
import torch
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import jiwer

ROOT = Path(__file__).resolve().parent.parent
QWEN2_DIR = ROOT / "models" / "Qwen2-Audio-7B-Instruct"
OMNI_DIR  = ROOT / "models" / "Qwen2.5-Omni-7B"
WAV_ROOT = ROOT / "data" / "data_aishell" / "wav" / "test"
TRANS_FILE = ROOT / "data" / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
SHERPA_CSV = ROOT / "results" / "accuracy_zipformer-bi_aishell.csv"

# 中文归一化 (同 bench_accuracy)
def normalize_zh(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？、；：""''（）《》【】,.!?;:\"'()<>\[\]·…—\-]", "", s)
    # 去常见标点
    s = re.sub(r"[、，。！？；：「」『』]", "", s)
    return s.strip().lower()


def build_model(which: str = "qwen2audio"):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    if which == "qwen2audio":
        print(f"[*] loading Qwen2-Audio (4-bit) from {QWEN2_DIR}")
        proc = AutoProcessor.from_pretrained(str(QWEN2_DIR))
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(QWEN2_DIR), quantization_config=bnb_cfg,
            device_map={"": 0}, torch_dtype=torch.float16,
            max_memory={0: "7GiB"},
        )
    elif which == "omni":
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        print(f"[*] loading Qwen2.5-Omni (4-bit) from {OMNI_DIR}")
        proc = Qwen2_5OmniProcessor.from_pretrained(str(OMNI_DIR))
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(OMNI_DIR), quantization_config=bnb_cfg,
            device_map={"": 0}, torch_dtype=torch.float16,
            max_memory={0: "7GiB"},
        )
    else:
        raise ValueError(which)
    model.eval()
    return proc, model


def load_transcripts():
    trans = {}
    with TRANS_FILE.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                trans[parts[0]] = parts[1]
    return trans


def load_sherpa_hyps():
    """读 sherpa-onnx 在 AISHELL 上的逐句结果，返回 {utt: hyp_norm}"""
    if not SHERPA_CSV.exists():
        print(f"[!] {SHERPA_CSV} 不存在，跳过对照")
        return {}
    sherpa = {}
    with SHERPA_CSV.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sherpa[row["utt_id"]] = row.get("hyp_norm", "")
    return sherpa


def gen_one(proc, model, audio_path: Path) -> tuple[str, float]:
    sr_target = getattr(proc, "feature_extractor", None)
    sr_target = sr_target.sampling_rate if sr_target is not None else 16000
    audio, sr = librosa.load(str(audio_path), sr=sr_target)
    conversation = [
        {"role": "system", "content": "You are a Chinese speech recognition system. Output ONLY the transcription text in Simplified Chinese. No punctuation, no English, no explanation."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": "请转写这段中文音频。只输出汉字内容，不要任何标点或英文。"},
        ]},
    ]
    text = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        # Omni 需要 return_audio=False 才返回纯文本 tensor
        try:
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 return_audio=False)
        except TypeError:
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.perf_counter() - t0
    if isinstance(out, tuple):
        out = out[0]
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    response = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return response, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="抽样大小")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["qwen2audio", "omni"], default="qwen2audio")
    p.add_argument("--tag", default=None, help="结果文件后缀 (默认 = --model)")
    args = p.parse_args()
    tag = args.tag or args.model

    trans = load_transcripts()
    sherpa = load_sherpa_hyps()

    # 收集 test wav 列表
    wavs = sorted(WAV_ROOT.rglob("*.wav"))
    print(f"[*] AISHELL test wav 总数: {len(wavs)}")
    random.seed(args.seed)
    sample = random.sample(wavs, min(args.n, len(wavs)))
    sample.sort(key=lambda p: p.stem)  # 排序便于断点续跑
    print(f"[*] 采样 {len(sample)} 条 (seed={args.seed})")

    proc, model = build_model(args.model)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"accuracy_{tag}_aishell.csv"
    cmp_path = out_dir / f"compare_aishell_{tag}_vs_sherpa.csv"

    refs, hyps_q, hyps_s = [], [], []
    audio_total = 0.0
    gen_total = 0.0

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv, \
         cmp_path.open("w", newline="", encoding="utf-8") as fcmp:
        wcsv = csv.writer(fcsv)
        wcmp = csv.writer(fcmp)
        wcsv.writerow(["utt_id", "ref", "hyp_qwen_raw", "ref_norm", "hyp_qwen_norm", "gen_sec"])
        wcmp.writerow(["utt_id", "ref_norm", "qwen_norm", "sherpa_norm"])

        for i, wav in enumerate(sample, 1):
            utt = wav.stem
            if utt not in trans:
                continue
            ref = trans[utt]
            try:
                hyp_raw, gen_sec = gen_one(proc, model, wav)
            except Exception as e:
                print(f"  [{i}/{len(sample)}] ERROR {utt}: {e}")
                continue

            ref_n = normalize_zh(ref)
            hyp_n = normalize_zh(hyp_raw)
            sherpa_n = sherpa.get(utt, "")

            refs.append(ref_n)
            hyps_q.append(hyp_n)
            if sherpa_n:
                hyps_s.append(sherpa_n)

            # 估算音频长度（不重复 load）
            audio_sec = librosa.get_duration(path=str(wav))
            audio_total += audio_sec
            gen_total += gen_sec

            wcsv.writerow([utt, ref, hyp_raw, ref_n, hyp_n, f"{gen_sec:.3f}"])
            wcmp.writerow([utt, ref_n, hyp_n, sherpa_n])

            if i % 25 == 0 or i == len(sample):
                cur_cer = jiwer.cer(refs, hyps_q) * 100
                rtf = gen_total / audio_total if audio_total else 0
                print(f"  [{i}/{len(sample)}] cur CER={cur_cer:.2f}% RTF={rtf:.2f} "
                      f"audio={audio_total:.0f}s gen={gen_total:.0f}s")
                fcsv.flush(); fcmp.flush()

    cer_q = jiwer.cer(refs, hyps_q) * 100
    model_label = {"qwen2audio": "Qwen2-Audio 4-bit", "omni": "Qwen2.5-Omni 4-bit"}[args.model]
    line_q = f"{model_label}  CER = {cer_q:.2f}%  on {len(refs)} utt"
    print("\n" + "="*60)
    print(line_q)
    line_cmp = ""
    if hyps_s:
        # 严格 same-utt 对照：从 hyps_q/refs 中匹配 sherpa 的 utt
        aligned_refs, aligned_q, aligned_s = [], [], []
        with cmp_path.open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["sherpa_norm"]:
                    aligned_refs.append(row["ref_norm"])
                    aligned_q.append(row["qwen_norm"])
                    aligned_s.append(row["sherpa_norm"])
        if aligned_refs:
            cer_q_match = jiwer.cer(aligned_refs, aligned_q) * 100
            cer_s_match = jiwer.cer(aligned_refs, aligned_s) * 100
            line_cmp = (
                f"On same {len(aligned_refs)} utt with sherpa hyps available:\n"
                f"  {model_label} CER  : {cer_q_match:.2f}%\n"
                f"  sherpa-onnx Zipformer : {cer_s_match:.2f}%"
            )
            print(line_cmp)
        else:
            line_cmp = "(no overlapping utterances with sherpa CSV)"

    rtf = gen_total / audio_total if audio_total else 0
    summary = (
        f"=== {model_label} on AISHELL-1 test (subset N={len(refs)}) ===\n"
        f"  audio total : {audio_total:.1f}s ({audio_total/3600:.2f}h)\n"
        f"  gen total   : {gen_total:.1f}s\n"
        f"  RTF         : {rtf:.3f}\n"
        f"  {line_q}\n"
    )
    if line_cmp:
        summary += "\n" + line_cmp + "\n"
    (out_dir / f"accuracy_{tag}_aishell.txt").write_text(summary, encoding="utf-8")
    print(f"\n[+] CSV: {csv_path}")
    print(f"[+] CMP: {cmp_path}")


if __name__ == "__main__":
    main()
