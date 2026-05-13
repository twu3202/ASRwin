"""
Qwen2-Audio-7B-Instruct 本地 4-bit 推理 (Blackwell sm_120 验证版)。

支持两种模式:
  --mode asr   : 纯转写 (system prompt 强制要求只输出转写)
  --mode qa    : 语音问答 (默认 prompt 让模型自由回答关于音频的问题)
                 可用 --question 自定义问题

用法:
    python scripts/qwen2_audio_infer.py audio.wav --mode asr
    python scripts/qwen2_audio_infer.py audio.wav --mode qa --question "这段音频在说什么？是什么语言？"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import torch
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2-Audio-7B-Instruct"

SYSTEM_ASR = (
    "You are a speech recognition model. Transcribe the user's audio verbatim. "
    "Output only the transcription text, no explanation, no punctuation guessing beyond what's spoken."
)


def build_model(use_4bit: bool = True):
    print(f"[*] loading processor from {MODEL_DIR}")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))

    print(f"[*] loading model (4-bit={use_4bit})...")
    t = time.perf_counter()
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            quantization_config=bnb_cfg,
            device_map={"": 0},  # 全 GPU；如 OOM 改 device_map="auto" + offload
            torch_dtype=torch.float16,
            max_memory={0: "7GiB"},  # 留 ~1GB 给 activation
        )
    else:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
        )
    print(f"[+] model loaded in {time.perf_counter()-t:.1f}s, "
          f"VRAM used: {torch.cuda.memory_allocated()//1024//1024} MiB")
    return processor, model


def run(audio_path: Path, mode: str, question: str | None,
        processor, model, max_new_tokens: int = 256):
    # 重采样到 16kHz mono fp32
    audio, sr = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate)
    print(f"[*] audio: {audio_path.name} ({len(audio)/sr:.2f}s @ {sr}Hz)")

    if mode == "asr":
        prompt_text = "Transcribe this audio."
        system = SYSTEM_ASR
    else:
        prompt_text = question or "Describe this audio in detail."
        system = "You are a helpful assistant. Listen to the audio and respond in the same language as the audio."

    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"[*] generating (max_new_tokens={max_new_tokens})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_t = time.perf_counter() - t0

    # 只保留新生成 token
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    print(f"\n{'='*70}")
    print(f"  Mode      : {mode}")
    print(f"  Prompt    : {prompt_text}")
    print(f"  Response  : {response}")
    print(f"  Gen time  : {gen_t:.1f}s  ({gen_ids.shape[1]} tokens, "
          f"{gen_ids.shape[1]/gen_t:.1f} tok/s)")
    print(f"  VRAM peak : {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    print(f"{'='*70}")
    return response


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=str)
    p.add_argument("--mode", choices=["asr", "qa"], default="asr")
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--no-4bit", action="store_true", help="不量化 (要 ~15GB VRAM, 8GB 装不下)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"file not found: {audio}")

    processor, model = build_model(use_4bit=not args.no_4bit)
    run(audio, args.mode, args.question, processor, model, args.max_new_tokens)


if __name__ == "__main__":
    main()
