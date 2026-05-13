"""
Qwen2.5-Omni-7B 4-bit 推理。Omni 比 Qwen2-Audio 大，要更小心 VRAM。

用法:
    python scripts/qwen25_omni_infer.py audio.wav --mode {asr,qa} [--question "..."]
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import librosa
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-Omni-7B"


def build_model():
    print(f"[*] loading processor from {MODEL_DIR}")
    proc = Qwen2_5OmniProcessor.from_pretrained(str(MODEL_DIR))
    print("[*] loading model (4-bit + max_memory=7GiB)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    t = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), quantization_config=bnb_cfg,
        device_map={"": 0}, torch_dtype=torch.float16,
        max_memory={0: "7GiB"},
    )
    model.eval()
    print(f"[+] loaded in {time.perf_counter()-t:.1f}s, VRAM: {torch.cuda.memory_allocated()//1024//1024} MiB")
    return proc, model


def run(audio_path: Path, mode: str, question: str | None, proc, model):
    audio, sr = librosa.load(str(audio_path), sr=16000)
    print(f"[*] audio: {audio_path.name} ({len(audio)/sr:.2f}s)")

    if mode == "asr":
        sys_prompt = "Transcribe the user's audio verbatim. Output only the text."
        prompt = "Transcribe this audio."
    else:
        sys_prompt = "You are a helpful assistant. Listen and respond in the audio's language."
        prompt = question or "Describe this audio in detail."

    conv = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    print("[*] generating...")
    t0 = time.perf_counter()
    with torch.no_grad():
        # Omni 默认会生成语音 token2wav, 这里只要文本
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                              return_audio=False)
    elapsed = time.perf_counter() - t0
    # Omni 的 generate 可能返回 tuple (text_ids, audio_waveform)
    if isinstance(out, tuple):
        out_ids = out[0]
    else:
        out_ids = out
    gen = out_ids[:, inputs["input_ids"].shape[1]:]
    resp = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    print(f"\n{'='*70}")
    print(f"  Model      : Qwen2.5-Omni-7B 4-bit")
    print(f"  Mode       : {mode}")
    print(f"  Prompt     : {prompt}")
    print(f"  Response   : {resp}")
    print(f"  Gen time   : {elapsed:.1f}s  ({gen.shape[1]} tokens, {gen.shape[1]/elapsed:.1f} tok/s)")
    print(f"  VRAM peak  : {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    print(f"{'='*70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str)
    ap.add_argument("--mode", choices=["asr", "qa"], default="asr")
    ap.add_argument("--question", default=None)
    args = ap.parse_args()
    proc, model = build_model()
    run(Path(args.audio), args.mode, args.question, proc, model)


if __name__ == "__main__":
    main()
