#!/usr/bin/env python3
"""Generate a curated set of conversation examples from the trained
GPT-2 medium SFT model (via the HF bridge + MPS) for embedding into
LLM_A_to_Z.nb Chapter 10.

Output: discussion-model/data/conversation_examples.json
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "data", "hf_model")
OUT_PATH = os.path.join(HERE, "..", "data", "conversation_examples.json")

PROMPTS = [
    "What is the capital of France?",
    "Give me three tips for staying healthy.",
    "Who wrote the play Hamlet?",
    "Recommend a book to read.",
    "Explain photosynthesis in two sentences.",
    "What is the difference between machine learning and deep learning?",
    "Name three famous physicists.",
    "Write a haiku about autumn.",
    "What is the boiling point of water at sea level?",
    "How do bees make honey?",
    "Suggest a creative use for an old smartphone.",
    "Why is the sky blue?",
]

MAX_NEW = 100
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.95
SEED = 2026

STOP_STRINGS = [" You:", " User:", "\nUser:", " Assistant:", "\nAssistant:"]


def truncate(text: str) -> str:
    """Trim at first stop string and collapse the t-t-t tail."""
    cut = len(text)
    for s in STOP_STRINGS:
        idx = text.find(s)
        if 0 <= idx < cut:
            cut = idx
    out = text[:cut].rstrip()
    # collapse runs of " t" repetition that occasionally still slip through
    while out.endswith(" t"):
        out = out[:-2].rstrip()
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading model from {MODEL_DIR} on {device}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16).to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f} s; "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    torch.manual_seed(SEED)
    results = []
    for prompt in PROMPTS:
        full = f"User: {prompt}\nAssistant:"
        ids = tok(full, return_tensors="pt").input_ids.to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=True,
                temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
                pad_token_id=tok.eos_token_id)
        elapsed = time.time() - t0
        new_tokens = out.shape[1] - ids.shape[1]
        raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        clean = truncate(raw)
        print(f"\n  User: {prompt}")
        print(f"  Assistant: {clean}")
        print(f"  [{elapsed:.1f}s, {new_tokens} tok, {new_tokens/elapsed:.1f} tok/s]")
        results.append({
            "user": prompt,
            "assistant": clean,
            "raw": raw,
            "elapsed_s": round(elapsed, 2),
            "tokens": int(new_tokens),
            "tok_per_s": round(new_tokens / elapsed, 1),
        })

    with open(OUT_PATH, "w") as f:
        json.dump({"prompts": PROMPTS, "examples": results,
                   "model_dir": MODEL_DIR, "device": device,
                   "temperature": TEMPERATURE, "top_k": TOP_K,
                   "top_p": TOP_P, "max_new": MAX_NEW},
                  f, indent=2)
    print(f"\nsaved {len(results)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
