#!/usr/bin/env python3
"""Benchmark untrained GPT-2 medium inference in Python (HF + Metal) so we
can compare against the Wolfram Language inference time the user is seeing
on Mac mini CPU. Same architecture, same prompt, same generation budget."""

import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROMPT = ("User: hi there! Can you tell me what your name is?\n"
          "Assistant:")
MAX_NEW = 100
TEMPERATURE = 0.7
TOP_K = 40
SEED = 2026

def bench(device_str: str):
    print(f"\n=== device={device_str} ===")
    device = torch.device(device_str)
    t0 = time.time()
    tok = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device).eval()
    print(f"  load: {time.time()-t0:.2f} s")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    torch.manual_seed(SEED)

    # warm-up: 5 tokens, not counted
    with torch.no_grad():
        _ = model.generate(ids, max_new_tokens=5, do_sample=True,
                            temperature=TEMPERATURE, top_k=TOP_K,
                            pad_token_id=tok.eos_token_id)

    # timed run
    torch.manual_seed(SEED)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=TEMPERATURE, top_k=TOP_K,
                             pad_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    new_tokens = out.shape[1] - ids.shape[1]
    print(f"  generated {new_tokens} new tokens in {elapsed:.2f} s "
          f"({new_tokens/elapsed:.1f} tok/s)")
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  output: {text!r}")
    return elapsed, new_tokens

if __name__ == "__main__":
    # CPU baseline (closest analog to what Wolfram is doing)
    cpu_t, cpu_n = bench("cpu")
    # Metal Performance Shaders (the Apple Silicon GPU path)
    if torch.backends.mps.is_available():
        mps_t, mps_n = bench("mps")
        print(f"\n=== summary ===")
        print(f"  CPU: {cpu_n/cpu_t:.1f} tok/s")
        print(f"  MPS: {mps_n/mps_t:.1f} tok/s   ({(cpu_t/cpu_n) / (mps_t/mps_n):.1f}x speedup)")
