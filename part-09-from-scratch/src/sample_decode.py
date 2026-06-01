#!/usr/bin/env python3
"""Detokenise nano-30M generations from sample.wls with the GPT-2 BPE.

Reads generations.json {prompts:[str], generations:[[int(gpt2 0-idx)]]} and
prints each decoded continuation."""
import json
import sys
import tiktoken

enc = tiktoken.get_encoding("gpt2")
data = json.load(open(sys.argv[1]))
for i, ids in enumerate(data["generations"]):
    ids = [int(t) for t in ids if 0 <= int(t) < enc.n_vocab]
    print(f"\n=== sample {i+1} (prompt: {data['prompts'][i]!r}) ===")
    print(enc.decode(ids))
