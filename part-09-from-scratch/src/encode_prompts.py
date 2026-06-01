#!/usr/bin/env python3
"""Encode seed prompts with the GPT-2 BPE for the nano-30M sampler.

Writes prompts_encoded.json {prompts:[str], ids:[[int]]} (ids are GPT-2
0-indexed; sample.wls shifts them +1 into WL's embedding space)."""
import json
import os
import tiktoken

enc = tiktoken.get_encoding("gpt2")
prompts = [
    "The best way to learn a new language is",
    "Once upon a time, there was a",
    "Question: What is the capital of France? Answer:",
    "My favorite food is pizza because",
    "The sun is a star that",
    "Q: How do plants make food?\nA:",
]
out = {"prompts": prompts, "ids": [enc.encode_ordinary(p) for p in prompts]}
d = os.path.join(os.path.dirname(__file__), "..", "data", "prompts_encoded.json")
json.dump(out, open(d, "w"))
print("wrote", len(prompts), "encoded prompts to", os.path.normpath(d))
