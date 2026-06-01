#!/usr/bin/env python3
"""Prepare Alpaca instruction data for SFT of the Part-9 nano model.

Downloads tatsu-lab/alpaca, formats each example with a FIXED instruction
template, appends the GPT-2 end-of-text token (50256) after the response so the
model learns to STOP (the base model has no learned EOS -- training-gotchas #9),
tokenises with the GPT-2 BPE, concatenates into a flat uint16 stream (+1 shifted
into WL's 1-indexed space, exactly like preprocess_fineweb.py), and writes it as
a .bin that the existing sharder/trainer consume unchanged.

The SAME template must be used at inference time (chat.wls) -- it is duplicated
there.  Keep them in sync.

Usage:
  python3 prepare_alpaca.py --out ../data/alpaca_sft.bin
"""
import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset

EOT_RAW = 50256
VOCAB = 50257
SHIFT = 1

# Fixed SFT template.  Mirror this exactly in chat.wls.
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "alpaca_sft.bin"))
    ap.add_argument("--max-examples", type=int, default=0,
                    help="0 = all (~52k)")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    toks = []
    n_used = 0
    for ex in ds:
        instr = ex["instruction"].strip()
        inp = ex.get("input", "").strip()
        out = ex["output"].strip()
        if not instr or not out:
            continue
        if inp:
            text = PROMPT_WITH_INPUT.format(instruction=instr, input=inp) + out
        else:
            text = PROMPT_NO_INPUT.format(instruction=instr) + out
        ids = enc.encode_ordinary(text)
        ids.append(EOT_RAW)          # learned end-of-turn
        toks.extend(ids)
        n_used += 1

    arr = (np.asarray(toks, dtype=np.uint16) + SHIFT)
    assert arr.min() >= 1 and arr.max() <= VOCAB, (arr.min(), arr.max())
    arr.tofile(args.out)
    print(f"wrote {len(arr):,} tokens from {n_used:,} examples to "
          f"{os.path.normpath(args.out)} ({arr.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
