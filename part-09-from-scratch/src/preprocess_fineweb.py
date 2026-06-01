#!/usr/bin/env python3
"""Preprocess FineWeb-Edu into a flat token stream for the Part-9 nano model.

Streams text from HuggingFaceFW/fineweb-edu, tokenises with the GPT-2 BPE
(tiktoken, vocab 50257 -- the same vocab nanoModelSpec*  assume), concatenates
documents with an end-of-text separator, and writes the token IDs as a flat
uint16 binary file.

Two deliberate choices keep the Python <-> Wolfram boundary trivial:

  * Output is a raw little-endian uint16 array, NOT WXF.  The WL job reads it
    with  BinaryReadList[file, "UnsignedInteger16"]  and windows it in-kernel.
  * IDs are shifted +1 (0..50256 -> 1..50257) so they index WL's 1-based
    EmbeddingLayer directly.  The GPT-2 end-of-text token 50256 becomes 50257.

Usage:
  python3 preprocess_fineweb.py --tokens 25_000_000 --out /tmp/fineweb_edu_calib.bin
"""

import argparse
import sys
import time

import numpy as np
import tiktoken
from datasets import load_dataset

EOT_RAW = 50256          # GPT-2 <|endoftext|>
VOCAB = 50257            # 0..50256
SHIFT = 1                # WL EmbeddingLayer is 1-indexed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=25_000_000,
                    help="target number of tokens to collect")
    ap.add_argument("--out", type=str, default="/tmp/fineweb_edu_calib.bin")
    ap.add_argument("--config", type=str, default="sample-10BT",
                    help="FineWeb-Edu subset/config name")
    ap.add_argument("--report-every", type=int, default=1_000_000)
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab == VOCAB, f"unexpected vocab {enc.n_vocab}"

    print(f"Streaming HuggingFaceFW/fineweb-edu [{args.config}] ...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.config,
                      split="train", streaming=True)

    buf = np.empty(args.tokens + 4096, dtype=np.uint16)  # small headroom
    n = 0
    ndocs = 0
    next_report = args.report_every
    t0 = time.time()

    for ex in ds:
        ids = enc.encode_ordinary(ex["text"])
        ids.append(EOT_RAW)
        chunk = np.asarray(ids, dtype=np.uint16) + SHIFT
        take = min(len(chunk), len(buf) - n)
        buf[n:n + take] = chunk[:take]
        n += take
        ndocs += 1
        if n >= next_report:
            rate = n / (time.time() - t0)
            print(f"  {n:,} tokens / {ndocs:,} docs  ({rate:,.0f} tok/s)",
                  flush=True)
            next_report += args.report_every
        if n >= args.tokens:
            break

    buf = buf[:n]
    buf.tofile(args.out)
    dt = time.time() - t0
    mb = buf.nbytes / 1e6
    print(f"\nWrote {n:,} tokens ({mb:.1f} MB uint16) from {ndocs:,} docs "
          f"to {args.out} in {dt:.0f}s", flush=True)
    print(f"  min id={buf.min()}  max id={buf.max()}  "
          f"(expect 1..{VOCAB})", flush=True)
    # sanity: shifted ids must be in 1..VOCAB
    if buf.min() < 1 or buf.max() > VOCAB:
        print("ERROR: token ids out of range", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
