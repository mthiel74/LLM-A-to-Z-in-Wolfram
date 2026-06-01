#!/usr/bin/env python3
"""Encode chat questions with the Alpaca template for chat.wls.

Wraps each question in the SAME no-input template prepare_alpaca.py uses, BPE-
encodes the full prompt (through '### Response:\n'), and writes
questions_encoded.json {questions:[str], ids:[[int]]}.  Keep the template in
sync with prepare_alpaca.py."""
import json
import os
import sys

import tiktoken

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

DEFAULT_QS = [
    "What is the capital of France?",
    "Give three tips for staying healthy.",
    "Explain what a star is in one sentence.",
    "Write a short poem about the sea.",
    "What is the best way to learn a new language?",
]


def main():
    enc = tiktoken.get_encoding("gpt2")
    qs = DEFAULT_QS
    if len(sys.argv) > 1:                       # questions from a file, one per line
        qs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    ids = [enc.encode_ordinary(PROMPT_NO_INPUT.format(instruction=q)) for q in qs]
    d = os.path.join(os.path.dirname(__file__), "..", "data", "questions_encoded.json")
    json.dump({"questions": qs, "ids": ids}, open(d, "w"))
    print(f"wrote {len(qs)} encoded questions to {os.path.normpath(d)}")


if __name__ == "__main__":
    main()
