#!/usr/bin/env python3
"""Interactive streaming chat REPL for the WL-trained GPT-2 medium SFT model
imported into HuggingFace format. Uses PyTorch + MPS (Metal) on Apple Silicon.

Ports the same heuristics as the WL chat_cli.wls:
  - top-k sampling with temperature (default top-k=40, T=0.7)
  - stop on " You:", " User:", " Assistant:", "\\nUser:", "\\nYou:",
    "\\nAssistant:"
  - stop on 3 identical token IDs in a row
  - stop on 2-or-fewer unique IDs in the last 6 tokens
  - 4-turn rolling history
Commands inside the chat:
  /reset    clear transcript
  /quit     exit
"""

from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("/Users/thiel/GitHub/LLM-A-to-Z-in-WL/discussion-model/data/hf_model")

# Device pick. MPS is the fast path; fall back to CPU on Intel.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float16   # MPS does fp16 cleanly; halves bandwidth
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

print(f"Loading model from {MODEL_DIR} on {DEVICE} ({DTYPE}) ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=DTYPE)
model.to(DEVICE)
model.eval()
print(f"  loaded in {time.time() - t0:.1f} s; "
      f"{sum(p.numel() for p in model.parameters()):,} params")

# Mask of valid HF token IDs (i.e. those whose embedding row is non-zero).
# WL's tokenizer doesn't cover ~18 malformed UTF-8 BPE fragments; their rows
# were zeroed by load_into_hf.py so they contribute zero hidden state, but
# the logits they produce are still 0 (vs. the trained tokens' very-negative
# unselected logits), which puts them inside top-k unfairly.  Mask them
# permanently to -inf.
with torch.no_grad():
    _norms = model.transformer.wte.weight.norm(dim=-1)
    INVALID_TOKEN_IDS = torch.nonzero(_norms < 1e-6, as_tuple=False).flatten().to(DEVICE)
print(f"  masking {len(INVALID_TOKEN_IDS)} unmapped tokens (WL vocab gaps)")

STOP_STRINGS = [" You:", " User:", " Assistant:",
                "\nUser:", "\nYou:", "\nAssistant:"]


@torch.no_grad()
def generate_stream(prompt: str, max_new: int = 100,
                    temperature: float = 0.7, top_k: int = 40):
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = []
    runtext = ""
    past = None
    cur = ids
    # Use KV cache: feed prompt once, then one token at a time.
    for step in range(max_new):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :] / temperature
        # Mask the unmapped-vocab tokens so they can never be sampled.
        logits[INVALID_TOKEN_IDS] = float("-inf")
        if top_k > 0 and top_k < logits.shape[-1]:
            top_vals, top_idx = torch.topk(logits, top_k)
            probs = torch.softmax(top_vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_id = top_idx[choice].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_id)
        chunk = tok.decode([next_id])
        runtext += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()

        # Stop conditions (port of WL heuristics).
        hit = next((s for s in STOP_STRINGS if s in runtext), None)
        if hit:
            runtext = runtext.split(hit, 1)[0]
            break
        if len(generated_ids) >= 3 and \
           generated_ids[-1] == generated_ids[-2] == generated_ids[-3]:
            break
        if len(generated_ids) >= 6 and \
           len(set(generated_ids[-6:])) <= 2:
            break

        cur = torch.tensor([[next_id]], device=DEVICE)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return runtext.strip(), len(generated_ids)


def main():
    print("\n" + "=" * 64)
    print("  Chat with GPT-2 medium SFT (20k Alpaca, 1 epoch) — Python / MPS")
    print("  Streams tokens as they generate.  Commands:")
    print("    /reset    clear transcript")
    print("    /quit     exit")
    print("=" * 64 + "\n")

    history = []
    try:
        while True:
            try:
                line = input("You: ").strip()
            except EOFError:
                print()
                break
            if line in {"/quit", "/exit"}:
                break
            if line == "/reset":
                history = []
                print("[transcript cleared]")
                continue
            if not line:
                continue

            prompt = "".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}\n"
                for t in history
            ) + f"User: {line}\nAssistant:"

            sys.stdout.write("AI: ")
            sys.stdout.flush()
            t0 = time.time()
            reply, n_tok = generate_stream(prompt, max_new=100,
                                           temperature=0.7, top_k=40)
            elapsed = time.time() - t0
            rate = n_tok / elapsed if elapsed > 0 else float("nan")
            print(f"[{elapsed:.1f} s, {n_tok} tok, {rate:.1f} tok/s]\n")
            history.append({"user": line, "assistant": reply})
            history = history[-4:]
    except KeyboardInterrupt:
        print("\n(interrupted)")
    print("bye.")


if __name__ == "__main__":
    main()
