#!/usr/bin/env python3
"""Construct a fresh HuggingFace GPT-2 medium model and load weights from the
.npy files exported by export_wl_to_hf.wls.

Critical step: WL's bundled GPT-2 NetModel uses its OWN ordering of the
50,257 BPE tokens — different from HuggingFace's gpt2 tokenizer ordering.
A permutation is required for transformer.wte.weight (and the tied
lm_head.weight): row j of the HF embedding must come from the WL row whose
decoded string equals HF token j's string.

The permutation is computed from data/hf_weights/wl_vocab.json (one decoded
string per WL token, position i → WL id i+1) versus the HF tokenizer's
decode of each id.

After running this you have a complete HF model directory at
discussion-model/data/hf_model/.
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

ROOT       = Path("/Users/thiel/GitHub/LLM-A-to-Z-in-WL/discussion-model")
NPY_DIR    = ROOT / "data" / "hf_weights"
OUT_DIR    = ROOT / "data" / "hf_model"
VOCAB_JSON = NPY_DIR / "wl_vocab.json"
EXPECTED_PARAMS = 354_823_168

# -------------------------------------------------------------------- config
config = GPT2Config(
    vocab_size=50_257,
    n_positions=1024,
    n_embd=1024,
    n_layer=24,
    n_head=16,
    n_inner=4096,
    activation_function="gelu_new",
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
)

print("Constructing fresh HF GPT-2 medium model ...")
model = GPT2LMHeadModel(config)
sd = model.state_dict()
total_unique = sum(p.numel() for n, p in model.named_parameters())
print(f"  fresh-model unique params = {total_unique:,}")

# -------------------------------------------------------------- vocab align
print("Loading WL vocabulary (one decoded string per WL token) ...")
with VOCAB_JSON.open() as f:
    wl_vocab = json.load(f)           # 50257 entries, position i = WL id i+1
assert len(wl_vocab) == 50_257

print("Loading HF tokenizer (gpt2-medium) ...")
tok = GPT2Tokenizer.from_pretrained("gpt2-medium")

# Build str -> WL row index (0-indexed). Duplicates exist (e.g. non-printable
# bytes); we keep the *first* occurrence — matches what NetEncoder would
# prefer if both decode to the same string.
wl_str_to_idx = {}
for i, s in enumerate(wl_vocab):
    if s not in wl_str_to_idx:
        wl_str_to_idx[s] = i

# Compute permutation: hf_to_wl[j] = WL row idx (0-indexed) whose string == HF j
hf_to_wl = np.full(50_257, -1, dtype=np.int64)
missing = []
for j in range(50_257):
    s = tok.decode([j])
    idx = wl_str_to_idx.get(s)
    if idx is None:
        missing.append((j, s))
    else:
        hf_to_wl[j] = idx
print(f"  permutation built; {len(missing)} HF tokens have no WL match "
      f"(typical for malformed UTF-8 BPE fragments).")

# ------------------------------------------------------------ load tensors
loaded = {}
print(f"Loading .npy files from {NPY_DIR} ...")
for key in sd.keys():
    npy = NPY_DIR / f"{key}.npy"
    if not npy.exists():
        continue
    arr = np.load(npy)
    if key == "transformer.wte.weight" or key == "lm_head.weight":
        # Permute rows: new[j] = WL[hf_to_wl[j]]; missing j get zeros so
        # those tokens have near-zero logits at the output and effectively
        # zero contribution at the input (we don't train them).
        permuted = np.zeros_like(sd[key].cpu().numpy(), dtype=np.float32)
        valid = hf_to_wl >= 0
        permuted[valid] = arr[hf_to_wl[valid]]
        arr = permuted
    if arr.shape != tuple(sd[key].shape):
        raise SystemExit(
            f"Shape mismatch for {key}: file={arr.shape} model={tuple(sd[key].shape)}")
    loaded[key] = torch.from_numpy(arr.copy())
print(f"  loaded {len(loaded)} tensors")

# Inject into model
missing_keys, unexpected = model.load_state_dict(loaded, strict=False)
print(f"missing keys ({len(missing_keys)}): {missing_keys[:6]}")
print(f"unexpected keys ({len(unexpected)}): {unexpected[:6]}")
model.tie_weights()

total_after = sum(p.numel() for p in model.parameters())
print(f"Total params after load/tie: {total_after:,}")
assert total_after == EXPECTED_PARAMS, \
    f"got {total_after}, expected {EXPECTED_PARAMS}"

# Sanity check: WL id 12727 (= 'User') is at WL index 12726.
# HF id for 'User' is 12982; after permutation, wte[12982] should equal
# WL row 12726 of the original wte file.
wte_loaded = np.load(NPY_DIR / "transformer.wte.weight.npy")
wte_now = model.transformer.wte.weight.detach().cpu().numpy()
assert np.allclose(wte_now[12982], wte_loaded[12726], atol=1e-6), \
    "permutation did NOT line up token 'User' (HF 12982 vs WL 12727)"
print("vocab permutation sanity check OK  (HF 12982 'User' == WL row 12726)")

# Save model + tokenizer
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"Saved HF model to {OUT_DIR}")
