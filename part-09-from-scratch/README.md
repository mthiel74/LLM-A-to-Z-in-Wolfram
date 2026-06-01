# Part 9 — Pretraining from scratch (nano-style)

Random-init → coherent chat in Wolfram Language, in the architecture
Karpathy's [nanochat](https://github.com/karpathy/nanochat) uses (same
family as LLaMA / Mistral / Qwen).

## What's different from Part 3

| Component        | Part 3 (`gptModel`)       | Part 9 (`nanoModel`)        |
|------------------|---------------------------|-----------------------------|
| Normalisation    | LayerNorm (with bias)     | RMSNorm (no learnable gain) |
| Position info    | Learned absolute embedding | Rotary (RoPE), per-head    |
| MLP non-linearity | GELU                      | SwiGLU                      |
| Linear biases    | Yes                       | No (anywhere)               |

Pre-norm residual structure and causal self-attention are unchanged.

## Source layout

- `src/NanoChat.wl` — package with `rmsNorm`, `applyRoPE`,
  `swiGluMLP`, `nanoSingleHeadAttention`, `nanoMultiHeadAttention`,
  `nanoTransformerBlock`, `nanoModel`, and four spec families
  (`Tiny`, `30M`, `100M`, `200M`).
- `src/test_nanochat.wls` — CPU sanity test; runs in ~30 s.

## Model sizes

| Spec           | dModel | nHeads | nLayers | dFF  | Params @ v=50257 |
|----------------|--------|--------|---------|------|------------------|
| `Tiny`         | 64     | 4      | 2       | 256  | ~144 k (tiny vocab) |
| `30M`          | 256    | 8      | 4       | 1024 | 29.9 M            |
| `100M`         | 512    | 8      | 12      | 2048 | 101.8 M           |
| `200M`         | 768    | 12     | 12      | 3072 | 190.4 M           |

The 200M spec matches Karpathy's nanochat architecturally; nanochat
reports 162 M with weight tying, which this implementation does not
yet do (so the untied count is ~190 M).
