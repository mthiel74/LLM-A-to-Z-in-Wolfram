# LLM A to Z in Wolfram Language

Building a small Large Language Model from scratch — entirely in Wolfram Language — as a
five-part series for the [Wolfram Community](https://community.wolfram.com).

Each part is self-contained and publishable on its own; the cumulative artefact is a clean,
readable WL implementation of a small transformer language model trained on Tiny Shakespeare.

## The five parts

| # | Title | Compute | Status |
|---|-------|---------|--------|
| 1 | Tokeniser and embeddings | CPU | draft for review |
| 2 | Self-attention and the QKV picture | CPU | draft for review |
| 3 | The full transformer block + parameter accounting | CPU | draft for review |
| 4 | Pre-training (CPU local + cloud option) | CPU / RemoteBatchSubmit | draft for review |
| 5 | Sampling, temperature, beam, speculative decoding | CPU | draft for review |
| 6 | Supervised fine-tuning (instruction-following) | CPU / RemoteBatchSubmit | draft for review |
| 7 | RL post-training (GRPO-style on verifiable task) | CPU / RemoteBatchSubmit | draft for review |
| 8 | Interactive chat widget + the deployed model | CPU | draft for review |
| 9 | Pretraining a model **from scratch** on cloud GPUs | AWS Batch (T4 spot) | draft for review |

The series has been extended past Raschka's book to match the scope of
Karpathy's `nanochat` (2024–25): a small but **functional** chat-style
LLM that has been pretrained, instruction-tuned, and tuned with
reinforcement learning. See `SERIES_PLAN.md` for the full per-part
outline, dataset choices, and compute strategy.

## Part 9: training a model from scratch

Parts 1–8 build and train small models on CPU, and the scale-up notebook
fine-tunes OpenAI's pretrained GPT-2 weights. Part 9 closes the loop: a complete
language model trained **from random initialisation** in Wolfram Language, on
rented cloud GPUs, for a tutorial-scale budget.

- **Architecture** (`part-09-from-scratch/src/NanoChat.wl`): a nano-style
  transformer in the LLaMA/Mistral/nanochat family — RMSNorm, rotary position
  embeddings (RoPE), SwiGLU MLP, no biases. ~40M parameters, GPT-2 BPE vocab
  (50,257), context 512.
- **Data**: 600M tokens of FineWeb-Edu (one Chinchilla-optimal epoch).
- **Compute**: a single NVIDIA T4 (AWS Batch spot, g4dn.xlarge), ~16,000
  tokens/s, ~10.5 hours, **about $3**.
- **Result**: loss falls from ln(50257) ≈ 11.3 to perplexity ~55; the model
  generates fluent, on-topic English from a cold start. Supervised fine-tuning
  on Alpaca then turns it into a chat model.

Training is sharded with checkpoints to S3, and a single self-looping cloud job
resumes automatically if its spot instance is reclaimed. See
[`PART9_RESULTS.md`](PART9_RESULTS.md) for the training curve, sample
generations, chat transcripts, costs, and engineering notes.

## What this is not

This series is a *companion-in-spirit* to Sebastian Raschka's
*Build a Large Language Model from Scratch* (Manning, 2024). It is not a translation of the
book or its accompanying repository — the language, derivations, code structure, examples,
and exposition are all original. The intersection with the book is the canonical mathematical
content of the transformer (which is shared by every text on the subject) and the use of
Tiny Shakespeare as the training corpus (a community convention going back to Karpathy's
`char-rnn`, 2015).

## Hardware target

A Mac mini. No GPU required at the educational scale used here.

## Reproducibility

Every part ships a single notebook that runs end-to-end from a fresh kernel using only
`NetModel`, `ExampleData`, and the Tiny Shakespeare corpus (downloaded once, cached locally).

## Layout

```
part-01-tokeniser-and-embeddings/
  src/         Wolfram Language sources (.wl) — the testable code
  notebook/    The Wolfram Community notebook (.nb) built from the sources
  figures/     Generated figures referenced from the post
  data/        Cached corpus + saved artefacts (gitignored beyond a stub)
  README.md    Per-part synopsis
  POST.md      The Wolfram Community post draft (markdown)
```

## License

MIT for the code. The Tiny Shakespeare corpus is in the public domain.
