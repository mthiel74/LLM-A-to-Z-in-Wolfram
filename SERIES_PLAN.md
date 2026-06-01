# Build a Small Working LLM from Scratch — in Wolfram Language

**A teaching series for the Wolfram Community.**

Companion-in-spirit to Sebastian Raschka's *Build a Large Language Model
from Scratch* (book + GitHub repo), extended past where the book ends so
that the final artefact is something one can actually *chat with*: a
small Wolfram-Language model that has been pretrained, instruction-tuned,
and tuned with reinforcement learning. Karpathy's `nanochat` (2024–25)
is the modern-pipeline reference; we keep his stage structure but stay
inside Wolfram Language throughout and let everything run on a Mac mini
by default, with an optional `RemoteBatchSubmit` cloud path for the
larger model and longer training runs.

## What each post delivers

Each part is publishable on its own. Every notebook is an *active
document* — `Manipulate` widgets, `Animate` cells, and interactive
inspectors of the model's internal state — not just code and prose.
The cumulative artefact is a working chat-style LLM with a
notebook UI.

| # | Title | Compute |
|---|-------|---------|
| 1 | Tokeniser and embeddings | CPU |
| 2 | Self-attention and the QKV picture | CPU |
| 3 | The full transformer block + parameter accounting | CPU |
| 4 | Pre-training on Tiny Shakespeare (CPU) + scaling on a real corpus via `RemoteBatchSubmit` | CPU **or** Cloud |
| 5 | Sampling, temperature, top-k/p, beam search, speculative decoding | CPU |
| 6 | Supervised fine-tuning: instruction & dialogue | CPU **or** Cloud |
| 7 | RL post-training (GRPO-style on a verifiable task) | CPU **or** Cloud |
| 8 | A working chat widget and the deployed model | CPU |

## Compute strategy

**CPU-first.** Every part runs on a Mac mini by default, with a model
size chosen so that the training loop finishes in well under an hour.
The 1M–5M parameter range is small but enough to see scaling-law
behaviour and produce qualitatively reasonable Shakespeare-like
continuations.

**Optional cloud path.** Each compute-heavy part also ships a "Go
bigger" cell that submits the same training to the Wolfram Cloud via
`RemoteBatchSubmit`. This lets the reader train a 25M-parameter model
on a larger corpus (FineWeb-Edu sample, ~100 MB) without owning a GPU.

## Datasets

| Stage | Dataset | Size | Why |
|---|---|---|---|
| BPE training + Part 4 CPU pretraining | Tiny Shakespeare | 1.1 MB | Karpathy-era convention; reproduces in seconds |
| Part 4 Cloud pretraining | FineWeb-Edu (sample) | 100 MB – 1 GB | Real-world distribution; HuggingFace `HuggingFaceFW/fineweb-edu` |
| Part 6 SFT mixture | TinyStories instructions + a slice of SmolTalk | ~10 MB | Conversational format the chat widget will use |
| Part 7 RL | GSM8K (grade-school math) | 8.8K problems | Verifiable binary reward; nanochat's choice |

## Differentiators from existing material

- **Wolfram Language throughout.** No Python.
- **Notebooks are active.** Every part contains `Manipulate` /
  `Animate` cells that let the reader change a hyperparameter and see
  the effect immediately.
- **First-principles derivations** of design choices that other
  treatments assert: the `1/√d_k` scaling (Part 2), the pre-norm vs
  post-norm gradient argument (Part 3), the Chinchilla compute-optimal
  exponent (Part 4), GRPO loss vs vanilla policy gradient (Part 7).
- **End-to-end working chat artefact.** The series produces something
  one can actually chat with, not a checkpoint that nobody runs.

---

## Per-part outlines

### Part 1 — Tokeniser and embeddings (CPU)

Done. See `part-01-tokeniser-and-embeddings/`.

### Part 2 — Self-attention and the QKV picture (CPU)

Done. See `part-02-self-attention/`.

### Part 3 — The full transformer block

Assemble the canonical block (pre-norm → attention → residual →
pre-norm → feed-forward → residual), stack `n_layers` of them, and
derive the parameter count as a closed-form function. Build three
model sizes (1M, 5M, 25M parameters). Cover (a) why pre-norm trains
more stably at depth, (b) weight-tying input and output embeddings,
and (c) the GELU / ReLU² choice. Optional: rotary positional
embeddings (RoPE) instead of the learned positions from Part 1, as
nanochat does.

### Part 4 — Pre-training on Tiny Shakespeare + cloud scaling

The CPU path: train the 1M and 5M models on Tiny Shakespeare to
recover a small-scale Chinchilla loss-vs-compute curve, with live
loss plotting via `TrainingProgressFunction`. Should finish in well
under an hour. The cloud path: submit the 25M model on a
FineWeb-Edu slice via `RemoteBatchSubmit`. Show the matching loss
curve from the cloud run. Discuss the bitter-lesson observation:
more compute and data is the dominant lever.

### Part 5 — Sampling, temperature, beam search

Implement greedy, temperature, top-k, top-p, and beam search from
scratch. Animate the output distribution as temperature sweeps from
0 to 2. Add speculative decoding (small draft model + large verifier)
and benchmark the throughput gain.

### Part 6 — Supervised fine-tuning

Switch from raw-text next-token training to instruction-following
training. Define the conversation markup: `<|user|>...
<|assistant|>...`. Mix a small dataset (a curated TinyStories
instruction subset plus a slice of SmolTalk if size permits) and
train for a small number of epochs. Show before-vs-after generations
on a held-out prompt.

### Part 7 — RL post-training

The GRPO recipe from nanochat, adapted: sample `k` rollouts per
training example from the current model, compute a binary reward
from the ground-truth answer of a verifiable task (a GSM8K subset,
or a simple format-following task we invent), centre advantages,
take a policy-gradient step. Discuss why this works (high-variance
reward, unbiased gradient, no reward model required). Discuss DPO
and PPO as alternatives.

### Part 8 — A working chat widget

A `Dynamic` + `EventHandler` widget inside the notebook: the reader
types a prompt, the chat model streams its continuation in real
time. The model is the one trained in Part 7. Final artefact: the
notebook is itself the chat UI. Mention that the same model can be
deployed via `CloudDeploy` to give it a permanent URL.

---

## Notebook conventions across the series

- Title + subtitle + dated front matter.
- A "hero illustration" near the top — a clean schematic of what the
  post builds, generated via `gpt-image-2`.
- *Active state inspectors*: at every stage where there is a
  hyperparameter, a `Manipulate` slider so the reader can see what
  changes when they change it.
- *Animations*: at every place where a process unfolds in time
  (BPE merge sequence, softmax saturation as `d_k` grows, attention
  weights as the network sees more context, training-loss curve as
  it descends), an `Animate` or `ListAnimate` of the unfolding state.
- *Live tokenisation cells*: any cell that demonstrates the
  tokeniser shows the actual coloured token boundaries on a string.
- *Real-data figures alongside synthetic*: where possible, show the
  effect on real text (Tiny Shakespeare, then later GPT-2 on a longer
  prompt) rather than only on random tensors.
- *CPU-default, cloud-optional* code paths for everything in Parts
  4, 6, 7.

## Repo layout

```
LLM-A-to-Z-in-WL/
  README.md
  LICENSE
  SERIES_PLAN.md
  part-01-tokeniser-and-embeddings/
  part-02-self-attention/
  part-03-transformer-block/
  part-04-pretraining/
  part-05-sampling/
  part-06-supervised-fine-tuning/
  part-07-rl-post-training/
  part-08-chat-widget/
```
