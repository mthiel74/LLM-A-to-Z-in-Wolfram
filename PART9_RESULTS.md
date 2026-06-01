# Part 9 — From-Scratch Pretraining: Results

Source material for the Wolfram Community post. Captures the trained base
model, its metrics, and representative generations. (Numbers are from the
**v2** 30M run; **v3** adds a global LR schedule — update when it finishes.)

## What we built

A nano-style transformer (`NanoChat.wl`: RMSNorm, RoPE, SwiGLU, no biases —
the LLaMA/Mistral/nanochat family) trained **from random initialisation** in
Wolfram Language on AWS Batch spot GPUs.

- **Model:** 30M-family spec — dModel 256, 8 heads, 4 layers, dFF 1024, vocab
  50257 (GPT-2 BPE), context 512. **29,925,888 (~29.9M) trainable parameters**
  (the token embedding + untied output projection dominate at full vocab, ~26M
  of the 30M). The full NetChain reports ~40.4M arrays, but ~10.5M of those are
  the *frozen* RoPE sine/cosine + causal-mask tables (no gradient).
- **Data:** FineWeb-Edu, 600M tokens (one Chinchilla-optimal epoch for this
  size), tokenised with tiktoken's GPT-2 BPE.
- **Compute:** one NVIDIA T4 (g4dn.xlarge spot, eu-central-1), ~16,100
  tokens/s, ~10.6 hours, **~$2.60**.

## Training curve: v2 vs v3

Two runs of the identical 30M model on the identical 600M tokens, differing only
in the learning-rate schedule. Per-shard tail perplexity over 30 × 20M-token
shards:

| Tokens seen | v2 (per-shard LR) | v3 (global cosine LR) |
|---|---|---|
| 20M  | 160 | 157 |
| 60M  | 91  | 89  |
| 100M | 75  | 76  |
| 200M | 63  | 63  |
| 300M | 56  | 58  |
| 500M | 53  | 53  |
| 600M | **55** | **52** |

Both start near ln(50257) ≈ 10.83 (the uniform-prior baseline for a
randomly-initialised model; `Log[50257]` = 10.825). **v2** resets its learning rate on every shard (30
short warmup/decay runs) and **plateaus around 300M tokens**. **v3** uses a
single global cosine schedule (2×10⁻³ → 2×10⁻⁴ across the whole run) and keeps
descending to the end.

The headline perplexity gap is small (52 vs 55), and at 30M parameters / 600M
tokens the model is near its capacity-and-data limit. **But the downstream
effect after instruction tuning is large** — see the chat comparison below.
Each run cost ~$2.60 and took ~10.4 hours on one T4 (~16,100 tokens/s).

## Sample generations (v2 base model, top-k k=40, temp 0.9)

These are raw continuations from the **base** (pretrained-only) model — it
continues text, it does not yet answer questions (that needs the SFT step in
the next section).

> **The history of** the American Revolution, and the history of slavery in
> America, was brought to life by the American Revolution. The Civil War was
> based on the battles of the Civil War… the battle of Fort Knox, the Battle…

> **Water is important because** it can be used to make a smooth, clear and
> clean surface and remove and maintain the sealant and/or other parts of the
> body. It protects the sealant from injury and reduces the risk of damage.

> **The best way to learn a new language is** to use a new language. Students
> with a new language experience to apply their new language skills… should
> learn the full spectrum of language in their native language.

> **The sun is a star that** is found in the eastern sky… There are three
> types of stars called supernovae, and you can get the number of
> star-determinants.

> **My favorite food is pizza because** they can be baked at the grocery
> store… it's been linked to health issues, your taste can be…

**Assessment:** fluent, grammatical, on-topic English with real lexical
association — a genuine from-scratch language model. Weaknesses (repetition
loops, thin world-knowledge, no instruction-following) are exactly what
perplexity ~55 at 30M parameters predicts.

## Engineering notes (the interesting part of the story)

The model was the easy bit; making spot training reliable was the work.

1. **IMDS hop limit.** Every AWS Batch job failed at "no AWS credentials"
   until the instance metadata hop limit was raised to 2 (containers are one
   network hop from IMDS). First job on the stack only ran after this fix.
2. **Logits vs probabilities.** `CrossEntropyLossLayer["Index"]` expects
   probabilities; feeding raw logits drove the loss unbounded-negative. Fixed
   by a softmax before the loss; the saved checkpoint keeps the logit head.
3. **No `aws` CLI in the image.** All S3 I/O is WL-native via
   `ServiceExecute[…, "GetObject"/"PutObject"]` (binary `.wlnet` round-trips
   byte-identical).
4. **Spot resilience.** v1 died when a spot instance was reclaimed mid-chunk
   (and the Mac orchestrator died from memory pressure). v2/v3 use one
   self-looping in-container job, S3 checkpoint between shards, and an AWS
   Batch retry strategy that auto-resumes on reclamation — no long-lived local
   process. The full v2 run then completed in one attempt with zero
   interruptions.

## Pipeline (all in `part-09-from-scratch/src/`)

- `preprocess_fineweb.py` — stream + tokenise FineWeb-Edu to a uint16 token stream.
- `shard_and_upload.py` — split into S3 shards + a run manifest.
- `run_pretraining_v2.wls` / `run_pretraining_v3.wls` — self-looping resumable trainer.
- `sample.wls` + `encode_prompts.py` / `sample_decode.py` — top-k generation.

## Instruction tuning (SFT) → a model you can talk to

The base model continues text but cannot answer. SFT on instruction/response
pairs (Alpaca, ~52k examples, 3 epochs) turns it into a chat model and teaches
it to stop (the EOT token as a learned end-of-turn). The SFT run took ~18
minutes on one T4 (~$0.10). See `prepare_alpaca.py`, `run_sft.wls`, `chat.wls`.

### Chat transcripts (v2 base + SFT, top-k k=40, temperature 0.8)

These are real answers from the instruction-tuned 30M model. Unlike the base
model, it follows the instruction format and **stops on its own** (emitting the
learned end-of-turn token) instead of rambling.

> **Q: Give three tips for staying healthy.**
> 1. Know your body and mind about all foods
> 2. Get enough sleep
> 3. Take regular breaks
> 4. Get enough sleep
> 5. Exercise regularly

> **Q: Write a short poem about the sea.**
> The ocean of the sea was a peaceful place. / The sea was so dry. / The sea was
> so thick that it slowly crept in the ocean, / And like a gentle shore of the
> shore, / ... / A tranquil atmosphere that was full of life. / ... / For ever
> again and again, the sea was so warm, / Soft surf, and a wonderful place

> **Q: What is the best way to learn a new language?**
> The best way to learn a new language is to learn a new language through natural
> language and create new experiences. Additionally, some language-based skills
> such as problem-solving ... can help to build skills that are easier to learn.

> **Q: What is the capital of France?**
> The capital of France is Britannica, France.

> **Q: Explain what a star is in one sentence.**
> A star is in one sentence.

**Assessment (v2):** the SFT is clearly working — the model adopts the
instruction/response shape and terminates cleanly. But on the v2 base its
factual reliability is weak: it has the right *form* but the wrong content.

### v2 vs v3 chat, head to head

The same questions, asked of the v2-SFT and v3-SFT models (the only upstream
difference is the base model's learning-rate schedule). These are the **verbatim
outputs** — including the wrong answers, because the honest result matters more
than a flattering one:

| Question | v2-SFT answer | v3-SFT answer |
|---|---|---|
| Capital of France? | "the capital of France is Britannica, France." | "The capital of France is Austria." |
| What is a star? (one sentence) | "A star is in one sentence." | "A star is in another sentence." |
| Three health tips | sleep / breaks / **sleep again** / exercise (5 items, repeats) | 11 items, repeats "Practice relaxation", "Get enough sleep" |
| Poem about the sea | "The ocean of the sea was a peaceful place. The sea was so dry…" | "The sea is like a river filled with a soft and hard sea. ### Response: Sea is like a sea of water." (leaks the template) |

**Honest assessment:** the global-LR base (perplexity 52 vs 55) did **not**
produce a meaningfully better chat model. Both versions learn the
instruction/response *form* — they answer in the right shape and stop on their
own — but neither is factually reliable: the capital of France is wrong in both
(Britannica / Austria), the one-sentence star definition collapses in both, and
the tip lists repeat themselves. The v3 model occasionally even leaks the
`### Response:` template.

The lesson is the opposite of what a tidy write-up would want: at 30M parameters
on 600M tokens (one epoch), a small perplexity gain (55 → 52) does **not**
translate into a better assistant. Getting real chat quality needs a bigger
model and/or far more data — not a better learning-rate schedule on a model that
is already near its capacity-and-data limit. What this project *does*
demonstrate, end to end, is the full pipeline working: random init → fluent
English → instruction-following form, in Wolfram Language, for ~$3 and ~11 hours
per version (pretraining ~$2.6 / ~10.4 h; SFT ~$0.10 / ~18 min).
