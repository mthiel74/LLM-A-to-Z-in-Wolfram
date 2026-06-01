# Part 2 — Self-attention and the QKV picture

The second instalment of *LLM A to Z in Wolfram Language*. Feed the
embedding tensor from Part 1 into scaled dot-product attention; build
the math up from first principles; visualise real attention on
pretrained GPT-2.

## What this part delivers

1. **Scaled dot-product attention from first principles**, including a
   derivation of why the `1/√d_k` factor is the *unique* scaling that
   keeps pre-softmax logits at unit variance.
2. **Single-head causal self-attention as a `NetGraph`**, with all
   intermediate tensors inspectable via `NetExtract`.
3. **Multi-head attention** as a parallel ensemble of single-head
   computations on subspaces of `d_model`.
4. **Real attention heatmaps**, taken from a pretrained GPT-2 small
   loaded via `NetModel["GPT-2 Transformer Trained on WebText Data"]`,
   showing head specialisation: previous-token tracker, syntactic-
   dependency tracker, near-uniform broadcast.

## Layout

```
part-02-self-attention/
  PLAN.md                       full plan with section outline & deliverables
  src/
    Attention.wl                the package: scaled dot-product + multi-head
    test_attention.wls          smoke tests on a toy input
    run_part02.wls              end-to-end: builds figures & extracts GPT-2 weights
    build_notebook.wls          assembles the Wolfram Community .nb
  data/                         cached GPT-2 attention tensors per prompt
  figures/                      generated figures referenced from the post
  notebook/Part02.nb            the deliverable
  POST.md                       markdown draft
```

## Reproducing

```bash
cd part-02-self-attention/src
wolframscript -file run_part02.wls    # downloads GPT-2 (~500 MB) on first run
wolframscript -file build_notebook.wls
```

First-time run downloads the pretrained GPT-2 weights via WL's
`NetModel`; subsequent runs are fast.
