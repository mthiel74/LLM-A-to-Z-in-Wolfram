# Part 3 — The full transformer block

The canonical pre-norm transformer block, stacked. Builds on
`Attention.wl` from Part 2 and the embedder from Part 1, exporting a
`gptModel[dModel, nHeads, nLayers, dFF, vocabSize, maxSeqLen]` that
returns a `NetChain` from token IDs to per-position logits.

## What this part delivers

1. **A pre-norm transformer block as a `NetGraph`** — LayerNorm →
   attention → residual → LayerNorm → feed-forward → residual.
2. **A position-wise feed-forward block** with the GELU non-linearity,
   the place where most of the model's parameters live.
3. **A closed-form parameter count** as a function of
   `(dModel, nHeads, nLayers, dFF, vocabSize, maxSeqLen)`, verified
   against the empirical `Information[net, "ArraysTotalElementCount"]`.
4. **Three model-size constructors** (`modelSpec1M`, `modelSpec5M`,
   `modelSpec25M`) sized for the training runs in Part 4.
5. **A pre-norm vs post-norm gradient-flow argument** and an empirical
   figure on a randomly-initialised deep stack.
6. **A `chainEmbedder`** — a single-input embedding NetGraph that uses
   `SequenceIndicesLayer` so it can sit inside a `NetChain` without
   requiring explicit position input.

## Layout

```
part-03-transformer-block/
  src/
    Transformer.wl              the package
    test_transformer.wls        VerificationTest assertions
    run_part03.wls              builds figures + cached model artefacts
    build_notebook.wls          assembles the Wolfram Community notebook
  data/                         cached model checkpoints + parameter summary
  figures/                      generated figures referenced from the post
  notebook/Part03.nb            the deliverable
  POST.md                       markdown draft
```
