# Part 1 — Tokeniser and embeddings

The first instalment of *LLM A to Z in Wolfram Language*. Turn raw text into the
input tensor a transformer expects.

## What this part delivers

1. **Three tokenisation strategies** implemented from scratch in WL and
   compared on Tiny Shakespeare:
   - character-level (smallest vocabulary, longest sequences),
   - whitespace word-level (largest vocabulary, shortest sequences),
   - byte-pair encoding (BPE) — the actual deliverable, trained from
     iterative merges of the most frequent adjacent pair.
2. **The vocabulary-size vs sequence-length trade-off** as a concrete
   log–log curve, with the two baselines marked as extreme points.
3. **A token + positional embedding `NetGraph`** that consumes a sequence of
   BPE token IDs and emits the (seqLen × dModel) input tensor of the
   transformer to come in Part 3.
4. **A pre-training visualisation** of the embedding geometry at random
   initialisation — the "before" picture we will revisit in Part 4 after
   training.

## Layout

```
part-01-tokeniser-and-embeddings/
  src/
    Tokeniser.wl              the package: preTokenise, trainBPE, bpeEncode,
                              buildBPEVocabulary, tokenPositionEmbedder, ...
    test_tokeniser.wls        smoke tests on a tiny toy corpus
    run_part01.wls            end-to-end Part 1 build (writes figures + data)
    build_notebook.wls        assembles the Wolfram Community .nb
  data/
    tinyshakespeare.txt       the corpus (downloaded on first run)
    bpe_trained.mx            cached BPE merges + base alphabet
    embedder.wlnet            initialised NetGraph (vocab→tensor)
    part01_summary.json       numbers quoted in the post
  figures/
    vocab_vs_seqlen.png       the trade-off curve
    embedding_geometry_random.png
    embedding_pca_random.png
  notebook/
    Part01.nb                 the Wolfram Community post (built from sources)
  POST.md                     the markdown draft of the post
```

## Reproducing

```bash
cd src
wolframscript -file run_part01.wls   # ~minutes on a Mac mini
wolframscript -file build_notebook.wls
```

The `run_part01` script downloads Tiny Shakespeare on first run, trains BPE,
sweeps merge counts to draw the vocab/sequence curve, builds the embedder,
and writes all figures and serialised artefacts.
