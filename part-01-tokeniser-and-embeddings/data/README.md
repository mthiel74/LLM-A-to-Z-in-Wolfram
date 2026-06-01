# Data cache

This directory caches the Tiny Shakespeare corpus and any saved artefacts produced
by Part 1 (learned BPE merge rules, vocabularies, trained or randomly-initialised
embedding networks). Everything in here except this README is gitignored — it is
either re-downloadable or regenerable.

## Tiny Shakespeare

- **Source:** `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- **Size:** ~1.1 MB
- **Provenance:** A concatenation of public-domain Shakespeare play text assembled
  by Andrej Karpathy for the original `char-rnn` repository (2015).
- **Why this corpus:** Small enough to train on a CPU in minutes, large enough that
  byte-pair encoding produces a non-trivial vocabulary and a transformer's loss curve
  is informative.

The notebook downloads this file on first run and caches it here.
