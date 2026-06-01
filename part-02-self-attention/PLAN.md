# Part 2 plan — Self-attention and the QKV picture

**Status:** drafting. Not yet a Wolfram Community post.

## Goal of this part

Build scaled dot-product attention from first principles, derive the
`1/√d_k` scaling factor from a variance argument, extend to multi-head
attention, then visualise attention heatmaps on real text using a
*pretrained* GPT-2 small so the reader sees what learned attention
structure actually looks like before they train their own in Part 4.

## What stays distinct from Raschka chapter 3

Per the audit done after Part 1, Raschka covers attention extensively
in chapter 3 of *Build a Large Language Model from Scratch*. The
important pedagogical opportunity for us is that **he does not derive
the `1/√d_k` factor** — he states the formula. We will derive it from
the variance of a dot product of two zero-mean unit-variance vectors of
length `d_k`. That derivation is the heart of Part 2 and the single
biggest signature of "this is a teaching post, not a code dump."

Other distinctness moves:

- Function/class names: `causalSelfAttention`, `multiHeadAttentionBlock`,
  `splitHeads`, `mergeHeads`, `attentionScores`. None overlap with
  Raschka's `SelfAttention_v1/v2`, `CausalAttention`,
  `MultiHeadAttentionWrapper`, `MultiHeadAttention`.
- Variable names: `wQ`, `wK`, `wV` for the projection weights (he uses
  `W_query`, `W_key`, `W_value`).
- Worked example: a single sentence from Tiny Shakespeare ("To be or
  not to be") fed through GPT-2 small via WL's `NetModel`. He uses
  his own from-scratch-trained attention; ours uses real pretrained
  weights.

## Section outline

1. **Why attention?** A token wants to look at the other tokens in
   the sequence and pull in information from them, weighted by
   relevance. Each weight is itself learned from the content of the
   token pair. That is all an attention layer is.

2. **Naive version: a weighted sum with hand-picked weights.** Just
   to fix what we are aiming at, do one weighted sum of a 4-token
   sequence with weights given by hand. WL: a single `Dot` call.

3. **Learnable queries, keys, values.** Each input vector `x_i` is
   projected three times by learned linear maps to produce a *query*,
   a *key*, and a *value*. The query of token `i` is dotted with the
   key of token `j` to produce the compatibility score
   `s_{ij} = q_i · k_j`. The softmax of those scores along `j` gives
   the attention weights. The output for token `i` is a weighted sum
   of the values, using those weights.

   Intuition (different from Raschka): the query is the question a
   token asks of the others; the key is the answer it provides; the
   value is what it actually passes along. They are three separate
   learned views of the same input.

4. **The `1/√d_k` factor — derivation, not assertion.**
   Suppose `q, k ∈ ℝ^{d_k}` with each component drawn from a
   zero-mean unit-variance distribution. Then `q · k = Σ q_i k_i` is a
   sum of `d_k` independent products with `E[q_i k_i] = 0` and
   `Var(q_i k_i) = 1`. By independence, `Var(q · k) = d_k`. So the
   pre-softmax logits have standard deviation `√d_k`, which means
   that for large `d_k` the softmax is fed inputs with very large
   spread, saturating it: the largest logit dominates and all others
   are crushed to zero. Gradient through such a saturated softmax is
   essentially zero — the gradient pathology that prevents the
   network from learning.

   Dividing by `√d_k` returns the pre-softmax logits to unit variance
   and keeps the softmax in its high-gradient regime. Show this
   empirically: histogram of `q · k` for `d_k ∈ {16, 64, 256}` with
   and without the scaling factor. (Figure.)

5. **Causal masking.** A language model can only attend to past
   tokens, not future ones. Implement by adding a strict-upper-
   triangular matrix of `-∞` to the pre-softmax score matrix; after
   softmax those entries become zero exactly. In WL:
   `UpperTriangularize[ConstantArray[-Infinity, {n, n}], 1]`.
   The triangular indexing here uses WL's 1-based convention; the
   second argument `1` means "above the main diagonal".

6. **Putting it together: single-head causal attention as a `NetGraph`.**
   Six nodes:
   - `Q`, `K`, `V`: three `LinearLayer[d_head]` projections of the
     input.
   - `scores`: `FunctionLayer[(#q . Transpose[#k]) / Sqrt[d_head] &]`.
   - `masked`: `ThreadingLayer[Plus]` of `scores` and the causal mask.
   - `weights`: `SoftmaxLayer["Input" -> "Vector"]` along the key axis.
   - `output`: `FunctionLayer[#weights . #v &]`.

   Render the `NetGraph` so the reader can see the data flow at a
   glance — one of the WL/PyTorch differentiators.

7. **Multi-head attention.** Run `n_heads` parallel single-head
   attentions on different subspaces of the same input. Conceptually:
   - Project input to `d_model = n_heads × d_head`.
   - Reshape `(seq_len, d_model) → (seq_len, n_heads, d_head)`.
   - For each head independently: compute scaled dot-product attention.
   - Concatenate the heads back to `(seq_len, d_model)`.
   - Apply a final output projection `W_o ∈ ℝ^{d_model × d_model}`.

   Implementation: use `ReshapeLayer[{n_heads, d_head}]` and
   `TransposeLayer[1 <-> 2]` to swap sequence and head axes so the
   per-head attention can be expressed as a single `FunctionLayer`
   call broadcast over the heads dimension.

8. **What real attention looks like: GPT-2 small heatmaps.**
   Load `NetModel["GPT-2 Transformer Trained on WebText Data"]`. For
   a short prompt ("To be or not to be" works well — short, familiar,
   has repeated tokens), extract the attention weight tensors from
   each of the 12 layers × 12 heads using `NetExtract`. Display a
   3×4 grid of heatmaps for one selected layer.

   The story to tell: heads specialise. One looks at the previous
   token (the "left-shift" head). One attends to syntactically
   related words across the sentence. One is approximately uniform,
   functioning as a no-op or as a positional broadcast.

9. **Summary & onward.** We finish Part 2 with a from-scratch
   `multiHeadAttentionBlock[d_model, n_heads]` constructor and a
   gallery of real attention heatmaps. In Part 3 we will wrap this
   block in pre-norm + residual + feed-forward, stack `n_layers` of
   them, and arrive at the canonical transformer block.

## Deliverables

- `src/Attention.wl` — package with `causalSelfAttention`,
  `multiHeadAttentionBlock`, `attentionScores`, `causalMask`, `splitHeads`,
  `mergeHeads`, `gpt2AttentionWeights[prompt, layer, head]`.
- `figures/qkdotproduct_variance.png` — empirical histogram showing
  the variance argument for `1/√d_k`.
- `figures/causal_mask_visualisation.png` — the triangular mask, plus
  before/after softmax of a small score matrix.
- `figures/gpt2_heads_gallery.png` — 12-head gallery of attention
  heatmaps on a chosen prompt, one selected layer.
- `figures/gpt2_head_specialisation.png` — three hand-picked heads
  illustrating the previous-token / syntactic / uniform pattern.
- `figures/illustration_attention_hero.png` — generated illustration:
  the QKV picture as a clean schematic for the post header.
- `notebook/Part02.nb` — the Wolfram Community post.
- `POST.md` — markdown draft.

## Compute backend

This part runs entirely on a single CPU; the heaviest operation is a
single forward pass through GPT-2 small (~117M params), well within
Mac mini memory. No `RemoteBatchSubmit` needed until Part 4.

## Open questions

- Do we cover **dropout on attention weights** in this part, or defer
  to Part 4 when we actually train? Raschka introduces it in
  chapter 3; I lean toward mentioning the parameter in the
  `multiHeadAttentionBlock` constructor but disabling it for the
  pedagogical demo, with a one-paragraph explanation.
- Do we cover **qkv_bias**? Default to no bias on Q/K/V projections
  (matches GPT-2 architecture) and note it as a one-line option.
