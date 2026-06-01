(* ::Package:: *)

(* ============================================================
   Attention.wl  ---  Part 2 of "LLM A to Z in Wolfram Language"
   ============================================================

   Scaled dot-product attention and multi-head attention, both
   as NetGraphs, plus the small helpers a teaching post needs:
   the causal mask, an explicit scoring function, head split/
   merge utilities, and a wrapper that pulls per-layer per-head
   attention weights out of a pretrained GPT-2 small.

   No external libraries. Every layer is either a built-in WL
   NetLayer or a transparent FunctionLayer; nothing is hidden
   behind a custom Block primitive. *)

BeginPackage["LLMAtoZ`Attention`"];

ClearAll[
  causalMask, attentionScores,
  singleHeadCausalAttention, multiHeadAttentionBlock,
  splitHeads, mergeHeads,
  softmaxRows, gpt2AttentionWeights, gpt2EncodePrompt
];

causalMask::usage =
  "causalMask[n] returns an n\[Times]n matrix with zeros on and below \
the main diagonal and a large negative number above it. Added to the \
pre-softmax score matrix it zeroes out the attention weight of any \
future token after the softmax.";

attentionScores::usage =
  "attentionScores[q, k] returns q.k\[Transpose] / Sqrt[d_k] where d_k \
is the trailing dimension of q. This is the unmasked, unscaled-by-softmax \
compatibility score matrix.";

singleHeadCausalAttention::usage =
  "singleHeadCausalAttention[dModel, dHead, maxSeqLen] returns a \
NetGraph implementing one head of scaled dot-product causal self-attention. \
Input shape: (maxSeqLen, dModel). Output shape: (maxSeqLen, dHead).";

multiHeadAttentionBlock::usage =
  "multiHeadAttentionBlock[dModel, nHeads, maxSeqLen] returns a NetGraph \
implementing multi-head causal self-attention with nHeads parallel heads \
each of width dModel/nHeads, followed by a final output projection back \
to dModel. Input/output shape: (maxSeqLen, dModel).";

splitHeads::usage =
  "splitHeads[nHeads] returns a NetChain that reshapes a (seqLen, dModel) \
tensor to (nHeads, seqLen, dHead) where dHead = dModel/nHeads. Used inside \
multiHeadAttentionBlock.";

mergeHeads::usage =
  "mergeHeads[nHeads] is the inverse of splitHeads.";

softmaxRows::usage =
  "softmaxRows[matrix] applies softmax to each row of a 2D real matrix.";

gpt2AttentionWeights::usage =
  "gpt2AttentionWeights[gptNet, tokenIds, layer] returns the attention \
weight tensor of shape (nHeads, seqLen, seqLen) for the requested \
transformer block of a pretrained WL GPT-2 net. The model is assumed to \
be the 12-layer 12-head 768-dim variant returned by \
NetModel[\"GPT-2 Transformer Trained on WebText Data\", \"Task\" -> \"LanguageModeling\"].";

gpt2EncodePrompt::usage =
  "gpt2EncodePrompt[gptNet, prompt] runs the WL GPT-2 net's input encoder \
on a string prompt and returns the list of integer token IDs.";

Begin["`Private`"];

(* ------------------------------------------------------------
   The causal mask.

   We use a large negative value rather than -Infinity because
   the latter participates in NaN-producing arithmetic during
   the dot product that builds the score matrix.  After softmax
   the masked entries become numerically zero either way. *)

$negativeInfinity = -1.0 * 10^9;

causalMask[n_Integer] :=
  UpperTriangularize[ConstantArray[$negativeInfinity, {n, n}], 1];

(* ------------------------------------------------------------
   Explicit scoring (handy for the variance-argument figure). *)

attentionScores[q_, k_] := Module[{dk = Last[Dimensions[q]]},
  q . Transpose[k] / Sqrt[N[dk]]
];

(* ------------------------------------------------------------
   Single-head causal self-attention as a NetGraph.

   Six nodes:

     q, k, v     three learnable LinearLayer projections of the input
     scores      Q.K^T / Sqrt[d_head]  +  causal mask
     weights     row-wise softmax of the masked scores
     output      weights . V

   Input is at NetPort["Input"] of shape (seqLen, dModel); output is
   at NetPort["Output"] of shape (seqLen, dHead). *)

singleHeadCausalAttention[
    dModel_Integer, dHead_Integer, maxSeqLen_Integer] := With[
  {mask = causalMask[maxSeqLen], scale = Sqrt[N[dHead]]},
  NetInitialize @ NetGraph[
    <|
      "q" -> NetMapOperator[LinearLayer[dHead, "Biases" -> None]],
      "k" -> NetMapOperator[LinearLayer[dHead, "Biases" -> None]],
      "v" -> NetMapOperator[LinearLayer[dHead, "Biases" -> None]],
      "scores" -> FunctionLayer[
        (#Q . Transpose[#K] / scale + mask) &,
        "Inputs" -> <|"Q" -> {maxSeqLen, dHead}, "K" -> {maxSeqLen, dHead}|>,
        "Output" -> {maxSeqLen, maxSeqLen}
      ],
      "weights" -> SoftmaxLayer[],
      "output" -> FunctionLayer[
        (#W . #V) &,
        "Inputs" -> <|"W" -> {maxSeqLen, maxSeqLen}, "V" -> {maxSeqLen, dHead}|>,
        "Output" -> {maxSeqLen, dHead}
      ]
    |>,
    {
      NetPort["Input"] -> "q",
      NetPort["Input"] -> "k",
      NetPort["Input"] -> "v",
      "q" -> NetPort["scores", "Q"],
      "k" -> NetPort["scores", "K"],
      "scores" -> "weights",
      "weights" -> NetPort["output", "W"],
      "v" -> NetPort["output", "V"]
    },
    "Input" -> {maxSeqLen, dModel}
  ]
];

(* ------------------------------------------------------------
   Head split / merge utilities for the multi-head block. *)

splitHeads[nHeads_Integer] := NetChain[{
  (* (seqLen, dModel) -> (seqLen, nHeads, dHead) *)
  ReshapeLayer[{Automatic, nHeads, Automatic}],
  (* -> (nHeads, seqLen, dHead) *)
  TransposeLayer[1 <-> 2]
}];

mergeHeads[nHeads_Integer] := NetChain[{
  (* (nHeads, seqLen, dHead) -> (seqLen, nHeads, dHead) *)
  TransposeLayer[1 <-> 2],
  (* -> (seqLen, dModel) *)
  ReshapeLayer[{Automatic, Automatic}]  (* flatten last two *)
}];

(* ------------------------------------------------------------
   Multi-head attention block.

   For exposition simplicity we implement the multi-head block as a
   NetGraph that holds nHeads parallel singleHeadCausalAttention
   subgraphs, then concatenates their outputs and applies a final
   output projection.  A production version would project Q, K, V
   once at dModel width and reshape into heads; the parallel-heads
   version is easier to read and has identical I/O. *)

multiHeadAttentionBlock[
    dModel_Integer, nHeads_Integer, maxSeqLen_Integer] := Module[
  {dHead, heads, headEdges, catAssoc},
  If[Mod[dModel, nHeads] =!= 0,
    Message[multiHeadAttentionBlock::dim, dModel, nHeads];
    Return[$Failed]
  ];
  dHead = Quotient[dModel, nHeads];
  heads = Association @ Table[
    "head" <> ToString[h] ->
      singleHeadCausalAttention[dModel, dHead, maxSeqLen],
    {h, nHeads}
  ];
  headEdges = Table[
    NetPort["Input"] -> "head" <> ToString[h], {h, nHeads}
  ];
  catAssoc = <|
    heads,
    "concat" -> CatenateLayer[2],
    "proj"   -> NetMapOperator[LinearLayer[dModel, "Biases" -> None]]
  |>;
  NetInitialize @ NetGraph[
    catAssoc,
    Join[
      headEdges,
      {(Table["head" <> ToString[h], {h, nHeads}]) -> "concat"},
      {"concat" -> "proj"}
    ],
    "Input" -> {maxSeqLen, dModel}
  ]
];

multiHeadAttentionBlock::dim =
  "Model dimension `1` must be divisible by head count `2`.";

(* ------------------------------------------------------------
   Numerically-safe row-wise softmax (subtract max for stability). *)

softmaxRows[m_?MatrixQ] := Module[{shifted, ex},
  shifted = m - Max /@ m;
  ex = Exp[shifted];
  ex / Total /@ ex
];

(* ------------------------------------------------------------
   Pulling per-head attention weights out of WL's pretrained GPT-2.

   GPT-2 in WL is a 5-layer NetChain:
     1. embedding NetGraph (token + positional)
     2. NetChain of 12 transformer blocks + final LayerNorm
     3. SequenceLastLayer
     4. output LinearLayer
     5. SoftmaxLayer

   Each transformer block is a NetChain of two sub-graphs (attention,
   feed-forward).  The attention sub-graph has six layers:
     1. NetMapOperator wrapping the Q projection
     2. NetMapOperator wrapping the K projection
     3. ElementwiseLayer (scaling)
     4. NetMapOperator wrapping the V projection
     5. AttentionLayer (combined softmax + value gather)
     6. NetMapOperator wrapping the output projection

   The internal AttentionLayer does not expose its softmax output,
   so we recompute the attention weight matrix by hand: apply the
   block's pre-norm, project to Q and K using the extracted weights,
   reshape into heads, do the scaled dot product, add the causal mask,
   and softmax row-wise. *)

$gpt2NHeads = 12;
$gpt2DHead  = 64;
$gpt2DModel = 768;

gpt2EncodePrompt[gpt_, prompt_String] := With[
  {enc = NetExtract[gpt, "Input"]},
  enc[prompt]
];

(* Run the model up to (but not including) the given transformer block,
   returning the residual-stream tensor that feeds into that block. *)
preBlockHidden[gpt_, ids_List, layer_Integer] := Module[{embed, h, stack},
  embed = NetExtract[gpt, {1}];
  h = embed[ids];
  If[layer > 1,
    stack = NetExtract[gpt, {2}];
    NetTake[stack, layer - 1][h],
    h
  ]
];

gpt2AttentionWeights[gpt_, ids_List, layer_Integer] := Module[
  {x, normLayer, normed, wQ, wK, q, k, qHeads, kHeads, mask, scores,
   nh = $gpt2NHeads, dh = $gpt2DHead, seqLen},
  seqLen = Length[ids];
  x = preBlockHidden[gpt, ids, layer];
  normLayer = NetExtract[gpt, {2, layer, 1, "norm"}];
  normed = normLayer[x];
  wQ = Normal @ NetExtract[gpt, {2, layer, 1, "attention", 1, "Net", "Weights"}];
  wK = Normal @ NetExtract[gpt, {2, layer, 1, "attention", 2, "Net", "Weights"}];
  q = normed . Transpose[wQ];
  k = normed . Transpose[wK];
  qHeads = Transpose[ArrayReshape[q, {seqLen, nh, dh}], {2, 1, 3}];
  kHeads = Transpose[ArrayReshape[k, {seqLen, nh, dh}], {2, 1, 3}];
  mask = causalMask[seqLen];
  scores = Table[
    qHeads[[h]] . Transpose[kHeads[[h]]] / Sqrt[N[dh]] + mask,
    {h, nh}
  ];
  softmaxRows /@ scores
];

End[];
EndPackage[];
