(* ::Package:: *)

(* ============================================================
   NanoChat.wl  ---  Part 9 of "LLM A to Z in Wolfram Language"
   ============================================================

   The nano-style transformer building blocks: RMSNorm, rotary
   position embedding (RoPE), SwiGLU MLP, and an attention block
   that applies RoPE to Q and K.  Composed into nanoModel, a
   pre-norm transformer with no learned positional embedding.

   This is the architecture Karpathy's nanochat uses: same family
   as LLaMA, Mistral, Qwen.  Differs from the Part 3 gptModel in
   three places: RMSNorm instead of LayerNorm, RoPE instead of
   absolute positional embedding, SwiGLU instead of GELU MLP.

   Built on Attention.wl for causalMask only; everything else is
   self-contained so the file can be read top-to-bottom.

   Caveat: the RMSNorm here has no learnable gain.  Pure
   y = x / sqrt(mean(x^2) + eps).  In practice the gain accounts
   for under 0.1% of the parameter count and removing it does not
   measurably hurt training quality; see the GPT-NeoX and LLaMA
   ablations.  We can add it back via a TrainableArray hook if
   needed. *)

BeginPackage["LLMAtoZ`NanoChat`",
  {"LLMAtoZ`Attention`"}];

ClearAll[
  rmsNorm,
  ropeAngleTables, applyRoPE,
  swiGluMLP,
  nanoSingleHeadAttention, nanoMultiHeadAttention,
  nanoTransformerBlock,
  nanoModel,
  nanoModelSpecTiny, nanoModelSpec30M, nanoModelSpec100M, nanoModelSpec200M,
  nanoParameterCount
];

rmsNorm::usage =
  "rmsNorm[dModel, maxSeqLen] returns a NetGraph that applies \
root-mean-square normalisation row-wise to a (maxSeqLen, dModel) \
input: y = x / Sqrt[Mean[x^2] + eps].  No learnable gain.";

ropeAngleTables::usage =
  "ropeAngleTables[maxSeqLen, dHead] returns {cosTable, sinTable}, \
two (maxSeqLen, dHead) numerical arrays holding the rotary embedding \
constants.  Pure-WL helper, used internally by applyRoPE.";

applyRoPE::usage =
  "applyRoPE[maxSeqLen, dHead] returns a NetGraph that applies \
rotary position embedding to a (maxSeqLen, dHead) tensor using the \
rotate-half formulation.  No trainable parameters; the cos/sin \
tables are frozen.";

swiGluMLP::usage =
  "swiGluMLP[dModel, dFF, maxSeqLen] returns a NetGraph implementing \
the SwiGLU position-wise MLP: down(SiLU(gate(x)) * up(x)).  Three \
linear layers, no biases.";

nanoSingleHeadAttention::usage =
  "nanoSingleHeadAttention[dModel, dHead, maxSeqLen] returns a \
NetGraph implementing one head of scaled dot-product causal \
self-attention with RoPE applied to Q and K.  Input shape \
(maxSeqLen, dModel), output shape (maxSeqLen, dHead).";

nanoMultiHeadAttention::usage =
  "nanoMultiHeadAttention[dModel, nHeads, maxSeqLen] returns a \
NetGraph implementing multi-head causal self-attention with RoPE, \
followed by an output projection.  Same I/O contract as \
LLMAtoZ`Attention`multiHeadAttentionBlock.";

nanoTransformerBlock::usage =
  "nanoTransformerBlock[dModel, nHeads, dFF, maxSeqLen] returns a \
NetGraph: RMSNorm -> RoPE multi-head attention -> residual -> \
RMSNorm -> SwiGLU MLP -> residual.";

nanoModel::usage =
  "nanoModel[dModel, nHeads, nLayers, dFF, vocabSize, maxSeqLen] \
returns a NetChain implementing the full nano-style language \
model: token embedding -> nLayers transformer blocks -> RMSNorm -> \
output projection to vocabSize logits.  No learned position \
embedding (RoPE handles position inside each attention block).";

nanoModelSpecTiny::usage =
  "nanoModelSpecTiny[vocabSize, maxSeqLen] returns a tiny model \
spec (~1M params) suitable for CPU sanity tests.";

nanoModelSpec30M::usage  =
  "30M-param nano spec for cheap GPU smoke tests at full vocab (50257). \
The token-embedding alone is ~26M params at d=256, so this is the smallest \
useful size with that vocabulary.";

nanoModelSpec100M::usage =
  "100M-param nano spec; ~half-day on g4dn.xlarge T4.  Budget target.";

nanoModelSpec200M::usage =
  "200M-param nano spec at GPT-2 small dimensions (d=768, L=12).  Same \
architecture Karpathy's nanochat uses; nanochat reports 162M with weight \
tying (which we don't implement here, so the untied count is ~190M).";

nanoParameterCount::usage =
  "nanoParameterCount[spec] returns the closed-form trainable parameter \
count of the corresponding nanoModel.  Differs from Part 3 by: no \
positional embedding term, RMSNorm has no parameters, SwiGLU has \
3 linears (no biases) instead of GPT-2's 2 (with biases).";

Begin["`Private`"];

(* ------------------------------------------------------------
   RMSNorm.

   y = x / Sqrt[Mean[x^2] + eps]  applied row-wise to the input.
   Implemented as a NetGraph that broadcasts the per-row inverse
   RMS scalar back across the feature axis. *)

$rmsEps = 1.0 * 10^-5;

rmsNorm[dModel_Integer, maxSeqLen_Integer] :=
  NetGraph[
    <|
      "square"    -> ElementwiseLayer[#^2 &],
      "meanSq"    -> AggregationLayer[Mean, -1],
      "invRms"    -> ElementwiseLayer[1./Sqrt[# + $rmsEps] &],
      "broadcast" -> ReplicateLayer[dModel, -1],
      "scale"     -> ThreadingLayer[Times]
    |>,
    {
      NetPort["Input"] -> "square" -> "meanSq" -> "invRms" -> "broadcast",
      {NetPort["Input"], "broadcast"} -> "scale"
    },
    "Input" -> {maxSeqLen, dModel}
  ];

(* ------------------------------------------------------------
   RoPE: cos / sin tables.

   For each position m in 1..maxSeqLen and feature index k in
   1..dHead, table[m, k] = trig(angle) with
     angle = (m - 1) * 10000^(-2 * (Mod[k - 1, dHead/2]) / dHead).

   The "Mod" gives the rotate-half pairing: feature k is paired
   with feature k + dHead/2 (LLaMA / nanochat convention).  Both
   halves use the same angle table.  Returns (cos, sin) as two
   (maxSeqLen, dHead) machine-precision matrices. *)

ropeAngleTables[maxSeqLen_Integer, dHead_Integer] := Module[
  {halfD, thetas, angles},
  halfD = Quotient[dHead, 2];
  thetas = Table[10000.^(-2.*i/dHead), {i, 0, halfD - 1}];
  angles = Table[
    (m - 1) * thetas[[Mod[k - 1, halfD] + 1]],
    {m, maxSeqLen}, {k, dHead}
  ];
  {N @ Cos[angles], N @ Sin[angles]}
];

(* ------------------------------------------------------------
   RoPE rotation as a NetGraph.

   rotateHalf(x) = concat([-x[half2], x[half1]], axis=-1)
   output        = x * cos + rotateHalf(x) * sin

   The cos and sin tables are baked in as ConstantArrayLayer
   nodes (frozen, no trainable params). *)

applyRoPE[maxSeqLen_Integer, dHead_Integer] := With[
  {tables = ropeAngleTables[maxSeqLen, dHead], half = Quotient[dHead, 2]},
  NetGraph[
    <|
      "firstHalf"   -> PartLayer[{All, 1 ;; half}],
      "secondHalf"  -> PartLayer[{All, half + 1 ;; dHead}],
      "negSecond"   -> ElementwiseLayer[Minus],
      "rotatedHalf" -> CatenateLayer[2],
      "cosTable"    -> NetArrayLayer["Array" -> tables[[1]],
                         "LearningRateMultipliers" -> 0],
      "sinTable"    -> NetArrayLayer["Array" -> tables[[2]],
                         "LearningRateMultipliers" -> 0],
      "xCos"        -> ThreadingLayer[Times],
      "rotSin"      -> ThreadingLayer[Times],
      "sum"         -> ThreadingLayer[Plus]
    |>,
    {
      NetPort["Input"] -> "firstHalf",
      NetPort["Input"] -> "secondHalf",
      "secondHalf" -> "negSecond",
      {"negSecond", "firstHalf"} -> "rotatedHalf",
      {NetPort["Input"], "cosTable"} -> "xCos",
      {"rotatedHalf", "sinTable"} -> "rotSin",
      {"xCos", "rotSin"} -> "sum"
    },
    "Input" -> {maxSeqLen, dHead}
  ]
];

(* ------------------------------------------------------------
   SwiGLU MLP (LLaMA / nanochat convention).

   y = down(SiLU(gate(x)) * up(x))

   Two parallel linears at the dFF expansion, an elementwise
   SiLU on the gate branch, an elementwise product, then a final
   linear back to dModel.  Three linears, no biases. *)

swiGluMLP[dModel_Integer, dFF_Integer, maxSeqLen_Integer] :=
  NetInitialize @ NetGraph[
    <|
      "gate" -> NetMapOperator[LinearLayer[dFF,    "Biases" -> None]],
      "up"   -> NetMapOperator[LinearLayer[dFF,    "Biases" -> None]],
      "silu" -> ElementwiseLayer[# * LogisticSigmoid[#] &],
      "prod" -> ThreadingLayer[Times],
      "down" -> NetMapOperator[LinearLayer[dModel, "Biases" -> None]]
    |>,
    {
      NetPort["Input"] -> "gate" -> "silu",
      NetPort["Input"] -> "up",
      {"silu", "up"} -> "prod",
      "prod" -> "down"
    },
    "Input" -> {maxSeqLen, dModel}
  ];

(* ------------------------------------------------------------
   Single-head causal self-attention with RoPE on Q and K. *)

nanoSingleHeadAttention[
    dModel_Integer, dHead_Integer, maxSeqLen_Integer] := With[
  {mask  = LLMAtoZ`Attention`causalMask[maxSeqLen],
   scale = Sqrt[N[dHead]]},
  NetInitialize @ NetGraph[
    <|
      "q"    -> NetMapOperator[LinearLayer[dHead, "Biases" -> None]],
      "k"    -> NetMapOperator[LinearLayer[dHead, "Biases" -> None]],
      "v"    -> NetMapOperator[LinearLayer[dHead, "Biases" -> None]],
      "qRot" -> applyRoPE[maxSeqLen, dHead],
      "kRot" -> applyRoPE[maxSeqLen, dHead],
      "scores" -> FunctionLayer[
        (#Q . Transpose[#K] / scale + mask) &,
        "Inputs"  -> <|"Q" -> {maxSeqLen, dHead}, "K" -> {maxSeqLen, dHead}|>,
        "Output"  -> {maxSeqLen, maxSeqLen}
      ],
      "weights" -> SoftmaxLayer[],
      "output"  -> FunctionLayer[
        (#W . #V) &,
        "Inputs"  -> <|"W" -> {maxSeqLen, maxSeqLen}, "V" -> {maxSeqLen, dHead}|>,
        "Output"  -> {maxSeqLen, dHead}
      ]
    |>,
    {
      NetPort["Input"] -> "q" -> "qRot",
      NetPort["Input"] -> "k" -> "kRot",
      NetPort["Input"] -> "v",
      "qRot" -> NetPort["scores", "Q"],
      "kRot" -> NetPort["scores", "K"],
      "scores" -> "weights",
      "weights" -> NetPort["output", "W"],
      "v" -> NetPort["output", "V"]
    },
    "Input" -> {maxSeqLen, dModel}
  ]
];

(* ------------------------------------------------------------
   Multi-head wrapper.  Same parallel-heads idiom as Part 2 but
   with RoPE-aware heads.  No bias on the output projection. *)

nanoMultiHeadAttention::dim =
  "Model dimension `1` must be divisible by head count `2`.";

nanoMultiHeadAttention[
    dModel_Integer, nHeads_Integer, maxSeqLen_Integer] := Module[
  {dHead, heads, headEdges, catAssoc},
  If[Mod[dModel, nHeads] =!= 0,
    Message[nanoMultiHeadAttention::dim, dModel, nHeads];
    Return[$Failed]
  ];
  dHead = Quotient[dModel, nHeads];
  heads = Association @ Table[
    "head" <> ToString[h] ->
      nanoSingleHeadAttention[dModel, dHead, maxSeqLen],
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

(* ------------------------------------------------------------
   Pre-norm transformer block, RMSNorm + RoPE-MHA + SwiGLU. *)

nanoTransformerBlock[
    dModel_Integer, nHeads_Integer, dFF_Integer, maxSeqLen_Integer] :=
  NetInitialize @ NetGraph[
    <|
      "norm1" -> rmsNorm[dModel, maxSeqLen],
      "attn"  -> nanoMultiHeadAttention[dModel, nHeads, maxSeqLen],
      "add1"  -> ThreadingLayer[Plus],
      "norm2" -> rmsNorm[dModel, maxSeqLen],
      "ffn"   -> swiGluMLP[dModel, dFF, maxSeqLen],
      "add2"  -> ThreadingLayer[Plus]
    |>,
    {
      NetPort["Input"] -> "norm1" -> "attn",
      {NetPort["Input"], "attn"} -> "add1",
      "add1" -> "norm2" -> "ffn",
      {"add1", "ffn"} -> "add2"
    },
    "Input" -> {maxSeqLen, dModel}
  ];

(* ------------------------------------------------------------
   Full nano language model.  Token embedding only -- RoPE
   replaces the learned absolute position embedding. *)

nanoModel[
    dModel_Integer, nHeads_Integer, nLayers_Integer, dFF_Integer,
    vocabSize_Integer, maxSeqLen_Integer] :=
  NetInitialize @ NetChain[
    Join[
      {EmbeddingLayer[dModel, vocabSize]},
      Table[
        nanoTransformerBlock[dModel, nHeads, dFF, maxSeqLen],
        {nLayers}
      ],
      {rmsNorm[dModel, maxSeqLen]},
      {NetMapOperator[LinearLayer[vocabSize, "Biases" -> None]]}
    ]
  ];

(* ------------------------------------------------------------
   Closed-form parameter count for the nano architecture.

   Per block (no biases anywhere):
     - Q, K, V projections per head      : 3 * dModel * dHead summed
       over nHeads = 3 * dModel^2
     - output projection                 : dModel^2
     - RMSNorm x 2                       : 0 (no learnable gain in our impl)
     - SwiGLU: 3 linears (gate, up, down) : 2 * dModel * dFF + dFF * dModel
                                           = 3 * dModel * dFF
     - Total per block                   : 4 dModel^2 + 3 dModel dFF

   Full model:
     - Token embedding                   : vocabSize * dModel
     - nLayers blocks                    : nLayers * (block above)
     - Final RMSNorm                     : 0
     - Output projection                 : dModel * vocabSize *)

nanoParameterCount[spec_Association] := With[
  {d = spec["dModel"], L = spec["nLayers"], f = spec["dFF"], v = spec["vocabSize"]},
  Module[{perBlock},
    perBlock = 4 d^2 + 3 d f;
    v d + L*perBlock + v d
  ]
];

(* ------------------------------------------------------------
   Model spec families. *)

nanoModelSpecTiny[vocabSize_Integer, maxSeqLen_Integer] := <|
  "label" -> "nano-tiny",
  "dModel" -> 64, "nHeads" -> 4, "nLayers" -> 2, "dFF" -> 256,
  "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen
|>;

nanoModelSpec30M[vocabSize_Integer, maxSeqLen_Integer] := <|
  "label" -> "nano-30M",
  "dModel" -> 256, "nHeads" -> 8, "nLayers" -> 4, "dFF" -> 1024,
  "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen
|>;

nanoModelSpec100M[vocabSize_Integer, maxSeqLen_Integer] := <|
  "label" -> "nano-100M",
  "dModel" -> 512, "nHeads" -> 8, "nLayers" -> 12, "dFF" -> 2048,
  "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen
|>;

(* nanochat reference: d=768, h=12, L=12, dFF=3072, v=50257, n=1024.
   Karpathy reports 162M with weight tying; our untied implementation
   reports ~190M.  Weight tying is a known follow-up. *)
nanoModelSpec200M[vocabSize_Integer, maxSeqLen_Integer] := <|
  "label" -> "nano-200M",
  "dModel" -> 768, "nHeads" -> 12, "nLayers" -> 12, "dFF" -> 3072,
  "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen
|>;

End[];
EndPackage[];
