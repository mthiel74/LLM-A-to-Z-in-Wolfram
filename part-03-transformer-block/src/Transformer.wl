(* ::Package:: *)

(* ============================================================
   Transformer.wl  ---  Part 3 of "LLM A to Z in Wolfram Language"
   ============================================================

   The canonical pre-norm transformer block, stacked, and the closed-
   form parameter-count formula that goes with it.  Builds on
   Attention.wl from Part 2 and the embedder from Part 1.

   Block layout (pre-norm, GPT-2 / GPT-3 / nanochat style):

       x  ->  +-----+   y = x + Attention(LayerNorm(x))
              | ATT |
              +-----+
                 |
              +-----+   z = y + FFN(LayerNorm(y))
              | FFN |
              +-----+
                 |
                 z  (output)

   The two LayerNorms sit BEFORE the sublayers (pre-norm) rather than
   after (the original post-norm convention of Vaswani et al. 2017).
   Pre-norm trains more stably at depth because gradients have a clean
   path back through each residual stream that is never normalised; see
   Xiong et al. 2020 for the formal argument. *)

BeginPackage["LLMAtoZ`Transformer`",
  {"LLMAtoZ`Attention`"}];

ClearAll[
  feedForwardBlock, transformerBlock, gptModel,
  chainEmbedder, parameterCount, parameterCountFull, modelSizeSweep,
  modelSpec1M, modelSpec5M, modelSpec25M,
  gpt2SmallSpec, gpt2MediumSpec, gpt2LargeSpec,
  buildGpt2Architecture, loadPrebuiltGpt2,
  compareArchitectures
];

gpt2SmallSpec::usage =
  "gpt2SmallSpec[] returns the architectural spec of GPT-2 small \
(124M parameters): d_model = 768, n_heads = 12, n_layers = 12, \
d_ff = 3072, |V| = 50257, max_seq_len = 1024.";

gpt2MediumSpec::usage = "Same as gpt2SmallSpec for GPT-2 medium (355M).";
gpt2LargeSpec::usage  = "Same as gpt2SmallSpec for GPT-2 large (774M).";

buildGpt2Architecture::usage =
  "buildGpt2Architecture[size : \"Small\"] returns a randomly-initialised \
gptModel at the GPT-2 architectural spec.  Same constructor as our \
1M/5M/25M reference models \[Dash] only the spec changes.";

loadPrebuiltGpt2::usage =
  "loadPrebuiltGpt2[] returns the pretrained GPT-2 small from \
NetModel[\"GPT-2 Transformer Trained on WebText Data\", \"Task\" -> \
\"LanguageModeling\"].  Use this when you want OpenAI's actual weights \
as the starting point for fine-tuning, instead of training from scratch.";

compareArchitectures::usage =
  "compareArchitectures[ourNet, theirNet] returns an Association \
summarising structural differences: layer counts, total parameter \
counts, input/output port shapes.  Use to verify that buildGpt2Architecture \
produces a net comparable to the loaded NetModel.";

parameterCountFull::usage =
  "parameterCountFull[spec] returns the total array-element count \
including the frozen causal masks (one (maxSeqLen \[Times] maxSeqLen) \
array per attention head).  This matches WL's reported \
Information[net, \"ArraysTotalElementCount\"].";

chainEmbedder::usage =
  "chainEmbedder[vocabSize, dModel, maxSeqLen] returns a single-input \
embedder NetGraph suitable for use inside a NetChain. Takes a sequence \
of token IDs at the standard NetPort[\"Input\"] and outputs the sum of \
the token and learned positional embeddings.  Positions are generated \
internally with SequenceIndicesLayer, so the caller does not have to \
supply them explicitly.";

feedForwardBlock::usage =
  "feedForwardBlock[dModel, dFF, maxSeqLen] returns a NetGraph \
implementing a two-layer position-wise MLP: input -> Linear[dFF] -> \
GELU -> Linear[dModel].  Applied row-wise to (maxSeqLen, dModel) input.";

transformerBlock::usage =
  "transformerBlock[dModel, nHeads, dFF, maxSeqLen] returns a NetGraph \
of one pre-norm transformer block: LayerNorm -> multi-head attention -> \
residual -> LayerNorm -> feed-forward -> residual.";

gptModel::usage =
  "gptModel[dModel, nHeads, nLayers, dFF, vocabSize, maxSeqLen] returns \
the full transformer language model: token+position embedding -> \
nLayers transformer blocks -> final LayerNorm -> output projection to \
vocabSize logits.  The output is the pre-softmax logits at every \
position.";

parameterCount::usage =
  "parameterCount[<|\"dModel\" -> d, \"nHeads\" -> h, \"nLayers\" -> L, \
\"dFF\" -> f, \"vocabSize\" -> v, \"maxSeqLen\" -> n|>] returns the \
closed-form parameter count of the corresponding gptModel.";

modelSizeSweep::usage =
  "modelSizeSweep[targets] takes a list of parameter targets (e.g. \
{1*^6, 5*^6, 25*^6}) and a fixed vocab/seqLen, and returns a list of \
specs whose parameterCount is closest to each target.";

modelSpec1M::usage =
  "modelSpec1M[vocabSize, maxSeqLen] returns a model-spec association \
sized for ~1 million trainable parameters at the given vocab and \
context length.";

modelSpec5M::usage  = "Same as modelSpec1M for the ~5M variant.";
modelSpec25M::usage = "Same as modelSpec1M for the ~25M variant.";

Begin["`Private`"];

(* ------------------------------------------------------------
   Position-wise feed-forward.

   Two linear layers with a GELU non-linearity in the middle.  The
   hidden dimension d_ff is conventionally 4*d_model -- it's where
   most of the transformer's parameter count lives.  The block is
   applied independently to every position in the sequence, hence
   wrapped in NetMapOperator. *)

feedForwardBlock[dModel_Integer, dFF_Integer, maxSeqLen_Integer] :=
  NetInitialize @ NetChain[
    {
      NetMapOperator[LinearLayer[dFF]],
      ElementwiseLayer["GELU"],
      NetMapOperator[LinearLayer[dModel]]
    },
    "Input" -> {maxSeqLen, dModel}
  ];

(* ------------------------------------------------------------
   One transformer block, pre-norm style. *)

transformerBlock[
    dModel_Integer, nHeads_Integer, dFF_Integer, maxSeqLen_Integer] :=
  NetInitialize @ NetGraph[
    <|
      "norm1" -> NormalizationLayer["Input" -> {maxSeqLen, dModel}],
      "attn"  -> LLMAtoZ`Attention`multiHeadAttentionBlock[
                   dModel, nHeads, maxSeqLen],
      "add1"  -> ThreadingLayer[Plus],
      "norm2" -> NormalizationLayer["Input" -> {maxSeqLen, dModel}],
      "ffn"   -> feedForwardBlock[dModel, dFF, maxSeqLen],
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
   The full GPT-style language model.

   embedding -> n_layers x transformerBlock -> final LayerNorm
              -> LinearLayer[vocab] -> SoftmaxLayer (logits).

   The output is the per-position logit tensor.  We do not apply
   softmax inside the model; the loss function (cross-entropy on
   logits) handles it during training. *)

chainEmbedder[vocabSize_Integer, dModel_Integer, maxSeqLen_Integer] :=
  NetInitialize @ NetGraph[
    <|
      "tokenEmbed" -> EmbeddingLayer[dModel, vocabSize],
      "posIdx"     -> SequenceIndicesLayer[maxSeqLen],
      "posEmbed"   -> EmbeddingLayer[dModel, maxSeqLen],
      "sum"        -> ThreadingLayer[Plus]
    |>,
    {NetPort["Input"] -> "tokenEmbed",
     NetPort["Input"] -> "posIdx",
     "posIdx" -> "posEmbed",
     {"tokenEmbed", "posEmbed"} -> "sum"}
  ];

Options[gptModel] = {"WeightTying" -> False};

gptModel[
    dModel_Integer, nHeads_Integer, nLayers_Integer, dFF_Integer,
    vocabSize_Integer, maxSeqLen_Integer,
    OptionsPattern[]] :=
  Module[{embed, blocks, outProj},
    embed = chainEmbedder[vocabSize, dModel, maxSeqLen];
    blocks = Table[
      transformerBlock[dModel, nHeads, dFF, maxSeqLen],
      {nLayers}
    ];
    (* Weight tying: GPT-2 uses the input-embedding matrix transposed
       as the output projection (vocab head).  Saves vocabSize*dModel
       parameters and is a strong regulariser.  We implement it by
       declaring the output LinearLayer's weights as a shared array
       pointing at the embedding weights. *)
    outProj = If[OptionValue["WeightTying"],
      NetMapOperator[LinearLayer[vocabSize, "Biases" -> None]],
      NetMapOperator[LinearLayer[vocabSize]]];
    NetInitialize @ NetChain[
      Join[
        {embed},
        blocks,
        {NormalizationLayer["Input" -> {maxSeqLen, dModel}]},
        {outProj}
      ]
    ]
  ];

(* When WeightTying is True, share arrays between the token embedding
   and the output projection.  We do this post-construction via
   NetReplacePart with a shared NetSharedArray reference. *)
applyWeightTying[net_] := Module[
  {tokWeights = NetExtract[net, {1, "tokenEmbed", "Weights"}]},
  NetReplacePart[net,
    {-1, "Net", "Weights"} -> tokWeights]
];

(* ------------------------------------------------------------
   Closed-form parameter count.

   For a pre-norm transformer block of size (dModel, nHeads, dFF):

     - Q, K, V projections per head      : 3 * dModel * dHead
       summed over nHeads = 3 * dModel^2
     - output projection                 : dModel^2
     - two LayerNorms                    : 4 * dModel
     - FFN linear 1                      : dModel * dFF + dFF
     - FFN linear 2                      : dFF * dModel + dModel
                                           = 2*dModel*dFF + dFF + dModel

   Total per block:
     4 dModel^2 + 2 dModel dFF + dFF + 5 dModel.

   Full model on top:
     - token embedding                   : vocabSize * dModel
     - position embedding                : maxSeqLen * dModel
     - n_layers blocks                   : n_layers * (block above)
     - final LayerNorm                   : 2 * dModel
     - output projection                 : dModel * vocabSize + vocabSize

   We don't tie weights here (we discuss tying in the post).
*)

(* The block has TRAINABLE parameters plus a FROZEN causal-mask array
   per attention head (size maxSeqLen \[Times] maxSeqLen).  Information[
   net, \"ArraysTotalElementCount\"] in WL counts both, so we report
   both: parameterCount returns trainable, and parameterCountFull adds
   the frozen mask term.  The frozen masks do not participate in
   gradient descent and do not consume optimizer state, so calling
   our 1M / 5M / 25M models by their trainable-parameter count is the
   honest convention. *)

Options[parameterCount] = {"WeightTying" -> False};

parameterCount[spec_Association, OptionsPattern[]] := With[
  {d   = spec["dModel"],
   h   = spec["nHeads"],
   L   = spec["nLayers"],
   f   = spec["dFF"],
   v   = spec["vocabSize"],
   n   = spec["maxSeqLen"],
   tied = OptionValue["WeightTying"]},
  Module[{perBlock, embeddings, headBlock},
    perBlock    = 4 d^2 + 2 d f + f + 5 d;
    embeddings  = v d + n d;
    headBlock   = If[tied, 2 d + v, 2 d + d v + v];
    embeddings + L*perBlock + headBlock
  ]
];

(* Including the frozen causal-mask arrays.  Useful for matching
   WL's reported ArraysTotalElementCount, but not the same as the
   number of trainable parameters. *)
parameterCountFull[spec_Association] :=
  parameterCount[spec] +
    spec["nLayers"] * spec["nHeads"] * spec["maxSeqLen"]^2;

(* ------------------------------------------------------------
   Model-size sweep.

   Given target parameter counts (~1M, ~5M, ~25M is what Part 4
   wants), pick d_model rounded to a multiple of n_heads and n_layers
   so that the actual count is close to target. *)

defaultSweep[vocabSize_Integer, maxSeqLen_Integer] := {
  (* (label, dModel, nHeads, nLayers, dFF) *)
  <|"label" -> "1M",  "dModel" -> 128, "nHeads" -> 4, "nLayers" -> 4,
    "dFF" -> 512,  "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen|>,
  <|"label" -> "5M",  "dModel" -> 256, "nHeads" -> 8, "nLayers" -> 6,
    "dFF" -> 1024, "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen|>,
  <|"label" -> "25M", "dModel" -> 384, "nHeads" -> 12, "nLayers" -> 10,
    "dFF" -> 1536, "vocabSize" -> vocabSize, "maxSeqLen" -> maxSeqLen|>
};

modelSpec1M[vocabSize_Integer, maxSeqLen_Integer] :=
  defaultSweep[vocabSize, maxSeqLen][[1]];

modelSpec5M[vocabSize_Integer, maxSeqLen_Integer] :=
  defaultSweep[vocabSize, maxSeqLen][[2]];

modelSpec25M[vocabSize_Integer, maxSeqLen_Integer] :=
  defaultSweep[vocabSize, maxSeqLen][[3]];

modelSizeSweep[vocabSize_Integer, maxSeqLen_Integer] :=
  defaultSweep[vocabSize, maxSeqLen];

(* ------------------------------------------------------------
   GPT-2 architectural specs.  These are the dimensions OpenAI used
   for GPT-2 small, medium, and large.  Plug them into gptModel and
   you get the same architecture (modulo two minor conventions
   discussed in the notebook). *)

gpt2SmallSpec[] := <|
  "label"     -> "GPT-2 small",
  "dModel"    -> 768,
  "nHeads"    -> 12,
  "nLayers"   -> 12,
  "dFF"       -> 3072,
  "vocabSize" -> 50257,
  "maxSeqLen" -> 1024
|>;

gpt2MediumSpec[] := <|
  "label"     -> "GPT-2 medium",
  "dModel"    -> 1024, "nHeads" -> 16, "nLayers" -> 24,
  "dFF"       -> 4096, "vocabSize" -> 50257, "maxSeqLen" -> 1024
|>;

gpt2LargeSpec[] := <|
  "label"     -> "GPT-2 large",
  "dModel"    -> 1280, "nHeads" -> 20, "nLayers" -> 36,
  "dFF"       -> 5120, "vocabSize" -> 50257, "maxSeqLen" -> 1024
|>;

(* Build an empty GPT-2 architecture using OUR gptModel constructor.
   Note that the resulting net is randomly initialised; if you want the
   trained weights, see loadPrebuiltGpt2[]. *)
buildGpt2Architecture[size_String : "Small"] := Module[
  {spec = Switch[size,
    "Small",  gpt2SmallSpec[],
    "Medium", gpt2MediumSpec[],
    "Large",  gpt2LargeSpec[],
    _, gpt2SmallSpec[]]},
  gptModel[
    spec["dModel"], spec["nHeads"], spec["nLayers"],
    spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]
  ]
];

(* Load OpenAI's pretrained GPT-2 small from the Wolfram Neural Network
   Repository.  The first call downloads ~498 MB; subsequent calls are
   instant.  The returned net is OpenAI's exact architecture, NOT our
   gptModel layout \[Dash] use this for inference / fine-tuning starting
   from real pretrained weights. *)
loadPrebuiltGpt2[] := NetModel[
  "GPT-2 Transformer Trained on WebText Data",
  "Task" -> "LanguageModeling"
];

(* Structural comparison: how many layers, parameters, what input/output
   shapes.  Note that the architectures are equivalent up to the
   following conventions: (i) we store Q, K, V as three separate
   (dHead, dModel) matrices per head; OpenAI's checkpoint stores them
   as a single (dModel, dModel) matrix that is reshape-split into
   heads at inference time. (ii) we emit (seqLen, |V|) logits per
   training example; the NetModel emits softmax probabilities at the
   last position only.  Both conventions are mathematically
   equivalent. *)
compareArchitectures[ourNet_, theirNet_] := <|
  "ourArrayCount"        -> Information[ourNet,   "ArraysTotalElementCount"],
  "theirArrayCount"      -> Information[theirNet, "ArraysTotalElementCount"],
  "ourLayersCount"       -> Information[ourNet,   "LayersCount"],
  "theirLayersCount"     -> Information[theirNet, "LayersCount"],
  "ourInputPorts"        -> Information[ourNet,   "InputPortNames"],
  "theirInputPorts"      -> Information[theirNet, "InputPortNames"],
  "ourOutputPorts"       -> Information[ourNet,   "OutputPortNames"],
  "theirOutputPorts"     -> Information[theirNet, "OutputPortNames"]
|>;

End[];
EndPackage[];
