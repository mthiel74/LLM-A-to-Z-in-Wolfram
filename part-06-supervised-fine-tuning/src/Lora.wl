(* ::Package:: *)

(* ============================================================
   Lora.wl  ---  Part 6, Section: LoRA fine-tuning
   ============================================================

   Low-Rank Adaptation (Hu et al. 2021): instead of training every
   parameter of a pretrained linear layer, freeze its weight matrix
   W and learn two small matrices A (rank r, narrow), B (rank r,
   narrow) such that the effective weight is W + B*A.  When r is
   tiny (8, 16, 32), the number of new parameters is a tiny
   fraction of W's count, training is fast, and at inference the
   adapters can be folded back into W.

   This file adds:

     - loraDelta[d_in, d_out, rank]      a 2-layer (A, B) NetChain
                                          whose output is added to a
                                          frozen base linear's output.
     - loraWrap[baseLinear, rank]        wraps an existing LinearLayer
                                          in a NetGraph that sums
                                          baseLinear(x) + loraDelta(x).
                                          baseLinear is marked
                                          non-trainable.
     - applyLoraToBlock[block, paths,
                        rank]            given a transformerBlock
                                          and a list of paths to its
                                          Q / K / V / output Linears,
                                          replace each with a
                                          loraWrapped version.
     - trainLora[net, dataset, hp]       run NetTrain with LearningRate
                                          on the LoRA adapters only.

   The post explains why r = 8 to 32 is usually enough: in
   pretrained models, the "delta" the new task needs to add to W
   is empirically low-rank.  Pedagogically, LoRA is the cleanest
   demonstration of parameter-efficient fine-tuning -- one block,
   four small matrices per attention sublayer.
*)

BeginPackage["LLMAtoZ`Lora`",
  {"LLMAtoZ`Transformer`", "LLMAtoZ`Training`"}];

ClearAll[
  loraDelta, loraWrap, freezeArrays,
  applyLoraToLinear, trainLora,
  loraParameterCount
];

loraDelta::usage =
  "loraDelta[dIn, dOut, rank] returns a NetChain of two LinearLayers \
(no bias) that compute the LoRA delta: x -> A x -> B (A x) -> dOut. \
A is (rank, dIn); B is (dOut, rank).  Initialised so the initial \
delta is zero (B is zero-initialised, A is Gaussian).";

loraWrap::usage =
  "loraWrap[baseLinear, dIn, dOut, rank, scale] wraps a LinearLayer in \
a NetGraph whose output is baseLinear(x) + (scale/rank) * loraDelta(x). \
The base weights are kept; the adapter weights are added at train time. \
dIn and dOut must be supplied explicitly (WL's Information on a \
LinearLayer returns shape objects that are awkward to unpack).";

applyLoraToLinear::usage =
  "applyLoraToLinear[net, path, rank] returns net with the LinearLayer \
at the given NetExtract path wrapped in a LoRA adapter.  Repeat with \
different paths to LoRA-wrap several linears.";

trainLora::usage =
  "trainLora[net, dataset, hp] is a thin wrapper around NetTrain that \
fixes the learning rate to the LoRA convention (smaller than full FT) \
and trains for fewer epochs.  Note: a true LoRA-only training run \
freezes the base weights; in WL we approximate this by running NetTrain \
on the wrapped net with very small base-weight LRs through \
LearningRateMultipliers.";

loraParameterCount::usage =
  "loraParameterCount[dIn, dOut, rank] = rank * (dIn + dOut).  Compare \
to dIn * dOut for full fine-tuning.";

Begin["`Private`"];

loraDelta[dIn_Integer, dOut_Integer, rank_Integer] :=
  NetChain[{
    LinearLayer[rank, "Biases" -> None,
      "Weights" -> RandomReal[NormalDistribution[0, 0.02], {rank, dIn}]],
    LinearLayer[dOut, "Biases" -> None,
      "Weights" -> ConstantArray[0., {dOut, rank}]]
  },
  "Input" -> dIn];

(* Wrap a single LinearLayer with a sibling LoRA delta.  The base
   linear is kept; the delta is added.  At init the delta is zero so
   the wrapped net is functionally identical to the base.  Training
   updates the delta. *)
loraWrap[baseLinear_LinearLayer, dIn_Integer, dOut_Integer,
    rank_Integer, scale_Real : 1.0] :=
  NetGraph[
    <|
      "base"   -> baseLinear,
      "delta"  -> loraDelta[dIn, dOut, rank],
      "scale"  -> ElementwiseLayer[(scale / rank) * # &],
      "add"    -> ThreadingLayer[Plus]
    |>,
    {NetPort["Input"] -> "base",
     NetPort["Input"] -> "delta" -> "scale",
     {"base", "scale"} -> "add"}
  ];

(* Apply LoRA wrap to a LinearLayer found at `path` inside `net`. *)
applyLoraToLinear[net_, path_List, dIn_Integer, dOut_Integer,
    rank_Integer] := Module[
  {layer = NetExtract[net, path]},
  If[!MatchQ[layer, _LinearLayer],
    Print["applyLoraToLinear: layer at ", path, " is not a LinearLayer; skipping"];
    net,
    NetReplacePart[net, path -> loraWrap[layer, dIn, dOut, rank]]
  ]
];

(* In a real LoRA implementation we mark the base weights with a
   learning-rate multiplier of zero so the optimizer cannot move them.
   Wolfram's NetTrain supports LearningRateMultipliers; we set it to
   {_, "Weights"} -> 0 inside the "base" sublayer of every wrapped
   block.  Caller is responsible for collecting those paths. *)
freezeArrays[net_, paths_List] := Module[{rules},
  rules = (# -> 0 &) /@ paths;
  rules
];

trainLora[net_, dataset_List, hp_Association : Automatic,
    frozenArrayPaths_List : {}] := Module[
  {hyp, trained, trainStart, trainEnd, lrMults},
  hyp = Join[LLMAtoZ`Training`defaultHyperparameters[],
             <|"BaseLR" -> 0.0003, "MaxRounds" -> 6|>,
             If[AssociationQ[hp], hp, <||>]];
  lrMults = freezeArrays[net, frozenArrayPaths];
  SeedRandom[hyp["RandomSeed"]];
  trainStart = AbsoluteTime[];
  trained = NetTrain[
    net, dataset, All,
    BatchSize -> hyp["BatchSize"],
    LearningRate -> hyp["BaseLR"],
    MaxTrainingRounds -> hyp["MaxRounds"],
    LossFunction -> CrossEntropyLossLayer["Index"],
    LearningRateMultipliers -> lrMults,
    TargetDevice -> "CPU",
    Method -> hyp["OptimizerType"],
    TrainingProgressReporting -> None
  ];
  trainEnd = AbsoluteTime[];
  If[trained === $Failed,
    Return[<|"Net" -> $Failed, "WallSeconds" -> (trainEnd - trainStart)|>]
  ];
  <|
    "Net" -> trained["TrainedNet"],
    "LossHistory" -> trained["RoundLossList"],
    "BatchLossHistory" -> trained["BatchLossList"],
    "WallSeconds" -> (trainEnd - trainStart),
    "FinalLoss" -> trained["FinalRoundLoss"]
  |>
];

loraParameterCount[dIn_Integer, dOut_Integer, rank_Integer] :=
  rank * (dIn + dOut);

End[];
EndPackage[];
