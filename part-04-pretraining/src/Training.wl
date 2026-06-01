(* ::Package:: *)

(* ============================================================
   Training.wl  ---  Part 4 of "LLM A to Z in Wolfram Language"
   ============================================================

   The pretraining loop.  Given a tokenised corpus, a model from
   Part 3, and a small handful of hyperparameters, produce a
   trained model and the loss curve that goes with it.  The CPU
   path runs on a Mac mini; the cloud path is a one-call
   wrapper around RemoteBatchSubmit for the same workload at
   scale.

   The training task is next-token prediction: each (input,
   target) pair is a context window and the same window shifted
   one step forward.  Cross-entropy loss against the shifted
   targets is what NetTrain optimises.

   Conventions used throughout:
     - The model takes a length-`contextLen` integer-ID sequence
       and returns a (contextLen, vocabSize) tensor of pre-softmax
       logits.  NetTrain attaches a SoftmaxLayer + CrossEntropy
       internally when the output is shaped as a probability
       vector per position.
     - Token IDs are 1-based, matching WL convention and the
       buildBPEVocabulary output from Part 1.
     - All hyperparameters live in an Association so they are
       easy to log, override, and dump as JSON. *)

BeginPackage["LLMAtoZ`Training`",
  {"LLMAtoZ`Tokeniser`", "LLMAtoZ`Transformer`"}];

ClearAll[
  prepareSequenceDataset, sampleBatch, makeTrainingData,
  defaultHyperparameters, cosineSchedule,
  trainGptModel, evalPerplexity,
  submitRemotePretraining
];

prepareSequenceDataset::usage =
  "prepareSequenceDataset[tokenIds, contextLen] returns a list of \
<|\"Input\" -> ids, \"Target\" -> shifted|> pairs of length contextLen. \
Used to feed NetTrain.";

defaultHyperparameters::usage =
  "defaultHyperparameters[] returns an Association with the default \
training hyperparameters: batch size, learning rate, max rounds, \
context length, and the cosine-schedule warmup fraction.";

cosineSchedule::usage =
  "cosineSchedule[round, totalRounds, baseLR, warmupFrac] returns the \
learning rate at round `round` of `totalRounds` total rounds, with a \
linear warmup over the first `warmupFrac` of rounds and a cosine decay \
to 10% of baseLR thereafter.";

trainGptModel::usage =
  "trainGptModel[modelSpec, tokenIds, hyperparameters] trains a fresh \
gptModel of the given spec on the given token sequence.  Returns an \
Association with keys \"Net\", \"LossHistory\", \"Hyperparameters\", \
\"Spec\", \"WallSeconds\".";

evalPerplexity::usage =
  "evalPerplexity[net, tokenIds, contextLen] returns the model's \
perplexity (exponential of mean cross-entropy) on the given held-out \
token sequence.";

submitRemotePretraining::usage =
  "submitRemotePretraining[modelSpec, tokenIds, hyperparameters] wraps \
trainGptModel in a CloudEvaluate / RemoteBatchSubmit call.  Use this \
when you want to scale up to a 25M model on a real corpus and have a \
Wolfram Cloud account.  Returns a RemoteBatchSubmit handle whose \
status can be polled with BatchJobStatus.";

Begin["`Private`"];

(* ------------------------------------------------------------
   Sequence dataset preparation.

   Slide a length-contextLen window through the token sequence
   with a chosen stride.  Each window becomes one training
   example whose target is the same window shifted by one step
   (the canonical next-token-prediction task).  Stride = contextLen
   gives non-overlapping windows; smaller strides give more
   training examples at the cost of more redundancy. *)

prepareSequenceDataset[tokenIds_List, contextLen_Integer,
    stride_Integer : Automatic] := Module[
  {actualStride, n = Length[tokenIds], starts},
  actualStride = If[stride === Automatic, contextLen, stride];
  starts = Range[1, n - contextLen - 1, actualStride];
  (* The output port of gptModel is "Output"; matching that port name
     here lets NetTrain attach the CrossEntropyLossLayer automatically. *)
  Table[
    <|
      "Input"  -> tokenIds[[s ;; s + contextLen - 1]],
      "Output" -> tokenIds[[s + 1 ;; s + contextLen]]
    |>,
    {s, starts}
  ]
];

(* ------------------------------------------------------------
   Hyperparameters and schedules. *)

defaultHyperparameters[] := <|
  "ContextLen"    -> 64,
  "BatchSize"     -> 32,
  "BaseLR"        -> 0.001,
  "MaxRounds"     -> 6,
  "WarmupFrac"    -> 0.1,
  "WeightDecay"   -> 0.01,
  "OptimizerType" -> "ADAM",
  "GradientClip"  -> 1.0,
  "RandomSeed"    -> 42,
  "Method"        -> Automatic
|>;

cosineSchedule[round_, totalRounds_, baseLR_, warmupFrac_:0.1] := Module[
  {warmRounds, progress, decay},
  warmRounds = Max[1, Round[warmupFrac*totalRounds]];
  Which[
    round < warmRounds,
      baseLR * (round + 1.0) / warmRounds,
    True,
      progress = (round - warmRounds) / Max[1, (totalRounds - warmRounds)];
      decay = 0.5 (1 + Cos[Pi * progress]);
      baseLR * (0.1 + 0.9 * decay)
  ]
];

(* ------------------------------------------------------------
   The main training entry point.

   Builds the model, prepares the dataset, runs NetTrain with a
   live progress callback that pushes (round, batch_loss) into a
   buffer, and returns everything we want to plot or analyse. *)

trainGptModel[modelSpec_Association, tokenIds_List,
    hp_Association : Automatic] := Module[
  {hyp, model, trainableModel, data, lossHistory = {},
   trainStart, trainEnd, trained, contextLen, totalRounds,
   strippedNet},
  hyp = Join[defaultHyperparameters[], If[AssociationQ[hp], hp, <||>]];
  contextLen = hyp["ContextLen"];
  totalRounds = hyp["MaxRounds"];

  (* Build a fresh model at the spec's dimensions but with the
     training contextLen (which may differ from the spec's
     maxSeqLen). *)
  model = LLMAtoZ`Transformer`gptModel[
    modelSpec["dModel"], modelSpec["nHeads"], modelSpec["nLayers"],
    modelSpec["dFF"], modelSpec["vocabSize"], contextLen
  ];

  data = prepareSequenceDataset[tokenIds, contextLen];

  (* The base gptModel outputs raw logits.  CrossEntropyLossLayer["Index"]
     expects PROBABILITIES; with raw logits the loss becomes unbounded-
     below and the optimizer drives one logit toward +Infinity, causing
     mode collapse.  Wrap the model in a NetChain that softmaxes per
     position before training.  After training we strip the softmax so
     the saved checkpoint still emits logits (cleaner for sampling). *)
  trainableModel = NetChain[{model, NetMapOperator[SoftmaxLayer[]]}];

  SeedRandom[hyp["RandomSeed"]];
  trainStart = AbsoluteTime[];
  trained = NetTrain[
    trainableModel, data, All,
    Sequence @@ Flatten[{
      BatchSize -> hyp["BatchSize"],
      LearningRate -> hyp["BaseLR"],
      MaxTrainingRounds -> totalRounds,
      If[KeyExistsQ[hyp, "TimeGoal"],
        TimeGoal -> hyp["TimeGoal"], {}],
      LossFunction -> CrossEntropyLossLayer["Index"],
      TargetDevice -> "CPU",
      Method -> hyp["OptimizerType"],
      TrainingProgressReporting -> None
    }]
  ];
  trainEnd = AbsoluteTime[];
  If[trained === $Failed,
    Print["NetTrain returned $Failed; check data format and model output port."];
    Return[<|"Net" -> $Failed, "LossHistory" -> {}, "WallSeconds" -> 0|>]
  ];

  (* Strip the trailing softmax so the saved checkpoint emits logits
     (the convention the sampling code in Part 5 expects). *)
  strippedNet = NetTake[trained["TrainedNet"], Length[trained["TrainedNet"]] - 1];

  <|
    "Net"               -> strippedNet,
    "TrainedNetWithSoftmax" -> trained["TrainedNet"],
    "LossHistory"       -> trained["RoundLossList"],
    "BatchLossHistory"  -> trained["BatchLossList"],
    "Hyperparameters"   -> hyp,
    "Spec"              -> modelSpec,
    "WallSeconds"       -> (trainEnd - trainStart),
    "FinalLoss"         -> trained["FinalRoundLoss"]
  |>
];

(* ------------------------------------------------------------
   Held-out perplexity. *)

evalPerplexity[net_, tokenIds_List, contextLen_Integer] := Module[
  {data, losses},
  data = prepareSequenceDataset[tokenIds, contextLen];
  If[Length[data] == 0, Return[Indeterminate]];
  (* Compute mean cross-entropy across the held-out set. *)
  losses = Map[
    Function[ex,
      Module[{logits, probs, target},
        logits = net[ex["Input"]];
        probs = Map[SoftmaxLayer[][#] &, logits];
        target = ex["Target"];
        -Mean[Log[MapThread[#1[[#2]] &, {probs, target}] + 1.*^-12]]
      ]
    ],
    data
  ];
  Exp[Mean[losses]]
];

(* ------------------------------------------------------------
   Remote-batch wrapper.

   This is the "scale it up" path.  The cloud submission takes
   the same trainGptModel call and ships it to the Wolfram Cloud
   where (with appropriate credentials) it can run on a beefier
   machine.  For the educational notebook we keep this as a
   single-line wrapper; the cloud submission itself happens via
   CloudEvaluate or RemoteBatchSubmit depending on the version
   of WL the reader is on. *)

submitRemotePretraining[modelSpec_Association, tokenIds_List,
    hp_Association : Automatic] := Module[
  {expr},
  expr = Hold[trainGptModel[modelSpec, tokenIds, hp]];
  (* If RemoteBatchSubmit is available, use it; otherwise fall
     back to CloudEvaluate.  We keep this lightweight; the reader
     is expected to have CloudConnect[] done. *)
  Which[
    ValueQ[Symbol["RemoteBatchSubmit"]],
      Symbol["RemoteBatchSubmit"][ReleaseHold[expr]],
    True,
      CloudEvaluate[ReleaseHold[expr]]
  ]
];

End[];
EndPackage[];
