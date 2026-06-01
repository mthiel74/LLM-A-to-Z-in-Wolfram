#!/usr/bin/env wolframscript
(* Overnight char-level training run.

   Trains a small (~3M parameter) transformer at character-level on
   the full Tiny Shakespeare corpus, with TimeGoal = 6.5 hours as
   the wall-time cap.

   Saves every artefact to disk so the user can pick it up the next
   morning: trained net, batch-loss history, sample generations at
   three temperatures, summary JSON. *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["../../part-03-transformer-block/src/Transformer.wl"];
Get["Training.wl"];
Get["CharTokeniser.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

text = Import["../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt", "Text"];
Print["Corpus: ", StringLength[text], " characters."];

vocab = LLMAtoZ`CharTokeniser`charVocabulary[text];
tokId = LLMAtoZ`CharTokeniser`charTokenToIdAssociation[vocab];
idTok = LLMAtoZ`CharTokeniser`charIdToTokenAssociation[vocab];
Print["Vocab size: ", Length[vocab]];

ids = LLMAtoZ`CharTokeniser`encodeChars[text, tokId];
Print["Encoded tokens: ", Length[ids]];

(* Train / eval split *)
nTotal = Length[ids];
split = Round[nTotal * 0.97];
trainIds = ids[[;; split]];
evalIds  = ids[[split + 1 ;;]];
Print["Train chars: ", Length[trainIds], "  eval chars: ", Length[evalIds]];

(* Model spec.  d_model 192, 4 layers, 6 heads, d_ff 768, char
   vocab ~65, context 128.  ~1.8M trainable parameters.  Sized to
   fit comfortably in Mac mini RAM during overnight backprop.
   Activation memory per batch is dominated by the attention scores:
   batch * n_heads * ctxLen^2 * n_layers ~= 50 MB; the 7M smallCPU
   run died from OOM because the previous (256-dim, 6-layer, 256-
   context) configuration pushed total per-batch memory past 1 GB
   plus optimizer state. *)
spec = <|
  "label" -> "char-1.8M",
  "dModel" -> 192, "nHeads" -> 6, "nLayers" -> 4,
  "dFF" -> 768, "vocabSize" -> Length[vocab], "maxSeqLen" -> 128
|>;
predicted = LLMAtoZ`Transformer`parameterCount[spec];
Print["Trainable params: ", predicted];

contextLen = spec["maxSeqLen"];

(* Call NetTrain directly here so we can attach a per-round
   TrainingProgressCheckpointing callback that writes the current
   model to disk after every round.  If the kernel dies (licence
   error, OOM, etc) mid-run we keep whatever rounds it completed. *)

data = LLMAtoZ`Training`prepareSequenceDataset[trainIds, contextLen, 64];
Print["Training windows: ", Length[data]];

(* Build the wrapped model (softmax for proper CE loss) *)
model = LLMAtoZ`Transformer`gptModel[
  spec["dModel"], spec["nHeads"], spec["nLayers"],
  spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]];
trainableModel = NetChain[{model, NetMapOperator[SoftmaxLayer[]]}];

batchSize = 32;
baseLR = 0.001;
maxRounds = 200;
ckptPath = FileNameJoin[{dataDir, "charlevel_shakespeare_trained.wlnet"}];
historyPath = FileNameJoin[{dataDir, "charlevel_training_history.wxf"}];

Print["batchSize=", batchSize, "  baseLR=", baseLR,
      "  maxRounds=", maxRounds, "  TimeGoal=5.5 h"];
Print["Per-round checkpoint at: ", ckptPath];
Print["Training start: ", DateString[]];

SeedRandom[123];
trainStart = AbsoluteTime[];

(* TrainingProgressFunction fires every round; we strip the
   trailing softmax and write the bare logit-emitting net + the
   running loss history. *)
saveCheckpoint[results_Association] := Module[{stripped, hist},
  stripped = NetTake[results["Net"], Length[results["Net"]] - 1];
  Export[ckptPath, stripped];
  hist = <|
    "spec" -> spec,
    "vocab" -> vocab,
    "currentRound" -> results["Round"],
    "currentLoss" -> results["RoundLoss"],
    "roundLossList" -> Lookup[results, "RoundLossList", {}],
    "wallSecondsSoFar" -> AbsoluteTime[] - trainStart
  |>;
  Export[historyPath, hist];
];

trained = NetTrain[
  trainableModel, data, All,
  BatchSize -> batchSize,
  LearningRate -> baseLR,
  MaxTrainingRounds -> maxRounds,
  TimeGoal -> Quantity[5.5, "Hours"],
  LossFunction -> CrossEntropyLossLayer["Index"],
  TargetDevice -> "CPU",
  Method -> "ADAM",
  TrainingProgressReporting -> None,
  TrainingProgressFunction -> {saveCheckpoint, "Interval" -> Quantity[5, "Minutes"]}
];
trainEnd = AbsoluteTime[];

If[trained === $Failed,
  Print["Training failed; aborting."];
  Exit[1]
];

(* Final save (in addition to the in-loop saves) *)
res = <|
  "Net" -> NetTake[trained["TrainedNet"], Length[trained["TrainedNet"]] - 1],
  "LossHistory" -> trained["RoundLossList"],
  "BatchLossHistory" -> trained["BatchLossList"],
  "FinalLoss" -> trained["FinalRoundLoss"]
|>;
elapsed = trainEnd - trainStart;
Print["Training end: ", DateString[]];
Print["Elapsed: ", Round[elapsed / 60.0, 0.1], " min."];

Print["Rounds completed: ", Length[res["LossHistory"]]];
Print["Final round loss: ", res["FinalLoss"]];

(* Final save (overwrites the periodic checkpoints with the best
   final version + full batch-loss history). *)
Export[ckptPath, res["Net"]];
Print["Saved final checkpoint."];

Export[historyPath,
  <|"spec" -> spec, "vocab" -> vocab,
    "lossHistory" -> res["LossHistory"],
    "batchLossList" -> res["BatchLossHistory"],
    "wallSeconds" -> elapsed,
    "rounds" -> Length[res["LossHistory"]]|>];
Print["Saved final history."];

(* Alias for the rest of the script. *)
trained = <|"TrainedNet" -> res["Net"],
            "RoundLossList" -> res["LossHistory"],
            "BatchLossList" -> res["BatchLossHistory"]|>;

(* Loss curve *)
lossFig = ListLinePlot[
  trained["BatchLossList"],
  Frame -> True,
  FrameLabel -> {"batch index", "cross-entropy loss"},
  PlotLabel -> Row[{"Char-level Shakespeare, ",
    Length[trained["RoundLossList"]], " rounds, ",
    Round[elapsed/60., 0.1], " min on CPU"}],
  PlotStyle -> Directive[ColorData[97][1], Thick],
  GridLines -> Automatic,
  ImageSize -> 700];
Export[FileNameJoin[{figDir, "charlevel_loss_curve.png"}], lossFig,
       ImageResolution -> 144];
Print["Saved loss curve."];

(* Sample generations at three temperatures. *)
generate[net_, promptStr_String, n_, T_] := Module[
  {pIds = LLMAtoZ`CharTokeniser`encodeChars[promptStr, tokId],
   gen, ctx, pos, logits, p, sampled},
  gen = pIds;
  Do[
    ctx = If[Length[gen] >= contextLen, gen[[-contextLen ;;]],
              PadRight[gen, contextLen, 1]];
    pos = Min[Length[gen], contextLen];
    logits = net[ctx][[pos]];
    p = Exp[(logits - Max[logits])/T]; p = p/Total[p];
    sampled = RandomChoice[p -> Range[Length[p]]];
    AppendTo[gen, sampled],
    {n}];
  LLMAtoZ`CharTokeniser`decodeChars[gen, idTok]
];

prompts = {"ROMEO:", "HAMLET:\n", "First Citizen:\n",
           "To be, or not to be"};
SeedRandom[1];
samples = <||>;
Do[
  Module[{key, gen},
    Do[
      gen = generate[trained["TrainedNet"], p, 400, T];
      key = "T=" <> ToString[T] <> "  prompt=" <> p;
      samples[key] = gen;
      Print["---  ", key, "  ---"];
      Print[gen];
      Print[""],
      {T, {0.5, 0.8, 1.0}}
    ]
  ],
  {p, prompts}
];

(* Summary JSON *)
summary = <|
  "tier" -> "charlevel-overnight",
  "spec" -> spec,
  "trainableParams" -> predicted,
  "vocabSize" -> Length[vocab],
  "elapsedSeconds" -> elapsed,
  "elapsedMinutes" -> elapsed/60.0,
  "rounds" -> Length[trained["RoundLossList"]],
  "finalRoundLoss" -> trained["FinalRoundLoss"],
  "samples" -> Normal[samples]
|>;
Export[FileNameJoin[{dataDir, "charlevel_summary.json"}], summary];
Print["Wrote charlevel_summary.json"];

Print["DONE."];
