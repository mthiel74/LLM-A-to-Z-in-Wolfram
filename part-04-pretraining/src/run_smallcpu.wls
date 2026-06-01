#!/usr/bin/env wolframscript
(* smallCPU tier training run: ~5M parameter model on Tiny
   Shakespeare for up to 10 rounds, capped at 28 minutes wall time.
   Saves checkpoint, loss history, sample generations. *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["../../part-03-transformer-block/src/Transformer.wl"];
Get["Training.wl"];
Get["../../Tiers.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

(* Load BPE and corpus *)
Print["Loading BPE and corpus..."];
bpe = Import["../../part-01-tokeniser-and-embeddings/data/bpe_trained.mx"];
mergeRules = bpe["MergeRules"];
vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[bpe];
tokId = AssociationThread[vocab -> Range[Length[vocab]]];
idTok = AssociationThread[Range[Length[vocab]] -> vocab];
Print["Vocab: ", Length[vocab]];

text = Import["../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt", "Text"];
ids = Lookup[tokId,
  Flatten[Lookup[bpe["FinalEncoding"], #,
    LLMAtoZ`Tokeniser`applyMerges[#, mergeRules]] & /@
    LLMAtoZ`Tokeniser`preTokenise[text]]];
Print["Total tokens: ", Length[ids]];

(* Use smallCPU tier; override vocab size to match our cached BPE *)
tierName = "smallCPU";
baseSpec = LLMAtoZ`Tiers`tier[tierName];
spec = baseSpec["modelSpec"];
spec["vocabSize"] = Length[vocab];

budget = LLMAtoZ`Tiers`trainingBudget[tierName];

Print["Tier: ", tierName];
Print["Spec: dModel=", spec["dModel"], " nHeads=", spec["nHeads"],
      " nLayers=", spec["nLayers"], " dFF=", spec["dFF"],
      " contextLen=", spec["maxSeqLen"]];
Print["Trainable parameters: ",
  LLMAtoZ`Transformer`parameterCount[spec]];
Print["Total array elements (with masks): ",
  LLMAtoZ`Transformer`parameterCountFull[spec]];

(* Build the model *)
model = LLMAtoZ`Transformer`gptModel[
  spec["dModel"], spec["nHeads"], spec["nLayers"],
  spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]];

(* Prepare training dataset.  Use 95% for training. *)
nTotal = Length[ids];
split = Round[nTotal * 0.95];
trainIds = ids[[;; split]];
evalIds  = ids[[split + 1 ;;]];
Print["Train tokens: ", Length[trainIds], "  eval tokens: ", Length[evalIds]];

contextLen = spec["maxSeqLen"];
data = LLMAtoZ`Training`prepareSequenceDataset[trainIds, contextLen];
Print["Training windows: ", Length[data]];

(* Train with TimeGoal as a hard cap.  Track wall time. *)
batchSize = budget["BatchSize"];
baseLR = budget["BaseLR"];
maxRounds = budget["MaxRounds"];
Print["batchSize=", batchSize, "  baseLR=", baseLR, "  maxRounds=", maxRounds];

SeedRandom[42];
trainStart = AbsoluteTime[];
Print["Starting training at ", DateString[]];

trained = NetTrain[
  model, data, All,
  BatchSize -> batchSize,
  LearningRate -> baseLR,
  MaxTrainingRounds -> maxRounds,
  TimeGoal -> Quantity[28, "Minutes"],
  LossFunction -> CrossEntropyLossLayer["Index"],
  TargetDevice -> "CPU",
  Method -> "ADAM",
  TrainingProgressReporting -> None
];

trainEnd = AbsoluteTime[];
elapsed = trainEnd - trainStart;
Print["Training finished at ", DateString[], "; elapsed ",
      Round[elapsed / 60.0, 0.1], " min."];

If[trained === $Failed,
  Print["NetTrain returned $Failed; aborting."];
  Exit[1]
];

Print["Final round loss: ", trained["FinalRoundLoss"]];
Print["Rounds completed: ", Length[trained["RoundLossList"]]];

(* Save checkpoint *)
ckptPath = FileNameJoin[{dataDir, "smallcpu_gpt_trained.wlnet"}];
Export[ckptPath, trained["TrainedNet"]];
Print["Saved checkpoint to ", ckptPath];

(* Save loss history *)
historyPath = FileNameJoin[{dataDir, "smallcpu_training_history.wxf"}];
Export[historyPath, <|
  "tier" -> tierName,
  "spec" -> spec,
  "lossHistory" -> trained["RoundLossList"],
  "batchLossList" -> trained["BatchLossList"],
  "wallSeconds" -> elapsed,
  "rounds" -> Length[trained["RoundLossList"]]
|>];

(* Loss curve figure *)
lossFig = ListLinePlot[
  trained["BatchLossList"],
  Frame -> True,
  FrameLabel -> {"batch index", "cross-entropy loss"},
  PlotLabel -> Row[{"smallCPU 5M model, ",
    Length[trained["RoundLossList"]], " rounds, ",
    Round[elapsed/60., 0.1], " min on CPU"}],
  PlotStyle -> Directive[ColorData[97][1], Thick],
  GridLines -> Automatic,
  ImageSize -> 600];
Export[FileNameJoin[{figDir, "smallcpu_loss_curve.png"}], lossFig,
       ImageResolution -> 144];
Print["Saved smallcpu_loss_curve.png"];

(* Generations *)
generate[net_, prompt_String, n_, T_] := Module[
  {pIds, gen, ctx, pos, logits, p, sampled},
  pIds = Lookup[tokId,
    Flatten[Lookup[bpe["FinalEncoding"], #,
      LLMAtoZ`Tokeniser`applyMerges[#, mergeRules]] & /@
      LLMAtoZ`Tokeniser`preTokenise[prompt]]];
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
  StringReplace[StringJoin @ Lookup[idTok, gen], "</w>" -> " "]
];

prompts = {"ROMEO:", "HAMLET:", "First Citizen:", "To be or not"};
samples = AssociationMap[generate[trained["TrainedNet"], #, 80, 0.8] &, prompts];
KeyValueMap[Print["---\n", #1, " --> ", #2] &, samples];

(* Save summary JSON *)
summary = <|
  "tier" -> tierName,
  "trainableParams" -> LLMAtoZ`Transformer`parameterCount[spec],
  "elapsedSeconds" -> elapsed,
  "elapsedMinutes" -> elapsed / 60.0,
  "rounds" -> Length[trained["RoundLossList"]],
  "finalRoundLoss" -> trained["FinalRoundLoss"],
  "samples" -> Normal[samples]
|>;
Export[FileNameJoin[{dataDir, "smallcpu_summary.json"}], summary];
Print["Wrote smallcpu_summary.json"];

Print["DONE."];
