#!/usr/bin/env wolframscript
(* End-to-end Part 4 pre-training pipeline.

   1. Load cached BPE merges from Part 1; encode Tiny Shakespeare.
   2. Build a small 1M-parameter GPT model.
   3. Train it for a small budget on CPU (capped to keep total run
      under ~15 minutes on a Mac mini).
   4. Save the trained net, the loss curve, the eval perplexity,
      and sample generations.
   5. Produce figures: the loss curve, the learning-rate schedule,
      and a before/after generation comparison.

   For full-scale runs the same `trainGptModel` call is wrapped by
   `submitRemotePretraining` and dispatched to the Wolfram Cloud --
   see section 7 of the notebook.
*)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["../../part-03-transformer-block/src/Transformer.wl"];
Get["Training.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

(* ---------- Load corpus & tokeniser ---------- *)

trained = Import["../../part-01-tokeniser-and-embeddings/data/bpe_trained.mx"];
mergeRules = trained["MergeRules"];
vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[trained];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[vocab];
idTok = LLMAtoZ`Tokeniser`idToTokenAssociation[vocab];
Print["Vocabulary: ", Length[vocab]];

text = Import["../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt", "Text"];
Print["Encoding the corpus..."];
{encodeTime, tokenStream} = AbsoluteTiming[
  (* Use the cached final encoding for known pre-tokens *)
  Flatten[
    Lookup[trained["FinalEncoding"], #,
      LLMAtoZ`Tokeniser`applyMerges[#, mergeRules]] & /@
    LLMAtoZ`Tokeniser`preTokenise[text]
  ]
];
ids = Lookup[tokId, tokenStream];
Print["Encoded ", Length[ids], " tokens in ", Round[encodeTime, 0.1], " s"];

(* Train/eval split: last 5% for eval *)
nTotal = Length[ids];
split  = Round[nTotal * 0.95];
trainIds = ids[[;; split]];
evalIds  = ids[[split + 1 ;;]];
Print["Train tokens: ", Length[trainIds], "; eval tokens: ", Length[evalIds]];

(* ---------- Configure hyperparameters ---------- *)

(* Use a smaller-than-1M model so the training run completes in a few
   minutes on a Mac mini.  The post explains how to scale up. *)
modelSpec = <|
  "label" -> "tiny",
  "dModel" -> 96,
  "nHeads" -> 4,
  "nLayers" -> 3,
  "dFF" -> 256,
  "vocabSize" -> Length[vocab],
  "maxSeqLen" -> 64
|>;
predicted = LLMAtoZ`Transformer`parameterCount[modelSpec];
Print["Spec ", modelSpec["label"], " -> trainable params: ", predicted];

hp = Join[LLMAtoZ`Training`defaultHyperparameters[], <|
  "ContextLen" -> 64,
  "BatchSize"  -> 32,
  "BaseLR"     -> 0.002,
  "MaxRounds"  -> 3
|>];

(* ---------- Train ---------- *)

Print["Starting training..."];
result = LLMAtoZ`Training`trainGptModel[modelSpec, trainIds, hp];
Print["Final round loss: ", N @ result["FinalLoss"]];
Print["Wall time: ", Round[result["WallSeconds"], 0.1], " s"];

(* Cache the trained checkpoint and metadata. *)
Export[FileNameJoin[{dataDir, "tiny_gpt_trained.wlnet"}], result["Net"]];
Export[FileNameJoin[{dataDir, "training_history.wxf"}], <|
  "lossHistory"     -> result["LossHistory"],
  "batchLossList"   -> result["BatchLossHistory"],
  "spec"            -> modelSpec,
  "hyperparameters" -> Normal[result["Hyperparameters"]],
  "wallSeconds"     -> result["WallSeconds"]
|>];

(* ---------- Figures ---------- *)

(* Loss curve *)
lossFig = ListLinePlot[
  result["BatchLossHistory"],
  Frame -> True,
  FrameLabel -> {"batch index", "cross-entropy loss"},
  PlotLabel -> Row[{"Training loss (tiny ", predicted,
                    "-param model, ", hp["MaxRounds"], " rounds)"}],
  PlotStyle -> Directive[ColorData[97][1], Thick],
  GridLines -> Automatic,
  ImageSize -> 540
];
Export[FileNameJoin[{figDir, "loss_curve.png"}], lossFig,
       ImageResolution -> 144];
Print["Saved loss_curve.png"];

(* LR schedule *)
totalRounds = hp["MaxRounds"];
lrs = Table[
  LLMAtoZ`Training`cosineSchedule[r, 100, hp["BaseLR"], hp["WarmupFrac"]],
  {r, 0, 99}];
lrFig = ListLinePlot[lrs,
  Frame -> True,
  FrameLabel -> {"step (% of total)", "learning rate"},
  PlotLabel -> "Linear warmup + cosine decay",
  PlotStyle -> Directive[ColorData[97][2], Thick],
  ImageSize -> 540, GridLines -> Automatic];
Export[FileNameJoin[{figDir, "lr_schedule.png"}], lrFig,
       ImageResolution -> 144];
Print["Saved lr_schedule.png"];

(* ---------- Sample generations before/after ---------- *)

generateSample[net_, prompt_String, nNew_Integer, temperature_:1.0] := Module[
  {promptIds, generated, logits, probs, sampled, ctx, pos},
  promptIds = Lookup[tokId,
    Flatten[Lookup[trained["FinalEncoding"], #,
              LLMAtoZ`Tokeniser`applyMerges[#, mergeRules]] & /@
            LLMAtoZ`Tokeniser`preTokenise[prompt]]];
  generated = promptIds;
  Do[
    ctx = If[Length[generated] >= 64,
              generated[[-64 ;;]],
              PadRight[generated, 64, 1]];
    pos = Min[Length[generated], 64];
    logits = net[ctx][[pos, All]];
    probs = Exp[logits / temperature - Max[logits / temperature]];
    probs = probs / Total[probs];
    sampled = RandomChoice[probs -> Range[Length[probs]]];
    AppendTo[generated, sampled],
    {nNew}
  ];
  StringRiffle[Lookup[idTok, generated], ""]
];

(* Untrained baseline (a fresh random model) *)
untrained = LLMAtoZ`Transformer`gptModel[
  modelSpec["dModel"], modelSpec["nHeads"], modelSpec["nLayers"],
  modelSpec["dFF"], modelSpec["vocabSize"], 64];

prompt = "ROMEO:";
before = generateSample[untrained, prompt, 40, 1.0];
after  = generateSample[result["Net"], prompt, 40, 1.0];
Print["Before training: ", before];
Print["After training:  ", after];

(* ---------- Summary JSON ---------- *)
summary = <|
  "vocabSize" -> Length[vocab],
  "trainTokens" -> Length[trainIds],
  "evalTokens"  -> Length[evalIds],
  "spec" -> modelSpec,
  "trainableParams" -> predicted,
  "finalLoss" -> result["FinalLoss"],
  "wallSeconds" -> result["WallSeconds"],
  "samplePrompt" -> prompt,
  "sampleBefore" -> before,
  "sampleAfter"  -> after
|>;
Export[FileNameJoin[{dataDir, "part04_summary.json"}], summary];
Print["Wrote part04_summary.json"];

Print["DONE."];
