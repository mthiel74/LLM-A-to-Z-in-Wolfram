#!/usr/bin/env wolframscript
(* Path A: SFT pretrained GPT-2 small on the toy instruction dataset.

   1.  Load GPT-2 small (14.3-compatible cache).
   2.  Strip the SequenceLast + Softmax tail so the model emits per-
       position logits for training.
   3.  Format the 20 Shakespeare Q&A pairs as plain text with
       "User: ...\nAssistant: ..." markers, encode with GPT-2's
       BPE.
   4.  Train for a few rounds at a small LR (5e-5) to instruction-
       tune without catastrophic forgetting.
   5.  Save the SFT'd model and a few generations.
*)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Quiet @ Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Quiet @ Get["../../part-02-self-attention/src/Attention.wl"];
Quiet @ Get["../../part-03-transformer-block/src/Transformer.wl"];
Quiet @ Get["SFT.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

Print["Loading GPT-2 small ..."];
gpt = Import["../../part-02-self-attention/data/gpt2_v143.wlnet"];
Print["NetChain length: ", Length[gpt]];

(* Surgery: drop SequenceLast (layer 3), the LinearLayer (layer 4),
   and SoftmaxLayer (layer 5).  Replace with a position-wise
   LinearLayer using the SAME weights as the original output linear.
   This gives a model that emits (seqLen, |V|) logits per position --
   the standard training-time output shape. *)

embedding = gpt[[1]];
stack     = gpt[[2]];
outLinear = gpt[[4]];
encoder   = NetExtract[gpt, "Input"];

trainableTrunk = NetChain[{embedding, stack,
  NetMapOperator[outLinear]}];
Print["Trunk built.  ArraysTotalElementCount: ",
  Information[trainableTrunk, "ArraysTotalElementCount"]];

(* Smoke test: feed text in via the encoder, check we get logits *)
sampleIds = encoder["Hello world."];
Print["Sample ids: ", sampleIds];
sampleOut = trainableTrunk[sampleIds];
Print["Sample output shape: ", Dimensions[sampleOut]];

(* Build SFT dataset *)
qaPairs = LLMAtoZ`SFT`toyInstructionDataset[];
Print["Q&A pairs: ", Length[qaPairs]];

formatPair[{u_, a_}] :=
  "User: " <> u <> "\nAssistant: " <> a <> "\n";

formatted = formatPair /@ qaPairs;
Print["Sample format: ", formatted[[1]]];

(* Encode + truncate / pad to a fixed length so we can batch.  GPT-2
   small was trained with max_seq_len = 1024; here we use 128 to keep
   each batch cheap on CPU. *)
maxLen = 128;
padId = 1;  (* GPT-2's <|endoftext|> token in WL is index 50257; pad
              with 1 is safer for the cross-entropy loss target. *)

encodePadded[text_] := Module[{ids = encoder[text]},
  If[Length[ids] >= maxLen + 1,
    ids = ids[[;; maxLen + 1]],
    ids = PadRight[ids, maxLen + 1, padId]];
  ids
];

trainData = Map[
  Function[txt,
    Module[{ids = encodePadded[txt]},
      <|"Input" -> ids[[;; -2]], "Output" -> ids[[2 ;;]]|>]],
  formatted];

Print["Built ", Length[trainData], " training examples of length ", maxLen];

(* Train.  Wrap with softmax so cross-entropy is proper. *)
trainable = NetChain[{trainableTrunk, NetMapOperator[SoftmaxLayer[]]}];

SeedRandom[7];
Print["Starting SFT at ", DateString[]];
trainStart = AbsoluteTime[];
trained = NetTrain[
  trainable, trainData, All,
  BatchSize -> 4,
  LearningRate -> 5*10^-5,
  MaxTrainingRounds -> 5,
  TimeGoal -> Quantity[20, "Minutes"],
  LossFunction -> CrossEntropyLossLayer["Index"],
  TargetDevice -> "CPU",
  Method -> "ADAM",
  TrainingProgressReporting -> None
];
trainEnd = AbsoluteTime[];
elapsed = trainEnd - trainStart;
Print["SFT end at ", DateString[], ", elapsed ", Round[elapsed/60., 0.1], " min."];
Print["Round losses: ", N /@ trained["RoundLossList"]];

(* Strip the softmax so the saved checkpoint emits logits *)
sftLogitModel = NetTake[trained["TrainedNet"], Length[trained["TrainedNet"]] - 1];
Export[FileNameJoin[{dataDir, "gpt2_sft_chat.wlnet"}], sftLogitModel];
Print["Saved gpt2_sft_chat.wlnet."];

(* Quick generations *)
generate[net_, prompt_String, n_, T_] := Module[
  {ids, gen, ctx, pos, logits, p, sampled, idTok},
  ids = encoder[prompt];
  gen = ids;
  Do[
    ctx = If[Length[gen] >= maxLen, gen[[-maxLen ;;]],
              PadRight[gen, maxLen, padId]];
    pos = Min[Length[gen], maxLen];
    logits = net[ctx][[pos]];
    p = Exp[(logits - Max[logits])/T]; p = p/Total[p];
    sampled = RandomChoice[p -> Range[Length[p]]];
    AppendTo[gen, sampled],
    {n}];
  (* Decode via inverse of encoder: NetDecoder *)
  Module[{decoder = NetExtract[gpt, "Output"]},
    If[Head[decoder] === NetDecoder,
      decoder[gen],
      StringJoin[ToString /@ gen]]
  ]
];

prompts = {
  "User: What is a sonnet?\nAssistant:",
  "User: Tell me a Shakespeare quote.\nAssistant:",
  "User: How are you?\nAssistant:"
};

SeedRandom[10];
Print["--- generations from SFT model ---"];
samples = Association @ Map[
  Function[p,
    Print["PROMPT: ", p];
    Module[{g = generate[sftLogitModel, p, 60, 0.7]},
      Print[g];
      Print[""];
      p -> g]],
  prompts];

summary = <|
  "elapsedMinutes" -> Round[elapsed/60., 0.1],
  "rounds" -> Length[trained["RoundLossList"]],
  "roundLossList" -> N /@ trained["RoundLossList"],
  "finalLoss" -> trained["FinalRoundLoss"],
  "generations" -> samples
|>;
Export[FileNameJoin[{dataDir, "gpt2_sft_summary.json"}], summary];
Print["Saved gpt2_sft_summary.json"];
Print["DONE."];
