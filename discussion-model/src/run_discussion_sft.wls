#!/usr/bin/env wolframscript
(* Overnight SFT: GPT-2 small (124M) on 10,000-example Alpaca-cleaned subset
   for 3 epochs.  Projected wall ~3-4 hours on Mac mini CPU at batch_size 4,
   contextLen 128.  Saves a checkpoint every 30 minutes so an early interrupt
   yields a usable model.

   Outputs:
     discussion-model/data/discussion_sft.wlnet     -- final trained model
     discussion-model/data/discussion_sft_ckpt.wlnet -- periodic checkpoint
     discussion-model/data/discussion_progress.txt   -- live progress log
     discussion-model/data/discussion_summary.json   -- training summary

   Invoke with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
       -file discussion-model/src/run_discussion_sft.wls

   Or with a smaller subset for a quick run:
     ... -file run_discussion_sft.wls 1000 2    (1000 examples, 2 epochs)         *)

$HistoryLength = 0;

(* Args: [nExamples=10000] [nEpochs=3] *)
nExamples = If[Length[$ScriptCommandLine] >= 2,
  ToExpression[$ScriptCommandLine[[2]]], 10000];
nEpochs   = If[Length[$ScriptCommandLine] >= 3,
  ToExpression[$ScriptCommandLine[[3]]], 3];

scriptDir = DirectoryName[$InputFileName];
partDir   = ParentDirectory[scriptDir];
dataDir   = FileNameJoin[{partDir, "data"}];
If[!DirectoryQ[dataDir], CreateDirectory[dataDir]];

progressLog = FileNameJoin[{dataDir, "discussion_progress.txt"}];
ckptPath    = FileNameJoin[{dataDir, "discussion_sft_ckpt.wlnet"}];
finalPath   = FileNameJoin[{dataDir, "discussion_sft.wlnet"}];
summaryPath = FileNameJoin[{dataDir, "discussion_summary.json"}];

logLine[s_] := (
  Print["[", DateString[], "] ", s];
  WriteString[progressLog,
    "[" <> DateString[] <> "] " <> ToString[s] <> "\n"]);

SetDirectory[scriptDir];
Get["Discussion.wl"];

Quiet[DeleteFile[progressLog]];

logLine["=== run_discussion_sft.wls starting ==="];
logLine[Row[{"nExamples=", nExamples, "  nEpochs=", nEpochs}]];

(* Load model *)
logLine["loading GPT-2 small ..."];
t0 = AbsoluteTime[];
gpt = LLMAtoZ`Discussion`loadGpt2Small[];
logLine[Row[{"loaded in ", Round[AbsoluteTime[] - t0, 0.1], " s; params=",
  Information[gpt, "ArraysTotalElementCount"]}]];

encoder = NetExtract[gpt, "Input"];
trunk   = LLMAtoZ`Discussion`buildTrainableTrunk[gpt];
logLine[Row[{"trunk params=", Information[trunk, "ArraysTotalElementCount"]}]];

(* Fetch + format dataset *)
logLine[Row[{"fetching ", nExamples, " Alpaca examples ..."}]];
examples  = LLMAtoZ`Discussion`loadAlpacaExamples[nExamples];
formatted = LLMAtoZ`Discussion`formatPair /@ examples;
logLine[Row[{"formatted ", Length[formatted], " examples"}]];

contextLen = 128;
dataset = LLMAtoZ`Discussion`buildSftDataset[formatted, encoder, contextLen];
logLine[Row[{"dataset built (contextLen=", contextLen, ", batch_size=4)"}]];

(* Run training *)
logLine[Row[{"starting SFT for ", nEpochs, " epochs ..."}]];
t0 = AbsoluteTime[];
result = LLMAtoZ`Discussion`trainOvernight[
  trunk, dataset,
  <|"MaxRounds" -> nEpochs,
    "BatchSize" -> 4,
    "LearningRate" -> 5.*^-5,
    "CheckpointEvery" -> Quantity[30, "Minutes"],
    "TimeGoal" -> Quantity[6, "Hours"]|>,
  ckptPath];
elapsed = AbsoluteTime[] - t0;
logLine[Row[{"training done in ", Round[elapsed/60., 0.1], " min"}]];
logLine[Row[{"round losses: ", N /@ result["roundLoss"]}]];
logLine[Row[{"final loss: ", N[result["finalLoss"]]}]];

(* Save final checkpoint *)
Export[finalPath, result["net"]];
logLine[Row[{"saved final model to ", finalPath, " (",
  Round[FileByteCount[finalPath] / 2^20, 0.1], " MB)"}]];

(* Eval prompts -- evidence the model actually chats now *)
decoderFn = LLMAtoZ`Discussion`makeIdDecoder[gpt];
evalPrompts = {
  "User: What is a sonnet?\nAssistant:",
  "User: How are you today?\nAssistant:",
  "User: Give three tips for staying healthy.\nAssistant:",
  "User: Who wrote Hamlet?\nAssistant:",
  "User: What's a good way to learn a new language?\nAssistant:",
  "User: Tell me a short joke.\nAssistant:"};

logLine["--- post-training generations ---"];
SeedRandom[2026];
generations = Association @ Map[
  Function[p,
    Module[{g = LLMAtoZ`Discussion`generate[
        result["net"], encoder, decoderFn, p,
        <|"MaxNew" -> 80, "Temperature" -> 0.7, "TopK" -> 40|>]},
      logLine[Row[{"PROMPT: ", p}]];
      logLine[Row[{"REPLY:  ", g}]];
      p -> g]],
  evalPrompts];

(* Save summary *)
summary = <|
  "elapsedMinutes" -> Round[elapsed/60., 0.1],
  "nExamples" -> nExamples,
  "nEpochs" -> nEpochs,
  "roundLossList" -> N /@ result["roundLoss"],
  "finalLoss" -> N[result["finalLoss"]],
  "modelPath" -> finalPath,
  "modelSizeMB" -> Round[FileByteCount[finalPath] / 2^20, 0.1],
  "generations" -> Association[
    StringReplace[#1, {"\n" -> "\\n", "\"" -> "'"}] -> #2 & @@@ Normal[generations]
  ]
|>;
Export[summaryPath, summary];
logLine[Row[{"summary saved to ", summaryPath}]];
logLine["=== run_discussion_sft.wls DONE ==="];
