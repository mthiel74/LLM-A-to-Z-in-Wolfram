(* ::Package:: *)

(* discussion-model: scale-up SFT of GPT-2 small (124M) on Alpaca-cleaned.
   Goal: a chat-formatted model that holds a short, coherent conversation.

   Why GPT-2 small and not medium?
   * GPT-2 medium hits an MXNet operator-shape bug when we strip the inference
     tail and re-chain for training (expected [4096], got [1024] at _copyto).
   * Mac mini CPU per-example wall time for medium projects to ~184 hrs/epoch
     on Alpaca-cleaned full set anyway; medium is only viable via cloud.
   * GPT-2 small training is proven by part-06 Path A; we scale it from
     20 toy pairs to 5000 real instruction pairs and train for 2-3 epochs.

   Public entry points:
     fetchAlpacaCleaned[]                 -- ensure data/alpaca_cleaned.json exists
     loadAlpacaExamples[n]                -- load first n examples (as Associations)
     formatPair[ex]                       -- one example -> "User:...\nAssistant:...\n"
     loadGpt2Small[]                      -- NetModel pull, returns the full LM
     buildTrainableTrunk[gpt2Net]         -- strip SequenceLast+Softmax, wrap
                                            classifier for per-position output
     buildSftDataset[strings, encoder, contextLen, padId]
                                          -- list of <|Input,Output|>
     trainOvernight[trunk, dataset, hp, checkpointPath]
                                          -- NetTrain with periodic checkpoint
     generate[net, encoder, decoder, prompt, hp]
                                          -- autoregressive sampling
     chatTurn[net, history, userMsg, encoder, decoder, hp]   *)

BeginPackage["LLMAtoZ`Discussion`"];

fetchAlpacaCleaned;
loadAlpacaExamples;
formatPair;
loadGpt2Small;
buildTrainableTrunk;
makeIdDecoder;
buildSftDataset;
trainOvernight;
generate;
chatTurn;

Begin["`Private`"];

(* ---------- dataset ---------- *)

$datasetUrl =
  "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json";

(* Snapshot the package directory at load time -- $InputFileName is only the
   package file's path during the Get[] call. *)
$pkgDir  = DirectoryName[$InputFileName];
$dataDir = FileNameJoin[{ParentDirectory[$pkgDir], "data"}];

fetchAlpacaCleaned[] := Module[
  {path = FileNameJoin[{$dataDir, "alpaca_cleaned.json"}]},
  If[!FileExistsQ[path],
    If[!DirectoryQ[$dataDir], CreateDirectory[$dataDir]];
    Print["Fetching Alpaca-cleaned (44 MB) ..."];
    URLDownload[$datasetUrl, path]];
  path];

loadAlpacaExamples[n_ : All] := Module[{raw},
  raw = Import[fetchAlpacaCleaned[], "RawJSON"];
  If[n === All, raw, raw[[1 ;; UpTo[n]]]]];

formatPair[ex_Association] := StringJoin[
  "User: ", ex["instruction"],
  If[StringLength[ex["input"]] > 0, " " <> ex["input"], ""],
  "\nAssistant: ", ex["output"], "\n"];

(* ---------- model surgery ---------- *)

loadGpt2Small[] := NetModel[{
  "GPT-2 Transformer Trained on WebText Data",
  "Size" -> "117M",
  "Task" -> "LanguageModeling"}];

(* The pretrained net is:
        embedding -> decoder -> SequenceLastLayer -> classifier -> SoftmaxLayer
   For SFT we need per-position logits.  Drop SequenceLast and SoftmaxLayer;
   wrap the classifier with NetMapOperator so it applies row-wise to the
   (seqLen, dModel) tensor coming out of the decoder.  This is the proven
   pattern from part-06/Path A. *)
buildTrainableTrunk[gpt2_NetChain] := NetChain[{
  NetExtract[gpt2, "embedding"],
  NetExtract[gpt2, "decoder"],
  NetMapOperator[NetExtract[gpt2, "classifier"]]}];

(* GPT-2's bundled NetDecoder expects probability vectors, not integer IDs.
   For autoregressive generation we want a function (idList -> string) that
   reuses the bundled token list without writing our own vocabulary loader.
   The trick: turn each integer ID into a length-50257 one-hot row and batch
   them through the existing decoder.  Sparse representation keeps memory
   small even for long generations.                                          *)
makeIdDecoder[gpt2_NetChain] := Module[{dec = NetExtract[gpt2, "Output"], V = 50257},
  Function[ids,
    (* NetDecoder accepts dense probability vectors only; build one-hots
       densely.  ~30 MB transient for a 100-token output -- fine for chat
       so long as we only call it once at the end of generation.            *)
    StringJoin @ dec @ Table[
      Module[{v = ConstantArray[0., V]}, v[[i]] = 1.; v], {i, ids}]]];

(* ---------- dataset assembly ---------- *)

(* encoder is the original GPT-2 NetEncoder (string -> ids in 1..50257).
   Each row is encoded, padded/truncated to contextLen+1, then split into
   shifted Input/Output pairs.  *)
buildSftDataset[strings_List, encoder_, contextLen_Integer, padId_Integer : 1] :=
  Module[{ids},
    Table[
      ids = encoder[s];
      If[Length[ids] >= contextLen + 1,
        ids = ids[[;; contextLen + 1]],
        ids = PadRight[ids, contextLen + 1, padId]];
      <|"Input" -> ids[[;; -2]], "Output" -> ids[[2 ;;]]|>,
      {s, strings}]];

(* ---------- training ---------- *)

defaultHp = <|
  "BatchSize" -> 4,
  "LearningRate" -> 5.*^-5,
  "MaxRounds" -> 3,
  "ContextLen" -> 128,
  "CheckpointEvery" -> Quantity[30, "Minutes"],
  "TimeGoal" -> Quantity[10, "Hours"]|>;

(* Train GPT-2 small on a chat-formatted dataset.  Periodically checkpoints
   to disk (every CheckpointEvery wall time) so the user can interrupt and
   resume with a usable model. *)
trainOvernight[trunk_NetChain, dataset_List, userHp_Association : <||>,
               checkpointPath_String : Automatic] := Module[
  {hp, wrapped, ckpt, lastCkptTime, progressFn, trained, stripped, t0},
  hp = Join[defaultHp, userHp];
  ckpt = If[checkpointPath === Automatic,
    FileNameJoin[{$dataDir, "discussion_sft_ckpt.wlnet"}],
    checkpointPath];
  wrapped = NetChain[{trunk, NetMapOperator[SoftmaxLayer[]]}];
  lastCkptTime = AbsoluteTime[];
  t0 = AbsoluteTime[];
  progressFn = Function[assoc,
    Module[{now = AbsoluteTime[]},
      If[now - lastCkptTime > QuantityMagnitude[hp["CheckpointEvery"], "Seconds"],
        Module[{logitsNet = NetTake[assoc["Net"], Length[assoc["Net"]] - 1]},
          Export[ckpt, logitsNet];
          Print["[", DateString[], "] ckpt @ batch ", assoc["Batch"],
                ", round ", assoc["Round"],
                ", batch_loss ", N[assoc["BatchLoss"]],
                " -> ", ckpt]];
        lastCkptTime = now]]];
  trained = NetTrain[
    wrapped, dataset, All,
    BatchSize -> hp["BatchSize"],
    LearningRate -> hp["LearningRate"],
    MaxTrainingRounds -> hp["MaxRounds"],
    TimeGoal -> hp["TimeGoal"],
    LossFunction -> CrossEntropyLossLayer["Index"],
    TargetDevice -> "CPU",
    Method -> "ADAM",
    TrainingProgressFunction -> progressFn,
    TrainingProgressReporting -> "Print"];
  stripped = NetTake[trained["TrainedNet"], Length[trained["TrainedNet"]] - 1];
  <|"net" -> stripped,
    "roundLoss" -> trained["RoundLossList"],
    "finalLoss" -> trained["FinalRoundLoss"],
    "wallSeconds" -> AbsoluteTime[] - t0,
    "hp" -> hp,
    "ckpt" -> ckpt|>];

(* ---------- generation ---------- *)

(* The trained `trunk` accepts integer-ID sequences directly through the
   "Class" NetEncoder. To use the bundled GPT-2 BPE encoder for input and the
   matching BPE decoder for output, we route through `encoderFn` and
   `decoderFn`. *)
generate[net_, encoderFn_, decoderFn_, prompt_String, hp_Association : <||>] :=
  Module[{opts, maxNew, T, topK, ids, scores, last, probs, nextId, j,
          generated, decoded, stopAt, stopIds, tail, ctxLen, recent,
          uniqueRecent},
  opts = Join[<|"MaxNew" -> 80, "Temperature" -> 0.7, "TopK" -> 40,
               "ContextLen" -> 1024, "StopOn" -> "\nUser:"|>, hp];
  maxNew = opts["MaxNew"]; T = opts["Temperature"];
  topK = opts["TopK"]; stopAt = opts["StopOn"]; ctxLen = opts["ContextLen"];
  (* Encode the stop string once so we can compare at the ID level and
     avoid the cost of decoding every step. *)
  stopIds = encoderFn[stopAt];
  ids = encoderFn[prompt];
  generated = {};
  Do[
    scores = net[Take[ids, -Min[Length[ids], ctxLen]]];
    last = Last[scores] / T;
    If[topK < Length[last],
      Module[{cut = Sort[last, Greater][[topK]]},
        last = Map[If[# < cut, -Infinity, #] &, last]]];
    probs = Exp[last - Max[last]];
    probs = probs / Total[probs];
    (* RandomChoice errors if probs degenerates (single number, all
       zeros, NaN, etc.).  Fall back to argmax in those cases. *)
    nextId = If[
      ListQ[probs] && Length[probs] > 0 && NumericQ[Total[probs]] &&
      Total[probs] > 0,
      RandomChoice[probs -> Range[Length[probs]]],
      First @ Ordering[last, -1]];
    AppendTo[ids, nextId];
    AppendTo[generated, nextId];
    (* Stop conditions, evaluated in order: *)
    (* 1. Natural stop: model emitted the chat-template prefix "\nUser:" *)
    tail = Take[generated, -Min[Length[generated], Length[stopIds]]];
    If[tail === stopIds, Break[]];
    (* 2. Degenerate repetition: same token 3+ times in a row, OR the
       last 6 tokens are dominated by 2 or fewer unique values.  The
       plain-text format has no <|eot|> so the model can babble; this
       catches the " t t t" tail. *)
    If[Length[generated] >= 3 &&
       MatchQ[Take[generated, -3], {x_, x_, x_}],
      Break[]];
    If[Length[generated] >= 6,
      uniqueRecent = Length @ Union @ Take[generated, -6];
      If[uniqueRecent <= 2, Break[]]],
    {j, maxNew}];
  decoded = decoderFn[generated];
  StringReplace[decoded, stopAt ~~ ___ :> ""]];

chatTurn[net_, history_List, userMsg_String, encoderFn_, decoderFn_,
         hp_Association : <||>] := Module[{prompt},
  prompt = StringJoin[
    Map[Function[turn,
      "User: " <> turn["user"] <> "\nAssistant: " <> turn["assistant"] <> "\n"],
      history],
    "User: ", userMsg, "\nAssistant:"];
  generate[net, encoderFn, decoderFn, prompt, hp]];

End[];
EndPackage[];
