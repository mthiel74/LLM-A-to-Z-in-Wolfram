#!/usr/bin/env wolframscript
(* Production GPU SFT submission.

   Args (positional, with defaults):
     1. modelSize   ("117M" | "345M" | "774M")  default 345M
     2. nExamples   integer                     default 51760 (full Alpaca)
     3. nEpochs     integer                     default 1
     4. jobLabel    string                      default "discussion-medium"

   Submits one RemoteBatchJob to GPU1xL40S that:
     - Downloads the matching GPT-2 size via NetModel (cached cloud-side)
     - Fetches Alpaca-cleaned (~44 MB) from HuggingFace
     - Performs the strip-and-NetMapOperator surgery
     - Runs NetTrain on GPU with periodic CloudObject checkpointing every
       ~20 minutes (TrainingProgressFunction)
     - CloudPuts the final stripped model
     - Returns a small Association: timing, loss, sample generations,
       CloudObject URL of the trained net

   The RemoteBatchSubmit second argument is NOT wrapped in Hold -- the
   function holds its argument internally; an explicit Hold survives to
   the remote and prevents evaluation (we burned ~£0.27 learning this).   *)

modelSize    = If[Length[$ScriptCommandLine] >= 2, $ScriptCommandLine[[2]], "345M"];
nExamples    = If[Length[$ScriptCommandLine] >= 3,
  ToExpression[$ScriptCommandLine[[3]]], 51760];
nEpochs      = If[Length[$ScriptCommandLine] >= 4,
  ToExpression[$ScriptCommandLine[[4]]], 1];
jobLabel     = If[Length[$ScriptCommandLine] >= 5, $ScriptCommandLine[[5]],
  "discussion-" <> ToLowerCase[modelSize]];
machineClass = If[Length[$ScriptCommandLine] >= 6, $ScriptCommandLine[[6]],
  "GPU1xL40S"];

cloudPath = "discussion-model/" <> jobLabel <> ".wlnet";

Print["=== submitting GPU SFT job ==="];
Print["  modelSize:    ", modelSize];
Print["  nExamples:    ", nExamples];
Print["  nEpochs:      ", nEpochs];
Print["  machineClass: ", machineClass];
Print["  cloud:        ", cloudPath];

env = RemoteBatchSubmissionEnvironment["WolframBatch",
  <|"RemoteMachineClass" -> machineClass|>];

job = RemoteBatchSubmit[env,
  Module[{out = <||>, gpt, encoder, embedding, decoder, classifier, trunk,
          alpacaRaw, formatted, dataset, wrapped, trained, stripped, decFn,
          prompts, generations, t0, finalObj, V},
    (* ---- model load ---- *)
    out["modelSize"] = modelSize;
    out["nExamples"] = nExamples;
    out["nEpochs"]   = nEpochs;
    out["startTime"] = DateString[];
    out["machine"]   = $MachineName;

    t0 = AbsoluteTime[];
    gpt = NetModel[{"GPT-2 Transformer Trained on WebText Data",
                    "Size" -> modelSize, "Task" -> "LanguageModeling"}];
    out["modelLoadSeconds"] = AbsoluteTime[] - t0;
    out["params"] = Information[gpt, "ArraysTotalElementCount"];

    encoder    = NetExtract[gpt, "Input"];
    embedding  = NetExtract[gpt, "embedding"];
    decoder    = NetExtract[gpt, "decoder"];
    classifier = NetExtract[gpt, "classifier"];
    trunk = NetChain[{embedding, decoder, NetMapOperator[classifier]}];
    V = 50257;

    (* ---- dataset ---- *)
    t0 = AbsoluteTime[];
    alpacaRaw = URLExecute[
      "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json",
      "RawJSON"];
    out["alpacaTotalRows"] = Length[alpacaRaw];
    alpacaRaw = Take[alpacaRaw, UpTo[nExamples]];
    out["alpacaUsedRows"] = Length[alpacaRaw];

    formatPair[ex_] := StringJoin[
      "User: ", ex["instruction"],
      If[StringLength[ex["input"]] > 0, " " <> ex["input"], ""],
      "\nAssistant: ", ex["output"], "\n"];
    formatted = formatPair /@ alpacaRaw;

    dataset = Table[
      Module[{ids = encoder[s]},
        If[Length[ids] >= 129, ids = ids[[;; 129]],
                                ids = PadRight[ids, 129, 1]];
        <|"Input" -> ids[[;; -2]], "Output" -> ids[[2 ;;]]|>],
      {s, formatted}];
    out["datasetBuildSeconds"] = AbsoluteTime[] - t0;

    (* ---- training ---- *)
    (* Periodic CloudPut from inside TrainingProgressFunction is the
       suspected source of the OOM on the previous run (each call
       materializes a 1.4 GB NetTake copy and GC may not reclaim
       between batches).  Trust the final CloudPut; no intermediate
       checkpoints.  Add an explicit TimeConstraint so the job can't run
       past its budget if something goes wrong.                              *)
    wrapped = NetChain[{trunk, NetMapOperator[SoftmaxLayer[]]}];

    SeedRandom[42];
    t0 = AbsoluteTime[];
    (* BatchSize=2 instead of 4 to halve activation memory.  The previous
       large run got to batch 2498/2500 then OOM'd in the post-train save
       cascade -- a halved activation footprint leaves us more headroom
       for the NetTake/CloudPut copy chain that follows. *)
    trained = NetTrain[wrapped, dataset, All,
      BatchSize -> 2,
      LearningRate -> 5.*^-5,
      MaxTrainingRounds -> nEpochs,
      LossFunction -> CrossEntropyLossLayer["Index"],
      TargetDevice -> "GPU",
      Method -> "ADAM",
      TrainingProgressReporting -> "Print"];
    out["trainSeconds"] = AbsoluteTime[] - t0;
    out["roundLossList"] = N /@ trained["RoundLossList"];
    out["finalLoss"]     = N[trained["FinalRoundLoss"]];
    out["batchesPerRound"] = Length[trained["BatchLossList"]] / nEpochs;

    (* ---- save final model BEFORE generations, with aggressive GC ----
       The previous Job B died here on large (3 GB net): NetTake copy +
       CloudPut serialization buffer + still-resident trained net pushed
       past memory.  Pull out the stripped net, drop every other
       reference, hint GC, *then* CloudPut.  Generations happen later
       and can rebuild any working memory they need.                       *)
    stripped = NetTake[trained["TrainedNet"], Length[trained["TrainedNet"]] - 1];
    trained = .;
    wrapped = .;
    Share[];
    ClearSystemCache[];
    finalObj = CloudPut[stripped, CloudObject[cloudPath]];
    out["modelCloudObject"] = ToString[finalObj];

    (* ---- sample generations ---- *)
    decFn = Function[ids,
      StringJoin @ NetExtract[gpt, "Output"] @ Table[
        Module[{v = ConstantArray[0., V]}, v[[i]] = 1.; v], {i, ids}]];

    prompts = {
      "User: What are the three primary colors?\nAssistant:",
      "User: How are you today?\nAssistant:",
      "User: Give three tips for staying healthy.\nAssistant:",
      "User: Who wrote Hamlet?\nAssistant:",
      "User: What's a good way to learn a new language?\nAssistant:",
      "User: Tell me a short joke.\nAssistant:",
      "User: What is the capital of France?\nAssistant:",
      "User: Recommend a book to read.\nAssistant:"};

    SeedRandom[2026];
    generations = Association @ Map[
      Function[p,
        Module[{ids = encoder[p], generated = {}, scores, last, probs,
                nextId, j},
          Do[
            scores = stripped[Take[ids, -Min[Length[ids], 256]]];
            last = Last[scores] / 0.7;
            Module[{cut = Sort[last, Greater][[40]]},
              last = Map[If[# < cut, -Infinity, #] &, last]];
            probs = Exp[last - Max[last]];
            probs = probs / Total[probs];
            nextId = RandomChoice[probs -> Range[Length[probs]]];
            AppendTo[ids, nextId];
            AppendTo[generated, nextId];
            If[Length[generated] >= 80, Break[]],
            {j, 80}];
          p -> decFn[generated]]],
      prompts];
    out["generations"] = generations;

    out["endTime"]    = DateString[];
    out["totalJobSeconds"] = SessionTime[];
    out]];

Print["=== submission result ==="];
Print["  UUID: ", job["JobUUID"]];
Print["  status: ", job["JobStatus"]];
Print["  machine: ", job["RemoteMachineClass"]];
Print[""];
Print["Track with:"];
Print["  RemoteBatchJobObject[\"", job["JobUUID"], "\"]"];
