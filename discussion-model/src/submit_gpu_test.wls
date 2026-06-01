#!/usr/bin/env wolframscript
(* Submit a small GPU verification job to Wolfram Cloud.

   Goal: confirm that
   (a) our strip-and-NetMapOperator pattern trains GPT-2 small on GPU,
   (b) GPT-2 medium also trains on GPU (the path that hit an MXNet bug
       on local Mac mini CPU), and
   (c) we get a credible per-example wall time at L40S.

   The job:
   - Loads GPT-2 small + medium from NetModel (downloads on cloud side).
   - Fetches 100 Alpaca examples from HuggingFace.
   - Builds the trainable trunk for each (drop SequenceLast + Softmax,
     wrap classifier with NetMapOperator).
   - Runs 1 NetTrain round at batch_size 4 on TargetDevice -> "GPU".
   - Returns timing + final loss + a couple of sample generations.

   Estimated cost on GPU1xL40S (1120 credits/hr):
   - Model downloads + setup: ~90 s
   - GPT-2 small training (1 epoch, 100 examples, batch 4): ~30 s
   - GPT-2 medium training (1 epoch, 100 examples, batch 4): ~60 s
   - Generations: ~20 s
   Total: ~3.5 min = ~65 credits = ~$0.65 (rough).

   On Basic1x8 the same job would take ~10 min CPU, ~$0.20.
*)

env = RemoteBatchSubmissionEnvironment["WolframBatch",
  <|"RemoteMachineClass" -> "GPU1xL40S"|>];

(* RemoteBatchSubmit holds its second argument internally; an explicit Hold
   here would be shipped as `Hold[Module[...]]` and the remote's evaluation
   of that is the Hold itself (Module never runs).  Pass the Module bare.    *)
job = RemoteBatchSubmit[env,
  Module[{out = <||>, gpts, mediums, gpt, encoder, trunk, alpacaUrl,
          alpacaRaw, examples, formatted, dataset, trainable, t0, res,
          elapsed, makeIdDec, decFn, gen1, gen2},

    out["GPU info"] = Quiet @ Check[CUDAInformation[],
      "no CUDA / not on GPU"];
    out["Wolfram version"] = $WolframVersion;
    out["MachineName"] = $MachineName;

    (* --- helper: build trainable trunk from a GPT-2 NetModel net --- *)
    makeTrunk[net_] := NetChain[{
      NetExtract[net, "embedding"],
      NetExtract[net, "decoder"],
      NetMapOperator[NetExtract[net, "classifier"]]}];

    (* --- helper: id decoder (one-hots) --- *)
    makeIdDec[net_] := Module[{dec = NetExtract[net, "Output"], V = 50257},
      Function[ids, StringJoin @ dec @ Table[
        Module[{v = ConstantArray[0., V]}, v[[i]] = 1.; v], {i, ids}]]];

    (* --- helper: format Alpaca rows --- *)
    formatPair[ex_] := StringJoin[
      "User: ", ex["instruction"],
      If[StringLength[ex["input"]] > 0, " " <> ex["input"], ""],
      "\nAssistant: ", ex["output"], "\n"];

    (* --- helper: encode + pad --- *)
    encPad[enc_, txt_, len_] := Module[{ids = enc[txt]},
      If[Length[ids] >= len + 1, ids[[;; len + 1]],
                                  PadRight[ids, len + 1, 1]]];

    (* --- fetch alpaca-cleaned --- *)
    alpacaUrl = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json";
    out["dataset fetch start"] = AbsoluteTime[];
    alpacaRaw = URLExecute[alpacaUrl, "RawJSON"];
    out["dataset fetch end"] = AbsoluteTime[];
    out["alpaca total rows"] = Length[alpacaRaw];
    examples = Take[alpacaRaw, 100];
    formatted = formatPair /@ examples;
    dataset = Table[
      Module[{ids = encPad[#, s, 128] &},
        <|"placeholder" -> True|>],
      {s, formatted}];

    (* --- test each model size --- *)
    Do[
      Module[{size = sz, gptN, encN, trunkN, dsN, wrappedN, resN, elapsedN,
              decN, genN1, genN2},
        out[size <> ": start"] = DateString[];
        t0 = AbsoluteTime[];
        gptN = NetModel[{"GPT-2 Transformer Trained on WebText Data",
                         "Size" -> size, "Task" -> "LanguageModeling"}];
        out[size <> ": load time"] = AbsoluteTime[] - t0;
        out[size <> ": params"] = Information[gptN,
          "ArraysTotalElementCount"];

        encN = NetExtract[gptN, "Input"];
        trunkN = makeTrunk[gptN];

        dsN = Table[
          Module[{ids = encN[s]},
            If[Length[ids] >= 129, ids = ids[[;; 129]],
                                    ids = PadRight[ids, 129, 1]];
            <|"Input" -> ids[[;; -2]], "Output" -> ids[[2 ;;]]|>],
          {s, formatted}];

        wrappedN = NetChain[{trunkN, NetMapOperator[SoftmaxLayer[]]}];

        SeedRandom[7];
        t0 = AbsoluteTime[];
        resN = Quiet @ Check[
          NetTrain[wrappedN, dsN, All,
            BatchSize -> 4, LearningRate -> 5.*^-5,
            MaxTrainingRounds -> 1,
            LossFunction -> CrossEntropyLossLayer["Index"],
            TargetDevice -> "GPU",
            Method -> "ADAM",
            TrainingProgressReporting -> None],
          $Failed];
        elapsedN = AbsoluteTime[] - t0;
        out[size <> ": train wall"] = elapsedN;
        out[size <> ": train status"] = If[resN === $Failed,
          "FAILED", "OK"];

        If[resN =!= $Failed,
          out[size <> ": final loss"] = N[resN["FinalRoundLoss"]];

          (* Generate from the trained net *)
          Module[{strippedN = NetTake[resN["TrainedNet"],
                                       Length[resN["TrainedNet"]] - 1],
                  decFn},
            decFn = makeIdDec[gptN];
            genN1 = Catch[
              Module[{ids = encN["User: What are the three primary colors?\nAssistant:"],
                       generated = {}, scores, last, probs, nextId},
                Do[
                  scores = strippedN[Take[ids, -Min[Length[ids], 256]]];
                  last = Last[scores] / 0.7;
                  Module[{cut = Sort[last, Greater][[40]]},
                    last = Map[If[# < cut, -Infinity, #] &, last]];
                  probs = Exp[last - Max[last]];
                  probs = probs / Total[probs];
                  nextId = RandomChoice[probs -> Range[Length[probs]]];
                  AppendTo[ids, nextId];
                  AppendTo[generated, nextId],
                  {50}];
                decFn[generated]]];
            out[size <> ": gen1"] = genN1];

          (* second prompt *)
          Module[{strippedN = NetTake[resN["TrainedNet"],
                                       Length[resN["TrainedNet"]] - 1],
                  decFn},
            decFn = makeIdDec[gptN];
            SeedRandom[42];
            genN2 = Catch[
              Module[{ids = encN["User: How are you today?\nAssistant:"],
                       generated = {}, scores, last, probs, nextId},
                Do[
                  scores = strippedN[Take[ids, -Min[Length[ids], 256]]];
                  last = Last[scores] / 0.7;
                  Module[{cut = Sort[last, Greater][[40]]},
                    last = Map[If[# < cut, -Infinity, #] &, last]];
                  probs = Exp[last - Max[last]];
                  probs = probs / Total[probs];
                  nextId = RandomChoice[probs -> Range[Length[probs]]];
                  AppendTo[ids, nextId];
                  AppendTo[generated, nextId],
                  {40}];
                decFn[generated]]];
            out[size <> ": gen2"] = genN2]]],
      {sz, {"117M", "345M"}}];

    out["TOTAL JOB SECONDS"] = SessionTime[];
    out
  ]
];

Print["=== submission result ==="];
Print["  head: ", Head[job]];
Print["  UUID: ", job["JobUUID"]];
Print["  name: ", job["JobName"]];
Print["  status: ", job["JobStatus"]];
Print["  machine: ", job["RemoteMachineClass"]];
Print["  submitted at: ", DateString[]];
Print[""];
Print["Save this UUID; check status with:"];
Print["  RemoteBatchJobObject[\"", job["JobUUID"], "\"][\"JobStatus\"]"];
Print[""];
Print["Or fetch results with:"];
Print["  RemoteBatchJobObject[\"", job["JobUUID"], "\"][\"JobResult\"]"];
