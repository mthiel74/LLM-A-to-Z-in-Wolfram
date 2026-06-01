#!/usr/bin/env wolframscript
(* Single-GPU calibration smoke test for Part 9.

   Goal: confirm on AWS Batch g4dn.xlarge spot that

     1. wolframengine:latest can build a 30M nano model and find a T4 GPU
     2. NetTrain with TargetDevice -> "GPU" actually trains
     3. Loss decreases on synthetic data (sanity)
     4. Throughput in examples/sec matches our $0.30/hr extrapolation

   Strict cost cap: NetTrain TimeGoal 1500s + TimeConstrained 1800s +
   AWS Batch attemptDurationSeconds 2700s.  Worst case 45 min @ $0.30/hr
   = $0.22 ~= GBP 0.18.  Pre-approved budget: GBP 1. *)

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];

(* Inline the two WL packages as strings so they ship with the Module. *)
attentionSrc = Import[
  FileNameJoin[{repo, "part-02-self-attention", "src", "Attention.wl"}],
  "Text"];
nanoSrc = Import[
  FileNameJoin[{repo, "part-09-from-scratch", "src", "NanoChat.wl"}],
  "Text"];

Print["Loaded Attention.wl: ", StringLength[attentionSrc], " chars"];
Print["Loaded NanoChat.wl : ", StringLength[nanoSrc], " chars"];

awsService = ServiceConnect["AWS"];
If[Head[awsService] =!= ServiceObject,
  Print["AWS connect failed: ", awsService]; Quit[1]];

env = RemoteBatchSubmissionEnvironment["AWSBatch", <|
  "JobQueue" ->
    "arn:aws:batch:REGION:ACCOUNT_ID:job-queue/MY-SPOT-QUEUE",
  (* Must be the GPU job def (4 vCPU / 1x T4 / 14 GiB).  The earlier
     WolframJobDefinition-eba2bee7cac9a36 was a 1-vCPU no-GPU standard def,
     which would have left TargetDevice -> "GPU" with no device. *)
  "JobDefinition" ->
    "arn:aws:batch:REGION:ACCOUNT_ID:job-definition/MY-GPU-JOBDEF:1",
  "IOBucket" -> "EXAMPLE-BUCKET",
  "ServiceObject" -> awsService
|>];

Print["Submitting calibration run..."];

job = RemoteBatchSubmit[env,
  TimeConstrained[
    Module[{out = <||>, spec, mdl, n, seqLen, vocab, data, trainable,
            t0, t1, trained, smi},
      out["job"]   = "nano-30M calibration";
      out["t_start"] = DateString[Now, "ISODateTime"];

      (* Bring NanoChat + Attention into the kernel from the inline source. *)
      Export["/tmp/Attention.wl", attentionSrc, "Text"];
      Export["/tmp/NanoChat.wl",  nanoSrc,      "Text"];
      Get["/tmp/Attention.wl"];
      Get["/tmp/NanoChat.wl"];

      out["machine"]   = $MachineName;
      out["sysid"]     = $SystemID;
      out["wlversion"] = $WolframVersion;

      (* GPU visibility *)
      out["cuda_info"]    = Quiet @ Check[CUDAInformation[],            "no-cuda"];
      out["cuda_devices"] = Quiet @ Check[CUDAResourcesInformation[],   "no-cuda"];
      smi = Quiet @ Check[
        RunProcess[{"nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader"},
                   "StandardOutput"], "no-nvidia-smi"];
      out["nvidia_smi"] = smi;

      (* Build a 30M-spec nano model at tiny vocab so the embedding term
         does not dominate runtime on the smoke test. *)
      vocab  = 1000;
      seqLen = 128;
      spec = LLMAtoZ`NanoChat`nanoModelSpec30M[vocab, seqLen];
      mdl = LLMAtoZ`NanoChat`nanoModel[
        spec["dModel"], spec["nHeads"], spec["nLayers"],
        spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]];
      out["spec"] = spec;
      out["paramCount"] = Information[mdl, "ArraysTotalElementCount"];

      (* Synthetic next-token data: 5000 random sequences. *)
      SeedRandom[42];
      n = 5000;
      data = Table[
        With[{ids = RandomInteger[{1, vocab}, seqLen + 1]},
          <|"Input" -> ids[[;; -2]], "Output" -> ids[[2 ;;]]|>],
        n];

      trainable = NetGraph[
        <|"net" -> mdl, "loss" -> CrossEntropyLossLayer["Index"]|>,
        {NetPort["Input"] -> "net" -> NetPort["loss", "Input"],
         NetPort["Output"] -> NetPort["loss", "Target"]}];

      t0 = AbsoluteTime[];
      trained = NetTrain[trainable, data, All,
        BatchSize        -> 8,
        MaxTrainingRounds -> 10,
        TimeGoal          -> 1500,
        TargetDevice      -> "GPU",
        TrainingProgressReporting -> None];
      t1 = AbsoluteTime[];

      out["train_wall_s"] = N[t1 - t0];
      out["round_losses"] = If[Head[trained] === NetTrainResultsObject,
        trained["RoundLossList"], "n/a"];
      out["batch_loss_first10"] = If[Head[trained] === NetTrainResultsObject,
        Take[trained["BatchLossList"], UpTo[10]], "n/a"];
      out["batch_loss_last10"] = If[Head[trained] === NetTrainResultsObject,
        Take[trained["BatchLossList"], -UpTo[10]], "n/a"];
      out["n_batches"] = If[Head[trained] === NetTrainResultsObject,
        Length[trained["BatchLossList"]], 0];
      out["throughput_examples_per_s"] =
        If[Head[trained] === NetTrainResultsObject && (t1 - t0) > 0,
          N[(Length[trained["BatchLossList"]] * 8) / (t1 - t0)],
          "n/a"];
      out["train_status"] = Head[trained];
      out["t_end"] = DateString[Now, "ISODateTime"];
      out
    ],
    1800,
    <|"out" -> "TimeConstrainedAt 1800s",
      "elapsed" -> 1800|>
  ]
];

Print["UUID:               ", job["JobUUID"]];
Print["Status:             ", job["JobStatus"]];
Print["MachineClass:       ", job["RemoteMachineClass"]];

Export["/tmp/calibration_uuid.txt", job["JobUUID"], "Text"];
Print["UUID saved to /tmp/calibration_uuid.txt"];
