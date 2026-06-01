#!/usr/bin/env wolframscript
(* Full-vocab calibration for Part 9 (option 1: the 30M tier).

   Differs from submit_calibration.wls (the synthetic smoke test) in three
   ways that make the throughput number meaningful for production sizing:

     1. Real data: a FineWeb-Edu token stream (vocab 50257), produced by
        preprocess_fineweb.py and SHIPPED WITH THE JOB as a packed integer
        array (the same WXF-input path the framework already uses for its
        own input file -- proven to work, and independent of whether the
        `aws` CLI is present in the container).
     2. Full vocabulary 50257, so the embedding term and the 50257-way output
        softmax -- the dominant cost the synthetic vocab=1000 run hid -- are
        measured honestly.
     3. seqLen 512 (vs 128), a realistic context for the 30M model.

   We also PROBE for the aws CLI in-job (non-fatal): Pretraining.wl's S3
   checkpointing assumes `aws` is on PATH, which is unverified.  The probe
   tells us whether that harness needs a WL-native rewrite before production.

   Cost cap: NetTrain TimeGoal 1200s + TimeConstrained 1500s.  Worst case
   ~25 min @ ~$0.25/hr = ~$0.10. *)

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];

attentionSrc = Import[
  FileNameJoin[{repo, "part-02-self-attention", "src", "Attention.wl"}], "Text"];
nanoSrc = Import[
  FileNameJoin[{repo, "part-09-from-scratch", "src", "NanoChat.wl"}], "Text"];

(* Ship 12M tokens (~24 MB packed array) -- plenty for a time-boxed
   calibration, comfortably within the WXF-input path. *)
nShip     = 12000000;
allTokens = BinaryReadList[
  FileNameJoin[{repo, "part-09-from-scratch", "data", "fineweb_edu_calib.bin"}],
  "UnsignedInteger16"];
tokenData = Take[allTokens, UpTo[nShip]];

Print["Loaded Attention.wl: ", StringLength[attentionSrc], " chars"];
Print["Loaded NanoChat.wl : ", StringLength[nanoSrc], " chars"];
Print["Shipping tokens    : ", Length[tokenData],
  " (range ", Min[tokenData], "..", Max[tokenData], ")"];

bucket  = "EXAMPLE-BUCKET";
ckptUri = "s3://" <> bucket <> "/runs/calib-30M-fineweb/ckpt.wlnet";

awsService = ServiceConnect["AWS"];
If[Head[awsService] =!= ServiceObject,
  Print["AWS connect failed: ", awsService]; Quit[1]];

env = RemoteBatchSubmissionEnvironment["AWSBatch", <|
  "JobQueue" ->
    "arn:aws:batch:REGION:ACCOUNT_ID:job-queue/MY-SPOT-QUEUE",
  "JobDefinition" ->
    "arn:aws:batch:REGION:ACCOUNT_ID:job-definition/MY-GPU-JOBDEF:1",
  "IOBucket" -> bucket,
  "ServiceObject" -> awsService
|>];

Print["Submitting full-vocab 30M calibration..."];

job = RemoteBatchSubmit[env,
  TimeConstrained[
    Module[{out = <||>, spec, mdl, seqLen, vocab, ids, nwin, windows,
            data, trainable, t0, t1, trained, smi, batchSize, awsProbe},
      out["job"]     = "nano-30M fineweb calibration";
      out["t_start"] = DateString[Now, "ISODateTime"];

      Export["/tmp/Attention.wl", attentionSrc, "Text"];
      Export["/tmp/NanoChat.wl",  nanoSrc,      "Text"];
      Get["/tmp/Attention.wl"];
      Get["/tmp/NanoChat.wl"];

      out["machine"]   = $MachineName;
      out["sysid"]     = $SystemID;
      out["wlversion"] = $VersionNumber;

      smi = Quiet @ Check[
        RunProcess[{"nvidia-smi",
          "--query-gpu=index,name,memory.total,driver_version",
          "--format=csv,noheader"}, "StandardOutput"], "no-nvidia-smi"];
      out["nvidia_smi"] = smi;

      (* probe: is the aws CLI available? (informs Pretraining.wl design) *)
      awsProbe = Quiet @ Check[
        RunProcess[{"aws", "--version"}], <|"ExitCode" -> "no-aws-cli"|>];
      out["aws_cli"] = awsProbe["ExitCode"];

      (* tokens arrive via the shipped variable -- no in-container S3 needed *)
      ids = tokenData;
      out["n_tokens"]  = Length[ids];
      out["token_min"] = Min[ids];
      out["token_max"] = Max[ids];

      (* --- 30M nano model at FULL vocab --- *)
      vocab  = 50257;
      seqLen = 512;
      spec = LLMAtoZ`NanoChat`nanoModelSpec30M[vocab, seqLen];
      mdl = LLMAtoZ`NanoChat`nanoModel[
        spec["dModel"], spec["nHeads"], spec["nLayers"],
        spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]];
      out["spec"] = spec;
      out["paramCount"] = Information[mdl, "ArraysTotalElementCount"];

      (* --- window the stream into next-token examples --- *)
      nwin = Quotient[Length[ids], seqLen + 1];
      windows = Partition[Take[ids, nwin*(seqLen + 1)], seqLen + 1];
      data = Map[
        <|"Input" -> #[[;; -2]], "Output" -> #[[2 ;;]]|> &, windows];
      out["n_windows"] = Length[data];

      (* nanoModel emits raw logits.  CrossEntropyLossLayer["Index"] expects a
         PROBABILITY vector -- feeding logits makes the loss unbounded below
         (training-gotchas #1: loss descends past 0 into negatives as a logit
         is driven to +Inf).  Insert a row-wise SoftmaxLayer for training; the
         saved checkpoint (NetExtract ... "net") keeps the logit head for
         sampling. *)
      trainable = NetGraph[
        <|"net" -> mdl,
          "sm"  -> NetMapOperator[SoftmaxLayer[]],
          "loss" -> CrossEntropyLossLayer["Index"]|>,
        {NetPort["Input"] -> "net" -> "sm" -> NetPort["loss", "Input"],
         NetPort["Output"] -> NetPort["loss", "Target"]}];

      batchSize = 8;
      out["batchSize"] = batchSize;

      t0 = AbsoluteTime[];
      trained = NetTrain[trainable, data, All,
        BatchSize         -> batchSize,
        MaxTrainingRounds -> 3,
        TimeGoal          -> 1200,
        TargetDevice      -> "GPU",
        TrainingProgressReporting -> None];
      t1 = AbsoluteTime[];

      out["train_wall_s"]  = N[t1 - t0];
      out["train_status"]  = Head[trained];
      If[Head[trained] === NetTrainResultsObject,
        With[{bl = trained["BatchLossList"], nb = Length[trained["BatchLossList"]]},
          out["round_losses"]   = trained["RoundLossList"];
          out["n_batches"]      = nb;
          out["loss_first"]     = First[bl];
          out["loss_min"]       = Min[bl];
          out["loss_last"]      = Last[bl];
          out["loss_tail_mean"] = N[Mean[Take[bl, -Min[100, nb]]]];
          (* coarse curve: ~50 evenly spaced points for plotting later *)
          out["loss_curve"]     = bl[[1 ;; ;; Max[1, Quotient[nb, 50]]]];
        ];
        out["throughput_examples_per_s"] =
          If[(t1 - t0) > 0,
            N[(Length[trained["BatchLossList"]]*batchSize)/(t1 - t0)], "n/a"];
        out["throughput_tokens_per_s"] =
          If[(t1 - t0) > 0,
            N[(Length[trained["BatchLossList"]]*batchSize*seqLen)/(t1 - t0)],
            "n/a"];
        (* best-effort checkpoint upload via aws CLI (non-fatal) *)
        out["ckpt_upload_exit"] = Quiet @ Check[
          Export["/tmp/ckpt.wlnet",
            NetExtract[trained["TrainedNet"], "net"], "WLNet"];
          RunProcess[{"aws", "s3", "cp", "/tmp/ckpt.wlnet", ckptUri}]["ExitCode"],
          "ckpt-save-failed"];
      ];
      out["t_end"] = DateString[Now, "ISODateTime"];
      out
    ],
    1500,
    <|"out" -> "TimeConstrainedAt 1500s", "elapsed" -> 1500|>
  ]
];

Print["UUID:         ", job["JobUUID"]];
Print["Status:       ", job["JobStatus"]];
Export["/tmp/calib_fineweb_uuid.txt", job["JobUUID"], "Text"];
Print["UUID saved to /tmp/calib_fineweb_uuid.txt"];
