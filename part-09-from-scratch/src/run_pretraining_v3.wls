#!/usr/bin/env wolframscript
(* ============================================================
   run_pretraining_v3.wls  ---  Part 9 self-looping pretraining
   ============================================================

   v3 = v2 + a GLOBAL cosine learning-rate schedule across shards.

   v2 produced a working 30M model (ppl ~55) but plateaued ~shard 15: each
   shard's NetTrain reset its own learning rate, so training was 30 separate
   warmup/decay mini-runs instead of one global descent.  v3 fixes this by
   setting each shard's LR from a single cosine decay keyed to the shard index:
     lr_i = lrMin + 0.5 (lrMax - lrMin) (1 + cos(pi (i-1)/(N-1)))
   high LR on the early shards, decaying to lrMin on the last -- one smooth
   schedule over the whole run.  lrMax/lrMin come from the manifest.  Optimizer
   moment state still resets per shard (WL NetTrain doesn't expose it), but the
   dominant effect was the LR schedule.

   Everything else is identical to v2 (self-looping single job, S3 checkpoint
   between shards, job-def :2 spot auto-retry).

   Hardened replacement for run_pretraining.wls after the v1 run died:
   chunk 0's spot instance was reclaimed at 43 min and, because v1 only
   checkpointed at chunk end, lost the lot -- and the Mac-side orchestrator
   process itself died (memory pressure), so nothing resumed.

   v2 removes both failure modes:
     * ONE self-contained Batch job loops over ALL shards in-container,
       keeping the model in memory across shards and writing a checkpoint +
       progress marker to S3 between each shard.  No long-lived Mac process.
     * The job runs under job-def revision :2, whose retryStrategy auto-
       resubmits on spot reclamation ("Host EC2*"); the retried attempt reads
       progress.json + ckpt.wlnet from S3 and continues from the next shard.
     * Shards are small (20M tokens, ~20 min) so a reclaim loses <= one shard,
       and checkpointing happens BETWEEN shards (never inside NetTrain), so we
       avoid the periodic-CloudPut memory leak (training-gotchas #6).

   Usage:  wolframscript -file run_pretraining_v2.wls manifest.json
   The Mac process can exit right after submission; resilience is in AWS. *)

manifestPath = First[$ScriptCommandLine[[2 ;;]], $Failed];
If[manifestPath === $Failed || ! FileExistsQ[manifestPath],
  Print["ERROR: pass a manifest JSON path"]; Quit[1]];
cfg = Import[manifestPath, "RawJSON"];

bucket    = "EXAMPLE-BUCKET";
region    = "eu-central-1";
runPrefix = cfg["runPrefix"];
ckptKey   = runPrefix <> "/ckpt.wlnet";
progKey   = runPrefix <> "/progress.json";
shardKeys = cfg["shardKeys"];
seqLen    = Lookup[cfg, "seqLen", 512];
vocab     = Lookup[cfg, "vocab", 50257];
batchSize = Lookup[cfg, "batchSize", 8];
freshStart = Lookup[cfg, "freshStart", True];
lrMax     = Lookup[cfg, "lrMax", 0.002];
lrMin     = Lookup[cfg, "lrMin", 0.0002];

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
attentionSrc = Import[FileNameJoin[{repo, "part-02-self-attention", "src", "Attention.wl"}], "Text"];
nanoSrc = Import[FileNameJoin[{repo, "part-09-from-scratch", "src", "NanoChat.wl"}], "Text"];

Print["Run prefix : ", runPrefix, "   shards: ", Length[shardKeys]];

awsService = ServiceConnect["AWS"];
If[Head[awsService] =!= ServiceObject, Print["AWS connect failed"]; Quit[1]];
s3o = ServiceExecute[awsService, "GetService", <|"Name" -> "S3"|>];

If[freshStart,
  Quiet @ Check[ServiceExecute[s3o, "DeleteObject", <|"Bucket" -> bucket, "Key" -> ckptKey|>], Null];
  ServiceExecute[s3o, "PutObject", <|"Bucket" -> bucket, "Key" -> progKey,
    "Body" -> StringToByteArray[ExportString[<|"nextShard" -> 0|>, "JSON"]]|>];
  Print["Fresh start: checkpoint cleared, progress=0"]];

env = RemoteBatchSubmissionEnvironment["AWSBatch", <|
  "JobQueue" -> "arn:aws:batch:REGION:ACCOUNT_ID:job-queue/MY-SPOT-QUEUE",
  "JobDefinition" -> "arn:aws:batch:REGION:ACCOUNT_ID:job-definition/MY-GPU-JOBDEF:2",
  "IOBucket" -> bucket, "ServiceObject" -> awsService|>];

Print["Submitting self-looping pretraining job (job-def :2, spot auto-retry)..."];

job = RemoteBatchSubmit[env,
  With[{bk = bucket, ckK = ckptKey, prK = progKey, sks = shardKeys,
        sl = seqLen, vc = vocab, bs = batchSize, lrHi = lrMax, lrLo = lrMin,
        attn = attentionSrc, nano = nanoSrc},
    TimeConstrained[
      Module[{out = <||>, aws, s3, getB, putB, startDone, mdl, spec, summaries,
              i, sh, shb, str, ids, nwin, windows, data, trainable, trained,
              t0, t1, nb, bl, ckbytes, pr, thisLR},
        out["t_start"] = DateString[Now, "ISODateTime"];
        out["machine"] = $MachineName;
        Export["/tmp/Attention.wl", attn, "Text"];
        Export["/tmp/NanoChat.wl", nano, "Text"];
        Get["/tmp/Attention.wl"]; Get["/tmp/NanoChat.wl"];

        aws = ServiceConnect["AWS"];
        s3 = ServiceExecute[aws, "GetService", <|"Name" -> "S3"|>];
        getB[key_] := Module[{r},
          r = Quiet @ Check[ServiceExecute[s3, "GetObject",
            <|"Bucket" -> bk, "Key" -> key|>], $Failed];
          If[Head[r] === Dataset, r["Body"], $Failed]];
        putB[key_, ba_] := Quiet @ Check[ServiceExecute[s3, "PutObject",
          <|"Bucket" -> bk, "Key" -> key, "Body" -> ba|>]; True, False];

        (* resume point *)
        pr = getB[prK];
        startDone = If[Head[pr] === ByteArray,
          Lookup[Quiet @ ImportByteArray[pr, "RawJSON"], "nextShard", 0], 0];
        out["start_done"] = startDone;

        (* model: load checkpoint if present else fresh *)
        ckbytes = getB[ckK];
        If[Head[ckbytes] === ByteArray,
          str = OpenWrite["/tmp/ckpt_in.wlnet", BinaryFormat -> True];
          BinaryWrite[str, ckbytes]; Close[str];
          mdl = Import["/tmp/ckpt_in.wlnet", "WLNet"]; out["resumed"] = True,
          spec = LLMAtoZ`NanoChat`nanoModelSpec30M[vc, sl];
          mdl = LLMAtoZ`NanoChat`nanoModel[spec["dModel"], spec["nHeads"],
            spec["nLayers"], spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]];
          out["resumed"] = False];

        summaries = {};
        Do[
          sh = sks[[i]];
          shb = getB[sh];
          If[Head[shb] =!= ByteArray,
            AppendTo[summaries, <|"shard" -> i, "error" -> "download"|>]; Continue[]];
          str = OpenWrite["/tmp/shard.bin", BinaryFormat -> True];
          BinaryWrite[str, shb]; Close[str];
          ids = BinaryReadList["/tmp/shard.bin", "UnsignedInteger16"];
          nwin = Quotient[Length[ids], sl + 1];
          windows = Partition[Take[ids, nwin*(sl + 1)], sl + 1];
          data = Map[<|"Input" -> #[[;; -2]], "Output" -> #[[2 ;;]]|> &, windows];

          trainable = NetGraph[
            <|"net" -> mdl, "sm" -> NetMapOperator[SoftmaxLayer[]],
              "loss" -> CrossEntropyLossLayer["Index"]|>,
            {NetPort["Input"] -> "net" -> "sm" -> NetPort["loss", "Input"],
             NetPort["Output"] -> NetPort["loss", "Target"]}];

          (* global cosine LR keyed to shard index (v3 fix for v2 plateau) *)
          thisLR = If[Length[sks] <= 1, lrHi,
            lrLo + 0.5 (lrHi - lrLo) (1 + Cos[Pi (i - 1)/(Length[sks] - 1)])];

          t0 = AbsoluteTime[];
          trained = NetTrain[trainable, data, All, BatchSize -> bs,
            MaxTrainingRounds -> 1, TargetDevice -> "GPU",
            LearningRate -> thisLR, Method -> "ADAM",
            TrainingProgressReporting -> None];
          t1 = AbsoluteTime[];
          mdl = NetExtract[trained["TrainedNet"], "net"];

          (* checkpoint + progress to S3 between shards *)
          Export["/tmp/ckpt_out.wlnet", mdl, "WLNet"];
          putB[ckK, ReadByteArray["/tmp/ckpt_out.wlnet"]];
          putB[prK, StringToByteArray[ExportString[<|"nextShard" -> i|>, "JSON"]]];

          nb = Length[trained["BatchLossList"]]; bl = trained["BatchLossList"];
          AppendTo[summaries, <|"shard" -> i, "n_batches" -> nb, "lr" -> N[thisLR],
            "loss_first" -> First[bl], "loss_tail" -> N[Mean[Take[bl, -Min[100, nb]]]],
            "tok_s" -> N[nb*bs*sl/(t1 - t0)], "wall_s" -> N[t1 - t0]|>];

          (* free memory before next shard (training-gotchas #6) *)
          ids =.; windows =.; data =.; trained =.; trainable =.; shb =.;
          ClearSystemCache[]; Share[];
          , {i, startDone + 1, Length[sks]}];

        out["summaries"] = summaries;
        out["done"] = Length[sks];
        out["t_end"] = DateString[Now, "ISODateTime"];
        out],
      46800,  (* 13h hard cap *)
      <|"fatal" -> "TimeConstrained"|>]]];

Print["UUID:   ", job["JobUUID"]];
Print["Status: ", job["JobStatus"]];
Export["/tmp/pretrain_v3_uuid.txt", job["JobUUID"], "Text"];
Print["Saved UUID to /tmp/pretrain_v3_uuid.txt"];
Print["Monitor: aws batch describe-jobs --region eu-central-1 --jobs <jobId> (auto-retries on spot)"];
