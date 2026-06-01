#!/usr/bin/env wolframscript
(* ============================================================
   run_pretraining.wls  ---  Part 9 resumable chunked pretraining
   ============================================================

   Orchestrates a multi-job pretraining run on AWS Batch spot.  Data is
   pre-sharded in S3; each Batch job ("chunk") pulls one shard + the latest
   checkpoint from S3 (WL-native -- the image has no aws CLI, see [[wl-s3-api]]),
   trains a time-boxed pass, writes the updated checkpoint back to S3, and
   returns a small loss/throughput summary.  The Mac-side loop submits chunks
   shard-by-shard, polling AWS Batch directly (reliable; the Wolfram JobStatus
   poller is flaky), and resubmits a shard if its job is spot-killed.

   Config is read from a JSON manifest passed as the first argument:
     { "runPrefix": "runs/pretrain-30M-v1",
       "shardKeys": ["data/shards/s00.bin", ...],
       "trainSecsPerChunk": 6000,
       "freshStart": true }

   Usage:  wolframscript -file run_pretraining.wls /path/to/manifest.json
   Designed to run under  caffeinate -i nohup ...  for the duration. *)

(* ---------- read manifest ---------- *)
manifestPath = First[$ScriptCommandLine[[2 ;;]], $Failed];
If[manifestPath === $Failed || ! FileExistsQ[manifestPath],
  Print["ERROR: pass a manifest JSON path"]; Quit[1]];
cfg = Import[manifestPath, "RawJSON"];

bucket    = "EXAMPLE-BUCKET";
region    = "eu-central-1";
runPrefix = cfg["runPrefix"];
ckptKey   = runPrefix <> "/ckpt.wlnet";
shardKeys = cfg["shardKeys"];
trainSecs = Lookup[cfg, "trainSecsPerChunk", 6000];
seqLen    = Lookup[cfg, "seqLen", 512];
vocab     = Lookup[cfg, "vocab", 50257];
batchSize = Lookup[cfg, "batchSize", 8];
freshStart = Lookup[cfg, "freshStart", True];

queueArn = "arn:aws:batch:REGION:ACCOUNT_ID:job-queue/MY-SPOT-QUEUE";
jobDefArn = "arn:aws:batch:REGION:ACCOUNT_ID:job-definition/MY-GPU-JOBDEF:1";

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
attentionSrc = Import[FileNameJoin[{repo, "part-02-self-attention", "src", "Attention.wl"}], "Text"];
nanoSrc = Import[FileNameJoin[{repo, "part-09-from-scratch", "src", "NanoChat.wl"}], "Text"];

Print["Run prefix : ", runPrefix];
Print["Shards     : ", Length[shardKeys]];
Print["TimeGoal/chunk: ", trainSecs, "s   seqLen ", seqLen, "  batch ", batchSize];

awsService = ServiceConnect["AWS"];
If[Head[awsService] =!= ServiceObject, Print["AWS connect failed"]; Quit[1]];
env = RemoteBatchSubmissionEnvironment["AWSBatch", <|
  "JobQueue" -> queueArn, "JobDefinition" -> jobDefArn,
  "IOBucket" -> bucket, "ServiceObject" -> awsService|>];

(* ---------- orchestrator-side S3 + progress persistence ----------
   Progress (next shard index) is mirrored to S3 so a dead local kernel
   (e.g. a long-run Wolfram license hiccup) can be restarted with
   freshStart:false and pick up exactly where it left off -- the model
   itself is always safe in S3 after each chunk. *)
s3o = ServiceExecute[awsService, "GetService", <|"Name" -> "S3"|>];
progressKey = runPrefix <> "/progress.json";

readProgress[] := Module[{r},
  r = Quiet @ Check[
    ServiceExecute[s3o, "GetObject", <|"Bucket" -> bucket, "Key" -> progressKey|>],
    $Failed];
  If[Head[r] === Dataset, Quiet @ ImportByteArray[r["Body"], "RawJSON"], $Failed]];

writeProgress[nextIdx_] := Quiet @ Check[
  ServiceExecute[s3o, "PutObject", <|"Bucket" -> bucket, "Key" -> progressKey,
    "Body" -> StringToByteArray[ExportString[
      <|"nextShard" -> nextIdx, "updated" -> DateString["ISODateTime"]|>,
      "JSON"]]|>], $Failed];

startIdx = 1;
If[freshStart,
  Quiet @ Check[ServiceExecute[s3o, "DeleteObject",
    <|"Bucket" -> bucket, "Key" -> ckptKey|>], Null];
  writeProgress[1];
  Print["Fresh start: cleared checkpoint, progress reset"],
  (* resume *)
  Module[{p = readProgress[]},
    startIdx = If[AssociationQ[p] && IntegerQ[p["nextShard"]], p["nextShard"], 1]];
  Print["Resuming from shard index ", startIdx]
];

(* ============================================================
   The in-job chunk trainer.  Injected literals via With so the
   expression is self-contained on the remote kernel. *)

submitChunk[shardKey_String, idx_Integer] := RemoteBatchSubmit[env,
  With[{sk = shardKey, ck = ckptKey, bk = bucket, sl = seqLen, vc = vocab,
        bs = batchSize, ts = trainSecs, attn = attentionSrc, nano = nanoSrc,
        ix = idx},
    TimeConstrained[
      Module[{out = <||>, aws, s3, getBytes, putBytes, shardBA, ids, tmp, str,
              ckBytes, mdl, spec, nwin, windows, data, trainable, t0, t1,
              trained, resumed, nb},
        out["chunk"] = ix; out["shardKey"] = sk;
        out["t_start"] = DateString[Now, "ISODateTime"];
        out["machine"] = $MachineName;

        Export["/tmp/Attention.wl", attn, "Text"];
        Export["/tmp/NanoChat.wl", nano, "Text"];
        Get["/tmp/Attention.wl"]; Get["/tmp/NanoChat.wl"];

        (* ---- WL-native S3 helpers (no aws CLI in image) ---- *)
        aws = ServiceConnect["AWS"];
        s3 = ServiceExecute[aws, "GetService", <|"Name" -> "S3"|>];
        getBytes[key_] := Module[{r},
          r = Quiet @ Check[
            ServiceExecute[s3, "GetObject", <|"Bucket" -> bk, "Key" -> key|>],
            $Failed];
          If[Head[r] === Dataset, r["Body"], $Failed]];
        putBytes[key_, ba_] :=
          ServiceExecute[s3, "PutObject",
            <|"Bucket" -> bk, "Key" -> key, "Body" -> ba|>];

        (* ---- pull this shard ---- *)
        shardBA = getBytes[sk];
        If[Head[shardBA] =!= ByteArray,
          out["fatal"] = "shard download failed"; Return[out]];
        str = OpenWrite["/tmp/shard.bin", BinaryFormat -> True];
        BinaryWrite[str, shardBA]; Close[str];
        ids = BinaryReadList["/tmp/shard.bin", "UnsignedInteger16"];
        out["n_tokens"] = Length[ids];

        (* ---- load checkpoint if present, else fresh model ---- *)
        ckBytes = getBytes[ck];
        If[Head[ckBytes] === ByteArray,
          str = OpenWrite["/tmp/ckpt_in.wlnet", BinaryFormat -> True];
          BinaryWrite[str, ckBytes]; Close[str];
          mdl = Import["/tmp/ckpt_in.wlnet", "WLNet"];
          resumed = True,
          (* else *)
          spec = LLMAtoZ`NanoChat`nanoModelSpec30M[vc, sl];
          mdl = LLMAtoZ`NanoChat`nanoModel[
            spec["dModel"], spec["nHeads"], spec["nLayers"],
            spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]];
          resumed = False];
        out["resumed"] = resumed;
        out["paramCount"] = Information[mdl, "ArraysTotalElementCount"];

        (* ---- window + train (softmax before CE; nanoModel emits logits) ---- *)
        nwin = Quotient[Length[ids], sl + 1];
        windows = Partition[Take[ids, nwin*(sl + 1)], sl + 1];
        data = Map[<|"Input" -> #[[;; -2]], "Output" -> #[[2 ;;]]|> &, windows];
        out["n_windows"] = Length[data];

        trainable = NetGraph[
          <|"net" -> mdl, "sm" -> NetMapOperator[SoftmaxLayer[]],
            "loss" -> CrossEntropyLossLayer["Index"]|>,
          {NetPort["Input"] -> "net" -> "sm" -> NetPort["loss", "Input"],
           NetPort["Output"] -> NetPort["loss", "Target"]}];

        t0 = AbsoluteTime[];
        trained = NetTrain[trainable, data, All,
          BatchSize -> bs, MaxTrainingRounds -> 1, TimeGoal -> ts,
          TargetDevice -> "GPU", TrainingProgressReporting -> None];
        t1 = AbsoluteTime[];
        out["train_wall_s"] = N[t1 - t0];
        out["train_status"] = ToString[Head[trained]];  (* string -> JSON-safe *)

        If[Head[trained] === NetTrainResultsObject,
          nb = Length[trained["BatchLossList"]];
          With[{bl = trained["BatchLossList"]},
            out["n_batches"] = nb;
            out["loss_first"] = First[bl];
            out["loss_min"] = Min[bl];
            out["loss_last"] = Last[bl];
            out["loss_tail_mean"] = N[Mean[Take[bl, -Min[100, nb]]]];
            out["loss_curve"] = bl[[1 ;; ;; Max[1, Quotient[nb, 50]]]]];
          out["throughput_tokens_per_s"] = N[nb*bs*sl/(t1 - t0)];
          (* ---- write checkpoint back to S3 (logit head only) ---- *)
          Export["/tmp/ckpt_out.wlnet",
            NetExtract[trained["TrainedNet"], "net"], "WLNet"];
          out["ckpt_put"] = Quiet @ Check[
            putBytes[ck, ReadByteArray["/tmp/ckpt_out.wlnet"]]; "ok",
            "put-failed"]];
        out["t_end"] = DateString[Now, "ISODateTime"];
        out],
      ts + 900,
      <|"fatal" -> "TimeConstrained", "chunk" -> ix|>]]];

(* ============================================================
   Mac-side: poll a Batch job to terminal via aws CLI. *)

batchStatus[jobId_String] := Module[{r},
  r = RunProcess[{"aws", "batch", "describe-jobs", "--region", region,
    "--jobs", jobId, "--query", "jobs[0].status", "--output", "text"}];
  StringTrim[r["StandardOutput"]]];

findJobId[uuid_String] := Module[{r, out},
  r = RunProcess[{"aws", "batch", "list-jobs", "--region", region,
    "--job-queue", "WolframG4SpotQueue", "--filters",
    "name=JOB_NAME,values=RemoteBatchSubmit-SingleJob-" <> uuid,
    "--query", "jobSummaryList[0].jobId", "--output", "text"}];
  out = StringTrim[r["StandardOutput"]];
  If[out === "" || out === "None", $Failed, out]];

waitForTerminal[jobId_String] := Module[{st = "", t0 = AbsoluteTime[]},
  While[True,
    st = batchStatus[jobId];
    If[MemberQ[{"SUCCEEDED", "FAILED"}, st], Return[st]];
    If[AbsoluteTime[] - t0 > 5*3600, Return["TIMEOUT"]];
    Pause[60]]];

(* Fetch a finished job's result by reading its output WXF straight from S3
   (more reliable than job["EvaluationResult"] for an async-completed job). *)
fetchResult[uuid_String] := Module[{key, r, ba, str, wxf},
  key = "jobs/Job-" <> uuid <> "/RemoteBatchSubmit-Job-Output.wxf";
  r = Quiet @ Check[
    ServiceExecute[s3o, "GetObject", <|"Bucket" -> bucket, "Key" -> key|>],
    $Failed];
  If[Head[r] =!= Dataset, Return[$Failed]];
  ba = r["Body"];
  str = OpenWrite["/tmp/chunk_out.wxf", BinaryFormat -> True];
  BinaryWrite[str, ba]; Close[str];
  wxf = Import["/tmp/chunk_out.wxf"];
  If[AssociationQ[wxf] && KeyExistsQ[wxf, "Result"], wxf["Result"], wxf]];

(* ============================================================
   Main loop: one chunk per shard, retry on spot-kill. *)

results = {};
Do[
  Module[{shardKey = shardKeys[[i]], attempt = 0, st, job, uuid, jid, res, ok = False},
    While[! ok && attempt < 3,
      attempt++;
      Print["[", DateString["ISODateTime"], "] chunk ", i, "/", Length[shardKeys],
        " attempt ", attempt, " shard ", shardKey];
      job = submitChunk[shardKey, i];
      uuid = job["JobUUID"];
      Print["  submitted UUID ", uuid];
      Pause[20];
      jid = findJobId[uuid];
      If[jid === $Failed, Print["  could not find jobId; retrying"]; Continue[]];
      Print["  jobId ", jid, " ; waiting..."];
      st = waitForTerminal[jid];
      Print["  terminal: ", st];
      If[st === "SUCCEEDED",
        res = fetchResult[uuid];
        AppendTo[results, res];
        Print["  loss_last=", Lookup[res, "loss_last", "?"],
          " tail=", Lookup[res, "loss_tail_mean", "?"],
          " tok/s=", Lookup[res, "throughput_tokens_per_s", "?"],
          " ckpt=", Lookup[res, "ckpt_put", "?"]];
        writeProgress[i + 1];
        ok = True,
        Print["  chunk failed (", st, "); resubmitting from checkpoint"]]];
    If[! ok, Print["chunk ", i, " gave up after ", attempt, " attempts"]]],
  {i, startIdx, Length[shardKeys]}];

Print["=== run complete: ", Length[results], "/", Length[shardKeys], " chunks ok ==="];
Export["/tmp/pretrain_results.json",
  <|"runPrefix" -> runPrefix, "chunks" -> results|>, "JSON"];
Print["Summary written to /tmp/pretrain_results.json"];
Print["Final checkpoint: s3://", bucket, "/", ckptKey];
