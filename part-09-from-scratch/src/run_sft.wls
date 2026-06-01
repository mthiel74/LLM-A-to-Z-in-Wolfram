#!/usr/bin/env wolframscript
(* ============================================================
   run_sft.wls  ---  Part 9 supervised fine-tuning (instruction tuning)
   ============================================================

   Same self-looping, spot-resilient harness as run_pretraining_v3.wls, but:
     * starts from a PRETRAINED base checkpoint (baseKey in S3) instead of
       random init -- the SFT checkpoint (ckptKey) only exists once an attempt
       has run, so on a fresh start we load the base;
     * trains on Alpaca instruction shards (uint16, same format), each listed
       multiple times in the manifest to get multiple epochs;
     * uses a low LR (SFT, not pretraining) from a gentle cosine schedule.

   The base model and the Alpaca data both use the GPT-2 BPE / vocab 50257, and
   each Alpaca example ends in the EOT token (50256 -> WL id 50257) so the model
   learns to stop -- see prepare_alpaca.py.

   Usage:  wolframscript -file run_sft.wls sft_manifest.json
   Manifest adds: "baseKey" (S3 key of the pretrained .wlnet). *)

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
baseKey   = cfg["baseKey"];
seqLen    = Lookup[cfg, "seqLen", 512];
vocab     = Lookup[cfg, "vocab", 50257];
batchSize = Lookup[cfg, "batchSize", 8];
freshStart = Lookup[cfg, "freshStart", True];
lrMax     = Lookup[cfg, "lrMax", 0.0002];
lrMin     = Lookup[cfg, "lrMin", 0.00002];

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
attentionSrc = Import[FileNameJoin[{repo, "part-02-self-attention", "src", "Attention.wl"}], "Text"];
nanoSrc = Import[FileNameJoin[{repo, "part-09-from-scratch", "src", "NanoChat.wl"}], "Text"];

Print["SFT run : ", runPrefix, "  base: ", baseKey, "  shards(epochs): ", Length[shardKeys]];

awsService = ServiceConnect["AWS"];
If[Head[awsService] =!= ServiceObject, Print["AWS connect failed"]; Quit[1]];
s3o = ServiceExecute[awsService, "GetService", <|"Name" -> "S3"|>];

If[freshStart,
  Quiet @ Check[ServiceExecute[s3o, "DeleteObject", <|"Bucket" -> bucket, "Key" -> ckptKey|>], Null];
  ServiceExecute[s3o, "PutObject", <|"Bucket" -> bucket, "Key" -> progKey,
    "Body" -> StringToByteArray[ExportString[<|"nextShard" -> 0|>, "JSON"]]|>];
  Print["Fresh SFT start: will load base ", baseKey]];

env = RemoteBatchSubmissionEnvironment["AWSBatch", <|
  "JobQueue" -> "arn:aws:batch:REGION:ACCOUNT_ID:job-queue/MY-SPOT-QUEUE",
  "JobDefinition" -> "arn:aws:batch:REGION:ACCOUNT_ID:job-definition/MY-GPU-JOBDEF:2",
  "IOBucket" -> bucket, "ServiceObject" -> awsService|>];

Print["Submitting SFT job..."];

job = RemoteBatchSubmit[env,
  With[{bk = bucket, ckK = ckptKey, prK = progKey, sks = shardKeys, baseK = baseKey,
        sl = seqLen, vc = vocab, bs = batchSize, lrHi = lrMax, lrLo = lrMin,
        attn = attentionSrc, nano = nanoSrc},
    TimeConstrained[
      Module[{out = <||>, aws, s3, getB, putB, startDone, mdl, summaries,
              i, sh, shb, str, ids, nwin, windows, data, trainable, trained,
              t0, t1, nb, bl, ckbytes, basebytes, pr, thisLR},
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

        pr = getB[prK];
        startDone = If[Head[pr] === ByteArray,
          Lookup[Quiet @ ImportByteArray[pr, "RawJSON"], "nextShard", 0], 0];
        out["start_done"] = startDone;

        (* model: SFT checkpoint if a prior attempt exists, else the base *)
        ckbytes = getB[ckK];
        If[Head[ckbytes] === ByteArray,
          str = OpenWrite["/tmp/ckpt_in.wlnet", BinaryFormat -> True];
          BinaryWrite[str, ckbytes]; Close[str];
          mdl = Import["/tmp/ckpt_in.wlnet", "WLNet"]; out["from"] = "sft-ckpt",
          (* load pretrained base *)
          basebytes = getB[baseK];
          If[Head[basebytes] =!= ByteArray,
            out["fatal"] = "base download failed"; Return[out]];
          str = OpenWrite["/tmp/base.wlnet", BinaryFormat -> True];
          BinaryWrite[str, basebytes]; Close[str];
          mdl = Import["/tmp/base.wlnet", "WLNet"]; out["from"] = "base"];
        out["paramCount"] = Information[mdl, "ArraysTotalElementCount"];

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

          thisLR = If[Length[sks] <= 1, lrHi,
            lrLo + 0.5 (lrHi - lrLo) (1 + Cos[Pi (i - 1)/(Length[sks] - 1)])];

          t0 = AbsoluteTime[];
          trained = NetTrain[trainable, data, All, BatchSize -> bs,
            MaxTrainingRounds -> 1, TargetDevice -> "GPU",
            LearningRate -> thisLR, Method -> "ADAM",
            TrainingProgressReporting -> None];
          t1 = AbsoluteTime[];
          mdl = NetExtract[trained["TrainedNet"], "net"];

          Export["/tmp/ckpt_out.wlnet", mdl, "WLNet"];
          putB[ckK, ReadByteArray["/tmp/ckpt_out.wlnet"]];
          putB[prK, StringToByteArray[ExportString[<|"nextShard" -> i|>, "JSON"]]];

          nb = Length[trained["BatchLossList"]]; bl = trained["BatchLossList"];
          AppendTo[summaries, <|"shard" -> i, "n_batches" -> nb, "lr" -> N[thisLR],
            "loss_first" -> First[bl], "loss_tail" -> N[Mean[Take[bl, -Min[100, nb]]]],
            "tok_s" -> N[nb*bs*sl/(t1 - t0)], "wall_s" -> N[t1 - t0]|>];

          ids =.; windows =.; data =.; trained =.; trainable =.; shb =.;
          ClearSystemCache[]; Share[];
          , {i, startDone + 1, Length[sks]}];

        out["summaries"] = summaries;
        out["done"] = Length[sks];
        out["t_end"] = DateString[Now, "ISODateTime"];
        out],
      46800,
      <|"fatal" -> "TimeConstrained"|>]]];

Print["UUID:   ", job["JobUUID"]];
Print["Status: ", job["JobStatus"]];
Export["/tmp/sft_uuid.txt", job["JobUUID"], "Text"];
Print["Saved UUID to /tmp/sft_uuid.txt"];
