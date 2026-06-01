(* ::Package:: *)

(* ============================================================
   Pretraining.wl  ---  Part 9 of "LLM A to Z in Wolfram Language"
   ============================================================

   Chunked-pretraining harness for the nano model on AWS Batch.
   Two responsibilities:

     1. In-job side: a TrainingProgressReporting callback that
        periodically saves (i) a JSON loss-curve snapshot and
        (ii) a full network checkpoint to S3, so an OOM near the
        end of training does not destroy hours of progress.

     2. Orchestrator side: helpers to fetch the latest loss curve
        from S3, decide whether continued training is paying off,
        and submit a resume job from the most recent checkpoint.

   Cost decision philosophy: NetTrain runs are atomic -- once
   started they consume their full time budget.  The "should I
   spend more?" decision is therefore between successive jobs,
   not inside one job.  We default to time-budgeted jobs of
   ~4 hours (~$1.20 of g4dn.xlarge spot) so the cost granularity
   of a continue/stop decision is small.

   No external dependencies.  S3 access uses the AWS CLI via
   RunProcess; on AWS Batch the EC2 instance profile provides
   credentials automatically. *)

BeginPackage["LLMAtoZ`Pretraining`"];

ClearAll[
  s3Upload, s3Download,
  pretrainingCallback,
  saveCheckpoint, loadCheckpoint,
  lossCurveSummary, decideToContinue,
  estimateJobCost
];

s3Upload::usage =
  "s3Upload[localPath, s3Uri] uploads localPath to s3Uri using the \
AWS CLI.  Assumes aws is on PATH and credentials are available via \
either ~/.aws/credentials or an EC2 instance profile.  Returns the \
RunProcess result (an Association with \"ExitCode\", \"StandardOutput\", \
\"StandardError\").";

s3Download::usage =
  "s3Download[s3Uri, localPath] downloads s3Uri to localPath.  Same \
contract as s3Upload.";

pretrainingCallback::usage =
  "pretrainingCallback[<|\"S3Prefix\"->\"s3://bucket/run42\", \
\"EveryBatches\"->100, \"NetExtractor\"->fn|>] returns a Function \
suitable for TrainingProgressReporting.  Every EveryBatches batches \
it writes (i) a JSON loss-curve snapshot to <prefix>/loss.json and \
(ii) a NetSave-format checkpoint to <prefix>/ckpt.wlnet.  NetExtractor \
is an optional Function applied to the in-flight net before saving \
(usually NetTake to strip off the wrapping CrossEntropyLossLayer).";

saveCheckpoint::usage =
  "saveCheckpoint[net, s3Uri] writes net to a temp file with \
Export[..., \"WLNet\"] and uploads to s3Uri.  Returns the s3Uri on \
success, $Failed otherwise.";

loadCheckpoint::usage =
  "loadCheckpoint[s3Uri] downloads s3Uri to a temp file and Imports \
as WLNet.  Returns the loaded net or $Failed.";

lossCurveSummary::usage =
  "lossCurveSummary[lossList] returns an Association with summary \
statistics suitable for the orchestrator: total batches seen, latest \
loss, slope of the last 100 losses (per batch), and a coarse \
plateau-detection flag based on the absolute slope.";

decideToContinue::usage =
  "decideToContinue[<|\"LossHistory\"->..., \"CostSpent\"->..., \
\"BudgetUSD\"->100, \"MinSlope\"->5*10^-6|>] returns \"Continue\" or \
\"Stop\" with a reason string.  The decision rule: stop if budget \
exhausted; stop if loss slope is shallower than MinSlope (plateau); \
otherwise continue.";

estimateJobCost::usage =
  "estimateJobCost[wallTimeHours, <|\"USDPerHour\"->0.30|>] returns \
the dollar cost of a job at the given hourly rate.  Default rate \
matches g4dn.xlarge spot in eu-central-1 as of 2026-05.";

Begin["`Private`"];

(* ------------------------------------------------------------
   S3 helpers via the AWS CLI. *)

s3Upload[localPath_String, s3Uri_String] :=
  RunProcess[{"aws", "s3", "cp", localPath, s3Uri}];

s3Download[s3Uri_String, localPath_String] :=
  RunProcess[{"aws", "s3", "cp", s3Uri, localPath}];

(* ------------------------------------------------------------
   Checkpoint save / load.

   We use WL's native net serialisation format (.wlnet) rather
   than MX because .wlnet is portable across versions and runs
   that re-load on a different machine class. *)

saveCheckpoint[net_, s3Uri_String] := Module[
  {tmp = CreateFile[CreateUUID[] <> ".wlnet"], res},
  Export[tmp, net, "WLNet"];
  res = s3Upload[tmp, s3Uri];
  Quiet @ DeleteFile[tmp];
  If[res["ExitCode"] === 0, s3Uri, $Failed]
];

loadCheckpoint[s3Uri_String] := Module[
  {tmp = FileNameJoin[{$TemporaryDirectory, CreateUUID[] <> ".wlnet"}], res},
  res = s3Download[s3Uri, tmp];
  If[res["ExitCode"] =!= 0, Return[$Failed]];
  Import[tmp, "WLNet"]
];

(* ------------------------------------------------------------
   In-training callback.

   NetTrain calls TrainingProgressReporting with an Association
   on every batch.  We branch on the "Batch" counter and only act
   every EveryBatches.  Writing happens on the training device's
   main kernel thread, so it must be cheap; we save the loss
   curve every EveryBatches and the full network at a slower
   cadence (every 10 * EveryBatches) since the latter costs MB
   per save. *)

Options[pretrainingCallback] = {
  "S3Prefix"      -> None,
  "EveryBatches"  -> 100,
  "NetExtractor"  -> Identity,
  "CheckpointEveryNCallbacks" -> 10
};

pretrainingCallback[opts : OptionsPattern[]] := Module[
  {prefix, every, extract, ckptEvery, counter = 0, lossBuf = {}},
  prefix     = OptionValue["S3Prefix"];
  every      = OptionValue["EveryBatches"];
  extract    = OptionValue["NetExtractor"];
  ckptEvery  = OptionValue["CheckpointEveryNCallbacks"];
  Function[info,
    AppendTo[lossBuf, <|"batch" -> info["Batch"], "loss" -> info["BatchLoss"]|>];
    If[Mod[info["Batch"], every] === 0 && prefix =!= None,
      counter += 1;
      writeLossSnapshotToS3[prefix, lossBuf];
      If[Mod[counter, ckptEvery] === 0,
        writeCheckpointToS3[prefix, info["Net"], extract]
      ]
    ]
  ]
];

writeLossSnapshotToS3[prefix_String, lossBuf_List] := Module[
  {tmp = FileNameJoin[{$TemporaryDirectory, "loss.json"}]},
  Export[tmp, <|
    "savedAt"   -> DateString[Now, "ISODateTime"],
    "nBatches"  -> Length[lossBuf],
    "losses"    -> lossBuf
  |>, "JSON"];
  s3Upload[tmp, prefix <> "/loss.json"];
];

writeCheckpointToS3[prefix_String, net_, extract_] := Module[
  {netToSave = extract[net]},
  saveCheckpoint[netToSave, prefix <> "/ckpt.wlnet"]
];

(* ------------------------------------------------------------
   Orchestrator-side helpers.

   These run on the user's Mac.  They fetch the loss snapshot
   from S3, summarise the trajectory, and produce a continue/
   stop decision. *)

lossCurveSummary[losses_List] := Module[
  {n, vals, tail, fit, slope},
  n = Length[losses];
  vals = #["loss"] & /@ losses;
  tail = Take[vals, -Min[100, n]];
  (* Linear fit: loss as a function of batch index within the tail. *)
  fit = If[Length[tail] >= 5,
    LinearModelFit[
      Transpose[{Range[Length[tail]], tail}], x, x],
    Missing[]];
  slope = If[fit === Missing[], 0., fit["BestFitParameters"][[2]]];
  <|
    "nBatches"   -> n,
    "latestLoss" -> If[vals === {}, Missing[], Last[vals]],
    "tailSlope"  -> slope,
    "plateau"    -> Abs[slope] < 5*10^-6
  |>
];

(* Decision rule:
     If CostSpent >= BudgetUSD: stop ("budget exhausted").
     If summary["plateau"]:     stop ("loss has plateaued").
     Else:                      continue.

   Both gates are conservative: better to over-spend a little
   on a still-improving model than to stop too early. *)

decideToContinue[opts : KeyValuePattern[{}]] := Module[
  {history, cost, budget, summary, reason},
  history = Lookup[opts, "LossHistory", {}];
  cost    = Lookup[opts, "CostSpent", 0.];
  budget  = Lookup[opts, "BudgetUSD", 100.];
  summary = lossCurveSummary[history];
  Which[
    cost >= budget,
      <|"decision" -> "Stop",
        "reason"   -> StringTemplate[
          "Spent $`1` of $`2` budget."][NumberForm[cost, {6, 2}],
            NumberForm[budget, {6, 2}]],
        "summary"  -> summary|>,
    summary["plateau"],
      <|"decision" -> "Stop",
        "reason"   -> StringTemplate[
          "Loss slope `1` is below the plateau threshold."][
          NumberForm[summary["tailSlope"], {6, 4}]],
        "summary"  -> summary|>,
    True,
      <|"decision" -> "Continue",
        "reason"   -> StringTemplate[
          "Slope still `1`/batch, $`2` of $`3` budget left."][
          NumberForm[summary["tailSlope"], {6, 4}],
          NumberForm[budget - cost, {6, 2}],
          NumberForm[budget, {6, 2}]],
        "summary"  -> summary|>
  ]
];

(* ------------------------------------------------------------
   Cost estimate.  Defaults to g4dn.xlarge spot in eu-central-1
   (about $0.30 / hour as of 2026-05; on-demand is ~$0.80/hour).
   Wolfram OnDemand adds a 30% overhead on RemoteBatchSubmit,
   but that is accounted for outside this helper -- here we just
   multiply by hours times dollars-per-hour. *)

Options[estimateJobCost] = {"USDPerHour" -> 0.30};

estimateJobCost[wallTimeHours_, opts : OptionsPattern[]] :=
  N[wallTimeHours * OptionValue["USDPerHour"]];

End[];
EndPackage[];
