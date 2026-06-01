(* ::Package:: *)

(* ============================================================
   Tiers.wl  ---  the single-dial compute-budget config
   ============================================================

   Every long-running notebook in the series begins with one line:

       tier = "tinyCPU"     (* or any of the names below *)

   and the rest of the notebook reads the model spec, the dataset
   choice, the training hyperparameters, and whether to use a
   pretrained model or train from scratch, from this file.

   Five named tiers, smallest to largest:

     "tinyCPU"        £0   ~1 minute    1M from-scratch  Tiny Shakespeare
     "smallCPU"       £0   ~30 minutes  5M from-scratch  Tiny Shakespeare
     "cloudCheap"   ~£30   ~1 hour      25M from-scratch FineWeb-Edu 100 MB
     "prebuiltGpt2"   £0   ~5 minutes   GPT-2 124M       (fine-tune only)
     "cloudReal"   £100+   hours        GPT-2 124M       FineWeb-Edu 1 GB

   The "prebuiltGpt2" tier is the most practical: it uses OpenAI's
   pretrained weights via loadPrebuiltGpt2[] as the starting point,
   skips pretraining (Part 4), and runs SFT (Part 6) on top.  The
   final chat model in Part 8 actually produces sensible English.

   The cost figures are rough Wolfram Cloud estimates and depend on
   your subscription level; treat them as orders of magnitude.  The
   public Wolfram Cloud landing page (wolfram.com/cloud) exposes only
   "Basic is free; contact sales for more" — concrete CCU rates,
   GPU options, and multi-hour-job limits are not publicly listed.
   The £30 / £120 numbers in this file are estimates derived from
   the cost-per-CCU one would expect at standard pricing tiers and
   should be replaced with the actual figures once a Cloud
   subscription is in place.
*)

BeginPackage["LLMAtoZ`Tiers`",
  {"LLMAtoZ`Transformer`"}];

ClearAll[
  availableTiers, tier, modelSpec, datasetChoice, trainingBudget,
  useCloudPath, useprebuiltModel, modelConstructor
];

availableTiers::usage =
  "availableTiers[] returns the list of supported tier names.";

tier::usage =
  "tier[name] returns the full Association of settings for the named \
tier: \"modelSource\" (\"fromScratch\" or \"prebuilt\"), \"modelSpec\" \
(spec for our gptModel) or \"netModelName\" (NetModel name), \
\"dataset\" (corpus identifier), \"contextLen\", \"batchSize\", \
\"baseLR\", \"maxRounds\", \"compute\" (\"cpu\" or \"cloud\"), \
\"estCostGBP\" (rough cost in GBP), \"estTimeMinutes\".";

modelSpec::usage =
  "modelSpec[tierName] is shorthand for tier[tierName][\"modelSpec\"].";

datasetChoice::usage =
  "datasetChoice[tierName] returns the corpus identifier for the named \
tier.  Identifiers map to download functions in the package.";

trainingBudget::usage =
  "trainingBudget[tierName] returns <|\"BatchSize\" -> ..., \
\"BaseLR\" -> ..., \"MaxRounds\" -> ..., \"ContextLen\" -> ...|>.";

useCloudPath::usage =
  "useCloudPath[tierName] returns True if the tier should dispatch \
training to the cloud (via submitRemotePretraining), False if it \
should run locally.";

useprebuiltModel::usage =
  "useprebuiltModel[tierName] returns True if the tier starts from \
NetModel's pretrained GPT-2 rather than training from scratch.";

modelConstructor::usage =
  "modelConstructor[tierName] returns a no-argument function that, \
when called, produces the starting net.  Either an empty gptModel \
(\"fromScratch\" tiers) or NetModel GPT-2 (\"prebuilt\" tiers).";

Begin["`Private`"];

availableTiers[] := {"tinyCPU", "smallCPU", "cloudCheap",
                     "prebuiltGpt2", "cloudReal"};

$tierTable = <|
  "tinyCPU" -> <|
    "modelSource" -> "fromScratch",
    "modelSpec" -> <|"label" -> "1M",
      "dModel" -> 128, "nHeads" -> 4, "nLayers" -> 4,
      "dFF" -> 512, "vocabSize" -> 1064, "maxSeqLen" -> 64|>,
    "dataset" -> "TinyShakespeare",
    "contextLen" -> 64, "batchSize" -> 32, "baseLR" -> 0.002,
    "maxRounds" -> 3,
    "compute" -> "cpu", "estCostGBP" -> 0.0,
    "estTimeMinutes" -> 2|>,

  "smallCPU" -> <|
    "modelSource" -> "fromScratch",
    "modelSpec" -> <|"label" -> "5M",
      "dModel" -> 256, "nHeads" -> 8, "nLayers" -> 6,
      "dFF" -> 1024, "vocabSize" -> 1064, "maxSeqLen" -> 256|>,
    "dataset" -> "TinyShakespeare",
    "contextLen" -> 256, "batchSize" -> 32, "baseLR" -> 0.001,
    "maxRounds" -> 10,
    "compute" -> "cpu", "estCostGBP" -> 0.0,
    "estTimeMinutes" -> 30|>,

  "cloudCheap" -> <|
    "modelSource" -> "fromScratch",
    "modelSpec" -> <|"label" -> "25M",
      "dModel" -> 384, "nHeads" -> 12, "nLayers" -> 10,
      "dFF" -> 1536, "vocabSize" -> 1064, "maxSeqLen" -> 256|>,
    "dataset" -> "FineWebEduSmall",
    "contextLen" -> 256, "batchSize" -> 64, "baseLR" -> 6.*^-4,
    "maxRounds" -> 8,
    "compute" -> "cloud", "estCostGBP" -> 30.,
    "estTimeMinutes" -> 60|>,

  "prebuiltGpt2" -> <|
    "modelSource" -> "prebuilt",
    "netModelName" -> "GPT-2 Transformer Trained on WebText Data",
    "modelSpec" -> <|"label" -> "GPT-2 small (prebuilt)",
      "dModel" -> 768, "nHeads" -> 12, "nLayers" -> 12,
      "dFF" -> 3072, "vocabSize" -> 50257, "maxSeqLen" -> 1024|>,
    "dataset" -> "SkipPretraining",
    "contextLen" -> 256, "batchSize" -> 4, "baseLR" -> 5.*^-5,
    "maxRounds" -> 3,
    "compute" -> "cpu", "estCostGBP" -> 0.0,
    "estTimeMinutes" -> 5|>,

  "cloudReal" -> <|
    "modelSource" -> "fromScratch",
    "modelSpec" -> <|"label" -> "GPT-2 (rebuild)",
      "dModel" -> 768, "nHeads" -> 12, "nLayers" -> 12,
      "dFF" -> 3072, "vocabSize" -> 50257, "maxSeqLen" -> 1024|>,
    "dataset" -> "FineWebEduLarge",
    "contextLen" -> 1024, "batchSize" -> 32, "baseLR" -> 6.*^-4,
    "maxRounds" -> 5,
    "compute" -> "cloud", "estCostGBP" -> 120.,
    "estTimeMinutes" -> 6 * 60|>
|>;

tier[name_String] :=
  Lookup[$tierTable, name,
    (Message[tier::badtier, name]; First[Values[$tierTable]])];

tier::badtier = "Unknown tier `1`. Use one of `2`.";

modelSpec[name_String]      := tier[name]["modelSpec"];
datasetChoice[name_String]  := tier[name]["dataset"];
trainingBudget[name_String] := With[{t = tier[name]},
  <|"BatchSize" -> t["batchSize"], "BaseLR" -> t["baseLR"],
    "MaxRounds" -> t["maxRounds"], "ContextLen" -> t["contextLen"]|>];
useCloudPath[name_String]    := tier[name]["compute"] === "cloud";
useprebuiltModel[name_String] := tier[name]["modelSource"] === "prebuilt";

modelConstructor[name_String] := Module[{t = tier[name]},
  If[t["modelSource"] === "prebuilt",
    (NetModel[t["netModelName"],
              "Task" -> "LanguageModeling"]) &,
    (LLMAtoZ`Transformer`gptModel[
        t["modelSpec"]["dModel"], t["modelSpec"]["nHeads"],
        t["modelSpec"]["nLayers"], t["modelSpec"]["dFF"],
        t["modelSpec"]["vocabSize"], t["modelSpec"]["maxSeqLen"]]) &
  ]
];

End[];
EndPackage[];
