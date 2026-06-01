(* ::Package:: *)

(* ============================================================
   Classification.wl  ---  Part 6, Section: classification FT
   ============================================================

   Raschka chapter 6: take the pretrained language model and fine-
   tune it for a classification task by replacing the vocab-head
   (output to |V| logits per position) with a small head that
   produces a fixed number of class logits per example.

   For sequence classification, the canonical pattern is to:
     1. Strip the per-position vocab projection.
     2. Take the hidden state at the LAST position (the position
        whose context spans the entire input).
     3. Project that vector to nClasses logits via one new linear
        layer.

   The whole model becomes a NetChain ending in a (nClasses)-
   shaped logit vector, ready to train with CrossEntropyLossLayer
   on integer class labels.

   In Wolfram terms: we wrap the model's pre-head layers in a
   NetChain that ends with SequenceLastLayer followed by a fresh
   LinearLayer[nClasses].

   Demo task: classify Shakespeare excerpts as "tragedy" vs
   "comedy" using the line-level metadata in Tiny Shakespeare's
   header tags.
*)

BeginPackage["LLMAtoZ`Classification`",
  {"LLMAtoZ`Tokeniser`", "LLMAtoZ`Transformer`", "LLMAtoZ`Training`"}];

ClearAll[
  attachClassifier, trainClassifier, classifySample,
  shakespeareGenreLabels, prepareClassificationDataset
];

attachClassifier::usage =
  "attachClassifier[pretrainedNet, nClasses, dModel] returns a new \
NetChain: the pretrained-model's layers up to but not including the \
vocab head, then SequenceLastLayer, then LinearLayer[nClasses].  Use \
this as the starting point for classification fine-tuning.";

trainClassifier::usage =
  "trainClassifier[net, dataset, hp] runs NetTrain over a classification \
dataset of <|\"Input\" -> tokenIds, \"Output\" -> classIndex|> rows.";

classifySample::usage =
  "classifySample[net, tokenIds] returns the predicted class index and \
the per-class probabilities.";

shakespeareGenreLabels::usage =
  "shakespeareGenreLabels[] returns a list of {sample text, class label} \
pairs from a handful of canonical Shakespeare plays.  Labels: \
1 = tragedy, 2 = comedy.  Just enough to demo classification FT.";

prepareClassificationDataset::usage =
  "prepareClassificationDataset[pairs, mergeRules, tokId, contextLen] \
turns {text, label} pairs into the <|\"Input\" -> ids, \"Output\" -> \
label|> rows trainClassifier expects.  Texts longer than contextLen \
are truncated; shorter ones are padded with token id 1.";

Begin["`Private`"];

(* Replace the LM head with a sequence-last + linear classification
   head.  The pretrained net is assumed to be a NetChain whose
   last layer is the vocab projection (typically a NetMapOperator
   wrapping a LinearLayer).  We drop the last layer and replace it
   with our classification head. *)
attachClassifier[pretrainedNet_, nClasses_Integer, dModel_Integer] := Module[
  {prehead, classifier},
  prehead = NetTake[pretrainedNet, Length[pretrainedNet] - 1];
  classifier = NetChain[{
    prehead,
    SequenceLastLayer[],
    LinearLayer[nClasses]
  }];
  NetInitialize[classifier]
];

trainClassifier[net_, dataset_List, hp_Association : Automatic] := Module[
  {hyp, trained, trainStart, trainEnd},
  hyp = Join[LLMAtoZ`Training`defaultHyperparameters[],
             <|"BaseLR" -> 0.0005, "MaxRounds" -> 8|>,
             If[AssociationQ[hp], hp, <||>]];
  SeedRandom[hyp["RandomSeed"]];
  trainStart = AbsoluteTime[];
  trained = NetTrain[
    net, dataset, All,
    BatchSize -> hyp["BatchSize"],
    LearningRate -> hyp["BaseLR"],
    MaxTrainingRounds -> hyp["MaxRounds"],
    LossFunction -> CrossEntropyLossLayer["Index"],
    TargetDevice -> "CPU",
    Method -> hyp["OptimizerType"],
    TrainingProgressReporting -> None
  ];
  trainEnd = AbsoluteTime[];
  If[trained === $Failed,
    Return[<|"Net" -> $Failed, "WallSeconds" -> (trainEnd - trainStart)|>]
  ];
  <|
    "Net" -> trained["TrainedNet"],
    "LossHistory" -> trained["RoundLossList"],
    "BatchLossHistory" -> trained["BatchLossList"],
    "WallSeconds" -> (trainEnd - trainStart),
    "FinalLoss" -> trained["FinalRoundLoss"]
  |>
];

classifySample[net_, tokenIds_List] := Module[{logits, probs},
  logits = net[tokenIds];
  probs = With[{shifted = logits - Max[logits]},
    Exp[shifted] / Total[Exp[shifted]]];
  <|"class" -> First[Ordering[probs, -1]],
    "probabilities" -> probs|>
];

(* A hand-curated 20-sample tragedy/comedy dataset.  Each entry is a
   short, recognisable line of Shakespeare; labels are 1 for tragedy
   and 2 for comedy. *)
shakespeareGenreLabels[] := {
  {"To be, or not to be, that is the question.", 1},
  {"Out, damned spot! out, I say!", 1},
  {"All the world's a stage, and all the men and women merely players.", 2},
  {"What light through yonder window breaks?", 1},
  {"If music be the food of love, play on.", 2},
  {"The lady doth protest too much, methinks.", 1},
  {"O, beware, my lord, of jealousy!", 1},
  {"Lord, what fools these mortals be!", 2},
  {"Some are born great, some achieve greatness.", 2},
  {"To die, to sleep \[LongDash] no more.", 1},
  {"A horse, a horse! my kingdom for a horse!", 1},
  {"Friends, Romans, countrymen, lend me your ears.", 1},
  {"Brevity is the soul of wit.", 2},
  {"Now is the winter of our discontent.", 1},
  {"My salad days, when I was green in judgement.", 2},
  {"Cry havoc! and let slip the dogs of war.", 1},
  {"Get thee to a nunnery.", 1},
  {"Journeys end in lovers meeting.", 2},
  {"Parting is such sweet sorrow.", 1},
  {"As merry as the day is long.", 2}
};

prepareClassificationDataset[pairs_List, mergeRules_List,
    tokId_Association, contextLen_Integer] := Module[{rows, vocabMax},
  vocabMax = Max[Values[tokId]];
  rows = Map[
    Function[pair,
      Module[{toks, ids},
        toks = LLMAtoZ`Tokeniser`bpeEncode[pair[[1]], mergeRules];
        (* Drop tokens absent from the vocabulary; safer than letting
           Missing reach NetTrain. *)
        toks = Select[toks, KeyExistsQ[tokId, #] &];
        ids = Lookup[tokId, toks];
        ids = Select[ids, IntegerQ[#] && 1 <= # <= vocabMax &];
        ids = If[Length[ids] >= contextLen,
                  ids[[;; contextLen]],
                  PadRight[ids, contextLen, 1]];
        <|"Input" -> ids, "Output" -> pair[[2]]|>
      ]
    ],
    pairs
  ];
  rows
];

End[];
EndPackage[];
