#!/usr/bin/env wolframscript
(* Post-training evaluation.

   Loads the saved char-level checkpoint, computes:
     - per-position cross-entropy on the 33k held-out chars
     - the generalisation gap (eval / train)
     - sample generations at 3 temperatures from 3 prompts
     - a brief verdict ("keep training" vs "overfit, stop").

   Writes everything to data/charlevel_eval_summary.json so we can
   look at it and decide whether to extend. *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Quiet @ Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Quiet @ Get["../../part-02-self-attention/src/Attention.wl"];
Quiet @ Get["../../part-03-transformer-block/src/Transformer.wl"];
Quiet @ Get["CharTokeniser.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

ckptPath = FileNameJoin[{dataDir, "charlevel_shakespeare_trained.wlnet"}];
histPath = FileNameJoin[{dataDir, "charlevel_training_history.wxf"}];

If[!FileExistsQ[ckptPath],
  Print["No checkpoint at ", ckptPath]; Exit[1]];

net  = Import[ckptPath];
hist = Import[histPath];

(* History file has two possible schemas: the per-round in-progress
   form written every 5 min during training, and the final form
   written when training completes.  Normalise. *)
If[!KeyExistsQ[hist, "currentLoss"],
  hist = Join[hist, <|
    "currentLoss"      -> Last[hist["lossHistory"]],
    "currentRound"     -> Length[hist["lossHistory"]],
    "wallSecondsSoFar" -> hist["wallSeconds"],
    "roundLossList"    -> hist["lossHistory"]
  |>]
];
text = Import["../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt", "Text"];
vocab = LLMAtoZ`CharTokeniser`charVocabulary[text];
tokId = LLMAtoZ`CharTokeniser`charTokenToIdAssociation[vocab];
idTok = LLMAtoZ`CharTokeniser`charIdToTokenAssociation[vocab];

ids = LLMAtoZ`CharTokeniser`encodeChars[text, tokId];
nTotal = Length[ids];
split  = Round[nTotal * 0.97];
trainIds = ids[[;; split]];
evalIds  = ids[[split + 1 ;;]];

contextLen = 128;

Print["Eval chars: ", Length[evalIds]];

(* Compute cross-entropy on the held-out chunk.  We process in 128-
   char chunks; for each chunk, run forward, softmax the per-position
   logits, take -log p[target], average across positions and chunks. *)

evalCrossEntropy[net_, ids_] := Module[
  {chunks, total = 0.0, n = 0, x, target, logits, sm, lps},
  chunks = Partition[ids, contextLen + 1, contextLen];
  Do[
    x = chunk[[;; contextLen]];
    target = chunk[[2 ;;]];
    logits = net[x];
    sm = Map[
      Function[v, With[{e = Exp[v - Max[v]]}, e / Total[e] + 1.*^-12]],
      logits];
    lps = MapThread[#1[[#2]] &, {sm, target}];
    total += -Total[Log[lps]];
    n += Length[lps],
    {chunk, chunks}
  ];
  total / n
];

Print["Computing eval cross-entropy ..."];
{evalTime, evalLoss} = AbsoluteTiming[evalCrossEntropy[net, evalIds]];
Print["Eval loss: ", N[evalLoss],
      "  (perplexity ", N[Exp[evalLoss]], ")",
      "  computed in ", Round[evalTime, 0.1], " s"];

trainLoss = hist["currentLoss"];
gap = N[evalLoss] / N[trainLoss];
verdict = Which[
  gap < 1.15, "still-generalising — extending training is likely worthwhile",
  gap < 1.40, "modest overfit — diminishing returns from more training",
  True,       "overfitting clearly — stop and use the current checkpoint"
];
Print["Train loss: ", N[trainLoss], "  Eval loss: ", N[evalLoss],
      "  Gap: ", Round[gap, 0.001]];
Print["Verdict: ", verdict];

(* Generations at 3 temperatures from 3 prompts, longer than the
   pulse samples (400 chars each) for clearer quality assessment. *)
prompts = {"ROMEO:", "HAMLET:\n", "First Citizen:\n"};
temperatures = {0.5, 0.8, 1.0};

generate[net_, prompt_String, n_, T_] := Module[
  {pIds = LLMAtoZ`CharTokeniser`encodeChars[prompt, tokId],
   gen, ctx, pos, logits, p, sampled},
  gen = pIds;
  Do[
    ctx = If[Length[gen] >= contextLen, gen[[-contextLen ;;]],
              PadRight[gen, contextLen, 1]];
    pos = Min[Length[gen], contextLen];
    logits = net[ctx][[pos]];
    p = Exp[(logits - Max[logits])/T]; p = p/Total[p];
    sampled = RandomChoice[p -> Range[Length[p]]];
    AppendTo[gen, sampled],
    {n}];
  LLMAtoZ`CharTokeniser`decodeChars[gen, idTok]
];

SeedRandom[42];
generations = Association @ Table[
  p -> Association @ Table[
    ("T=" <> ToString[T]) -> generate[net, p, 400, T],
    {T, temperatures}],
  {p, prompts}
];

Do[
  Do[
    Print["--- T=", T, "  prompt=", InputForm[p], " ---"];
    Print[generations[p]["T=" <> ToString[T]]];
    Print[],
    {T, temperatures}
  ],
  {p, prompts}
];

(* Write the summary JSON so we have everything to look at later. *)
summary = <|
  "trainLoss"      -> N[trainLoss],
  "evalLoss"       -> N[evalLoss],
  "evalPerplexity" -> N[Exp[evalLoss]],
  "trainPerplexity" -> N[Exp[trainLoss]],
  "generalisationGap" -> Round[gap, 0.001],
  "verdict"        -> verdict,
  "round"          -> hist["currentRound"],
  "wallMinutes"    -> Round[hist["wallSecondsSoFar"]/60., 0.1],
  "generations"    -> generations
|>;
Export[FileNameJoin[{dataDir, "charlevel_eval_summary.json"}], summary];
Print["Saved charlevel_eval_summary.json"];
Print["EVAL DONE."];
