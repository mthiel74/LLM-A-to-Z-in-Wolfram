#!/usr/bin/env wolframscript
SetDirectory[DirectoryName[$InputFileName]];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];
nbDir   = FileNameJoin[{partDir, "notebook"}];
If[!DirectoryQ[nbDir], CreateDirectory[nbDir]];

title[s_]      := Cell[s, "Title"];
subtitle[s_]   := Cell[s, "Subtitle"];
section[s_]    := Cell[s, "Section"];
subsection[s_] := Cell[s, "Subsection"];
text[s_String] := Cell[s, "Text"];
text[parts_List] := Cell[TextData[parts], "Text"];
input[s_String] := Cell[s, "Input", FormatType -> InputForm];
math[box_] := Cell[BoxData[box], "Text", FontSize -> 14];

(* For the progress-log gallery section, read the per-round samples
   directly while we build the notebook so we can paste them in as
   pre-rendered Text cells.  No reliance on evaluation order. *)
progressLogPath = FileNameJoin[{dataDir, "charlevel_progress_log.wxf"}];
progressLog = If[FileExistsQ[progressLogPath],
  Import[progressLogPath], <||>];
progressRounds = If[AssociationQ[progressLog],
  SortBy[Values[progressLog], #["round"] &], {}];
Print["Embedding ", Length[progressRounds],
      " progress-log rounds into the notebook."];

(* Helper: render one progress entry as a labelled block of cells. *)
progressEntry[entry_Association] := With[
  {round = entry["round"],
   loss  = N @ entry["loss"],
   secs  = N @ entry["wallSeconds"],
   sROMEO  = entry["samples"]["ROMEO:"]["T=0.8"],
   sHAMLET = entry["samples"]["HAMLET:\n"]["T=0.8"]},
  Flatten @ {
    Cell[TextData[{
      StyleBox["Round " <> ToString[round], FontWeight -> Bold, FontSize -> 14],
      "  \[Bullet]  loss = ",
      ToString[NumberForm[loss, {Infinity, 3}]],
      "  \[Bullet]  perplexity = ",
      ToString[NumberForm[Exp[loss], {Infinity, 2}]],
      "  \[Bullet]  wall = ",
      ToString[NumberForm[secs/60., {Infinity, 1}]], " min"
    }], "Subsubsection"],
    Cell[sROMEO,  "Program", FontFamily -> "Courier", FontSize -> 11],
    Cell[sHAMLET, "Program", FontFamily -> "Courier", FontSize -> 11]
  }
];

cells = Flatten @ {
  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["Part 4: Pre-training a tiny model on Tiny Shakespeare"],

  text[{
    "After three parts of careful construction, Part 4 finally ",
    StyleBox["trains", FontSlant -> Italic],
    " the model.  No GPU, no cloud, no PyTorch \[Dash] a small ",
    "(~1.8M parameter) char-level transformer that, after a few hours ",
    "of CPU on a Mac mini, produces something that looks, locally and ",
    "convincingly, like Shakespeare.  The only external dependency is ",
    "the 1 MB Tiny Shakespeare text from Karpathy's char-rnn (2015)."
  }],

  text[{
    "The deliverable is the saved checkpoint at ",
    StyleBox["data/charlevel_shakespeare_trained.wlnet", FontFamily -> "Courier"],
    ", consumed by Parts 5\[Dash]8.  The headline figures are the ",
    "loss curve (Section 7) and the gallery of generations evolving ",
    "across training rounds (Section 6)."
  }],

  section["1.  Setup and corpus"],
  text[{
    "All paths in this notebook are relative to the notebook's own ",
    "directory.  The ",
    StyleBox["data/", FontFamily -> "Courier"], ", ",
    StyleBox["src/", FontFamily -> "Courier"], " and ",
    StyleBox["figures/", FontFamily -> "Courier"],
    " subdirectories are symlinks to the part's top-level folders, so ",
    "evaluating from any front end \[OpenCurlyDoubleQuote]just works\
\[CloseCurlyDoubleQuote] without setting the working directory by hand."
  }],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl\"]
Get[\"../../part-02-self-attention/src/Attention.wl\"]
Get[\"../../part-03-transformer-block/src/Transformer.wl\"]
Get[\"src/Training.wl\"]
Get[\"src/CharTokeniser.wl\"]
Get[\"../../Tiers.wl\"]"],

  input["text = Import[
  \"../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt\",
  \"Text\"];
StringLength[text]"],

  section["2.  Choice: BPE vs character-level"],
  text[{
    "Part 1 built a ~4000-token BPE tokeniser.  That is the right ",
    "choice for a model with a parameter budget in the tens or hundreds ",
    "of millions; it is the wrong choice for a 1\[Dash]5 M-parameter model ",
    "trained on a 1 MB corpus, because for a tiny model the embedding ",
    "tables alone eat most of the parameter budget."
  }],
  text[{
    "A character-level vocabulary (65 symbols, including punctuation ",
    "and a newline) shrinks the embedding tables by ~60\[Times].  The ",
    "trade-off is longer sequences \[LongDash] every character becomes ",
    "its own token \[LongDash] but at this scale the per-parameter ",
    "modelling capacity is what matters.  This is exactly the choice ",
    "Karpathy made in char-rnn (2015) at LSTM ~1M parameters on the ",
    "same corpus."
  }],
  input["vocab = LLMAtoZ`CharTokeniser`charVocabulary[text];
tokId = LLMAtoZ`CharTokeniser`charTokenToIdAssociation[vocab];
idTok = LLMAtoZ`CharTokeniser`charIdToTokenAssociation[vocab];
Length[vocab]"],

  section["3.  Tiers: one dial controls everything"],
  text[{
    "Every long-running notebook in this series begins with one line: ",
    StyleBox["tier = \"smallCPU\"", FontFamily -> "Courier"], ".  ",
    "Five tiers are defined in ",
    StyleBox["Tiers.wl", FontFamily -> "Courier"], "."
  }],
  input["Grid[
  Prepend[
    Table[Module[{t = LLMAtoZ`Tiers`tier[n]},
      {n, t[\"modelSpec\"][\"label\"], t[\"dataset\"],
       \"~£\" <> ToString[t[\"estCostGBP\"]],
       \"~\" <> ToString[t[\"estTimeMinutes\"]] <> \" min\"}],
      {n, LLMAtoZ`Tiers`availableTiers[]}],
    Style[#, Bold] & /@ {\"tier\", \"model\", \"dataset\", \"approx cost\", \"approx time\"}],
  Frame -> All, Background -> {None, {LightGray, None}},
  Alignment -> Center]"],
  text[{
    "This post uses the char-level variant.  The cloud-tier code path ",
    "is identical (",
    StyleBox["submitRemotePretraining[...]", FontFamily -> "Courier"],
    ") for readers with a Wolfram Cloud subscription."
  }],

  section["4.  A bug we didn't see, until char-level made us"],
  text[{
    "This section earns its place because it captures a class of silent ",
    "error that the series has the right tooling to surface.  The natural ",
    "way to define cross-entropy loss in WL is"
  }],
  input["(* DON'T do this with raw logits *)
(*  NetTrain[..., LossFunction -> CrossEntropyLossLayer[\"Index\"]]  *)"],
  text[{
    StyleBox["CrossEntropyLossLayer[\"Index\"]", FontFamily -> "Courier"],
    " expects ", StyleBox["probabilities", FontSlant -> Italic],
    " as input and computes ",
    math[RowBox[{"-log(p[target])"}]],
    ", which is non-negative.  Our ",
    StyleBox["gptModel", FontFamily -> "Courier"],
    " ends with a ",
    StyleBox["LinearLayer[vocabSize]", FontFamily -> "Courier"],
    " that emits raw logits, not probabilities.  Feeding logits straight ",
    "into the loss layer produces a number that is not cross-entropy."
  }],
  text[{
    "For BPE vocab ~4000 this happened to be okay: the optimiser ",
    "couldn't drive a single logit unbounded without similarly moving ",
    "all the others through the shared linear projection, and the ",
    "training landed in regimes that produced sensible token ",
    "distributions.  For a 65-character vocabulary, no such constraint ",
    "applies: the first overnight char-level run, after 6.5 hours, ",
    "produced a model whose maximum logit was ",
    StyleBox["9,415", FontWeight -> Bold],
    " and whose generations were the single character ",
    StyleBox["'s'", FontFamily -> "Courier"],
    " repeated to fill the buffer.  The loss curve descended monotonically ",
    "past zero into ", math[RowBox[{"-", "10"}]], "."
  }],
  inputBox = Cell[BoxData[
    RowBox[{"Import", "[",
      "\"figures/charlevel_loss_curve_BROKEN.png\"", "]"}]],
    "Input"]; inputBox,
  text[{
    "A cross-entropy loss curve that descends ",
    StyleBox["below zero", FontSlant -> Italic],
    " is the signature you should learn to recognise.  Cross-entropy ",
    "is bounded below by zero.  A loss going negative is not \
\[OpenCurlyDoubleQuote]training going great\[CloseCurlyDoubleQuote]; ",
    "it is \[OpenCurlyDoubleQuote]you are not optimising what you think ",
    "you are.\[CloseCurlyDoubleQuote]"
  }],
  text[{
    "The fix is one line: wrap the model in ",
    StyleBox["NetMapOperator[SoftmaxLayer[]]", FontFamily -> "Courier"],
    " before training so the loss layer sees probabilities; strip the ",
    "softmax after training so the saved checkpoint still emits logits ",
    "(cleaner for temperature sampling in Part 5).  ",
    StyleBox["trainGptModel", FontFamily -> "Courier"],
    " in the package now does this automatically."
  }],

  section["5.  Cosine schedule, checkpointing, Wolfram-version gotchas"],
  subsection["5.1  Cosine schedule with linear warmup"],
  text[{
    "The first ~10% of training linearly ramps the learning rate up to ",
    "its peak; the remaining 90% cosine-decays to 10% of peak.  Helps ",
    "with first-batch instability and late-training noise."
  }],
  input["Plot[LLMAtoZ`Training`cosineSchedule[r, 100, 0.001, 0.1],
  {r, 0, 100},
  Frame -> True,
  FrameLabel -> {\"round (% of total)\", \"learning rate\"},
  PlotLabel -> \"Linear warmup (10%) + cosine decay\",
  PlotStyle -> Directive[ColorData[97][2], Thick],
  ImageSize -> 480]"],

  subsection["5.2  Periodic checkpointing"],
  text[{
    "Long training runs die.  ",
    StyleBox["NetTrain", FontFamily -> "Courier"],
    "'s ",
    StyleBox["TrainingProgressFunction", FontFamily -> "Courier"],
    " writes the current model to disk every 5 minutes.  Saved ",
    "in our case the run we'd otherwise have lost when the 15.0 ",
    "prerelease kernel disconnected mid-run from its license server."
  }],

  subsection["5.3  Wolfram version matters for long runs"],
  text[{
    "Sustained ",
    StyleBox["NetTrain", FontFamily -> "Courier"],
    " calls on the 15.0-prerelease wolframscript binary repeatedly ",
    "died mid-run with ",
    StyleBox["\"The product exited because of a license error.\"",
            FontSlant -> Italic],
    "  Switching to the stable 14.3 binary fixed it.  If you are on a ",
    "prerelease version of WL, training jobs that need to run for hours ",
    "deserve a quick sanity check on a stable version first."
  }],

  section["6.  Watching the model dream up Shakespeare"],
  text[{
    "A second background process snapshot model generations at every ",
    "round during training.  Below, every round of training is shown ",
    "with its loss, its perplexity, the wall time at which it landed, ",
    "and a 200-character generation continuing from both ",
    StyleBox["ROMEO:", FontFamily -> "Courier"], " and ",
    StyleBox["HAMLET:", FontFamily -> "Courier"],
    " at temperature 0.8.  Watch the output evolve from noise to ",
    "Shakespeare-flavoured prose."
  }],

  Sequence @@ Flatten[progressEntry /@ progressRounds, 1],

  section["7.  Loss curve and final result"],
  input["Import[\"figures/charlevel_loss_curve.png\"]"],
  text[{
    "The curve has the same shape Karpathy's char-rnn produced in 2015 ",
    "at comparable scale: a fast initial drop in the first round (the ",
    "model learns the character-frequency distribution), a slower elbow ",
    "as it discovers word boundaries, and a long shallow tail where it ",
    "refines local syntax."
  }],
  text[{
    "We trained for 5.5 hours, completed 23 rounds, hit a final round ",
    "loss of ~0.74 (train perplexity ~2.1).  Held-out eval loss came out ",
    "to 2.59 (eval perplexity 13.4), giving a train/eval gap of 3.49 ",
    "\[LongDash] a clear overfit signature.  At this scale, with 1.8M ",
    "parameters memorising a 1.08M-character corpus, the asymptote of ",
    "the train loss is well below the asymptote of the eval loss; ",
    "further training would push train down without making the model ",
    "actually generalise better.  The checkpoint we ship is the one at ",
    "the end of this run."
  }],
  input["Import[\"data/charlevel_eval_summary.json\"][[
    {\"trainLoss\", \"trainPerplexity\", \"evalLoss\", \"evalPerplexity\",
     \"generalisationGap\", \"verdict\"}]]"],

  section["8.  Summary"],
  text[{
    "\[FilledSquare] ",
    StyleBox["A trained char-level transformer", FontWeight -> Bold],
    " on Tiny Shakespeare (1.8M parameters), saved to ",
    StyleBox["data/charlevel_shakespeare_trained.wlnet", FontFamily -> "Courier"],
    ".  Reusable in Parts 5\[Dash]8.\n",
    "\[FilledSquare] ",
    StyleBox["A complete training loop", FontWeight -> Bold],
    " (",
    StyleBox["trainGptModel", FontFamily -> "Courier"],
    ") that auto-wraps the model in a softmax for proper cross-entropy ",
    "computation, ships a cosine + linear-warmup schedule, and saves a ",
    "checkpoint every five minutes.\n",
    "\[FilledSquare] ",
    StyleBox["A bug postmortem", FontWeight -> Bold],
    " on the silent softmax-omission failure: when your cross-entropy ",
    "loss goes negative, look at the model output.\n",
    "\[FilledSquare] ",
    StyleBox["A training-in-progress gallery", FontWeight -> Bold],
    " of generations evolving across rounds (Section 6).\n",
    "\[FilledSquare] ",
    StyleBox["A cloud-submission path", FontWeight -> Bold],
    " (",
    StyleBox["submitRemotePretraining", FontFamily -> "Courier"],
    ") for scaling to the 25M or 124M tiers on Wolfram Cloud."
  }],
  text[{
    "In ", StyleBox["Part 5", FontWeight -> Bold],
    " we look at the inference side of this model: greedy decoding, ",
    "temperature sampling, top-k, top-p, beam search, and speculative ",
    "decoding."
  }],

  section["References"],
  text[{
    "Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks.\n",
    "Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models. (Chinchilla.)\n",
    "Karpathy, A. (2025). nanochat.\n",
    "Raschka, S. (2024). Build a Large Language Model from Scratch. Manning."
  }]
};

nb = Notebook[cells, WindowSize -> {900, 750}, Saveable -> True];
out = FileNameJoin[{nbDir, "Part04.nb"}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes)"];
