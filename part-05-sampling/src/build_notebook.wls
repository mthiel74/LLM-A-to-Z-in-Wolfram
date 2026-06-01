#!/usr/bin/env wolframscript
SetDirectory[DirectoryName[$InputFileName]];
partDir = ParentDirectory[];
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

cells = Flatten @ {
  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["Part 5: Sampling \[LongDash] greedy, temperature, top-k, top-p, beam, speculative"],

  text[{
    "The trained model in Part 4 is, at every step, a function from a ",
    "context window to a length-",
    math["|V|"], " logit vector. ",
    StyleBox["Sampling", FontSlant -> Italic],
    " is the policy that turns those logits into a chosen next token. ",
    "Part 5 implements five strategies plus speculative decoding, all ",
    "from scratch, and shows what each does to the output distribution."
  }],

  section["1.  Setup"],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"src/Sampling.wl\"]"],

  section["2.  Temperature: the easiest dial"],
  text[{
    "The temperature scaling ",
    math[RowBox[{"softmax(logits", "/", "T", ")"}]],
    " is the simplest knob on the distribution. ",
    math["T = 0"], " is pure argmax (greedy); ",
    math["T = 1"], " is the model's calibrated distribution; ",
    math["T > 1"], " softens toward uniform; ",
    math[RowBox[{"T", " \[Rule] ", "\[Infinity]"}]], " is exactly uniform."
  }],
  input["Import[\"figures/temperature_sweep.png\"]"],

  subsection["2.1  Interactive: turn the temperature dial"],
  input["SeedRandom[3]; logits = Sort[RandomReal[NormalDistribution[0, 1.5], 20], Greater];
Manipulate[
  BarChart[
    LLMAtoZ`Sampling`softmaxTemperature[logits, T],
    Frame -> True, FrameLabel -> {\"token id\", \"probability\"},
    PlotLabel -> Row[{\"T = \", NumberForm[T, 3], \"  \[Bullet]  entropy = \",
      NumberForm[-Total[LLMAtoZ`Sampling`softmaxTemperature[logits, T] *
        Log[LLMAtoZ`Sampling`softmaxTemperature[logits, T] + 1.*^-12]], 3]}],
    PlotRange -> {Automatic, {0, 1}}, ChartStyle -> ColorData[97][1],
    ImageSize -> 500],
  {{T, 1.0, \"temperature\"}, 0.05, 3.0, 0.05, Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  section["3.  Top-k and top-p (nucleus)"],
  text[{
    "Truncation strategies. ",
    StyleBox["Top-k", FontWeight -> Bold],
    " keeps the k highest-probability tokens and zeros the rest before ",
    "re-normalising. ",
    StyleBox["Top-p", FontWeight -> Bold],
    " (nucleus sampling) instead keeps the smallest set of top tokens ",
    "whose cumulative probability exceeds ",
    math["p"], "; this adapts the cutoff to the local sharpness of the ",
    "distribution."
  }],
  input["Import[\"figures/topk_vs_topp.png\"]"],

  section["4.  Beam search"],
  text[{
    "For translation-style tasks where one answer is correct, beam ",
    "search is the deterministic alternative: track the ",
    StyleBox["beamWidth", FontFamily -> "Courier"],
    " most likely partial sequences at every step, and at the end ",
    "return the highest-scoring complete one."
  }],
  input["(* Demo: beam search on a tiny 4-token problem.
   We define a fake stepFn that just returns scores based on the
   sequence so far -- enough to see beam search pick the best path. *)
fakeStep[ids_] := Module[{n = Length[ids]},
  RandomReal[NormalDistribution[0, 1], 6] +
    Table[If[i == Mod[n, 6] + 1, 2.0, 0.0], {i, 6}]];
SeedRandom[1];
LLMAtoZ`Sampling`beamSearch[fakeStep, {1}, 3, 8]"],

  section["5.  Speculative decoding"],
  text[{
    "A small draft model proposes ",
    math["k"], " tokens autoregressively; the main model checks them in ",
    "a single forward pass; accepted tokens commit, the first rejected ",
    "token is resampled from the main. The throughput gain is roughly the ",
    "expected accept-length divided by (1 + draft_cost_ratio * k).  Useful ",
    "when the draft is much cheaper than the main."
  }],
  input["Import[\"figures/speculative_throughput.png\"]"],

  section["6.  Summary and what comes next"],
  text[{
    "\[FilledSquare] ",
    StyleBox["Five sampling strategies", FontWeight -> Bold],
    " (greedy, temperature, top-k, top-p, beam) implemented from ",
    "scratch as one-liners or short functions over a length-|V| logit ",
    "vector.\n",
    "\[FilledSquare] ",
    StyleBox["Speculative decoding", FontWeight -> Bold],
    " (Leviathan et al. 2023) for throughput speedup with a cheap ",
    "draft model.\n",
    "\[FilledSquare] ",
    StyleBox["An interactive temperature dial", FontWeight -> Bold],
    " showing the softmax distribution shape and entropy live."
  }],
  text[{
    "In ",
    StyleBox["Part 6", FontWeight -> Bold],
    " we move from pre-training to ",
    StyleBox["supervised fine-tuning", FontSlant -> Italic],
    ": same next-token loss, but on instruction-following data ",
    "formatted in the chat template <|user|>...<|assistant|>...<|eot|>."
  }],

  section["References"],
  text[{
    "Holtzman, A. et al. (2020). The Curious Case of Neural Text Degeneration. (Top-p / nucleus sampling.)\n",
    "Leviathan, Y., Kalman, M., Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding.\n",
    "Karpathy, A. (2025). nanochat."
  }]
};

nb = Notebook[cells, WindowSize -> {900, 750}, Saveable -> True];
out = FileNameJoin[{nbDir, "Part05.nb"}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes)"];
