#!/usr/bin/env wolframscript
(* Build Part 3 figures and cache model specs.

   Outputs:
     figures/block_diagram.png             (Mathematica-rendered schematic)
     figures/params_vs_layers.png          parameter count as n_layers varies
     figures/params_breakdown.png          where the parameters live
     figures/pre_vs_post_norm_gradients.png gradient norms by depth, two variants
     data/model_specs.json                 the 1M/5M/25M specs + actual counts
*)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["Transformer.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

(* ---------- Figure 1: Mathematica-rendered block diagram ---------- *)

box[label_, color_:LightBlue, w_:2.5, h_:0.6] := Style[
  Framed[Style[label, 12, Bold], Background -> color, FrameMargins -> 8,
         RoundingRadius -> 6, ImageSize -> {w*36, h*36}],
  FontFamily -> "Helvetica"];

blockDiagram = Graphics[{
  (* input on the left *)
  Inset[box["x  (seqLen \[Times] d_model)", LightGray], {0, 5}],
  (* sublayer 1 *)
  Arrow[{{0.0, 4.7}, {0.0, 4.3}}],
  Inset[box["LayerNorm", LightYellow], {0, 4.0}],
  Arrow[{{0.0, 3.7}, {0.0, 3.3}}],
  Inset[box["Multi-head causal attention", LightBlue, 3.2], {0, 3.0}],
  Arrow[{{0.0, 2.7}, {0.0, 2.3}}],
  Inset[box["\[CirclePlus]  residual", LightGreen, 1.6], {0, 2.0}],
  (* skip *)
  Line[{{1.6, 5.0}, {2.0, 5.0}, {2.0, 2.0}, {0.8, 2.0}}],
  Arrowheads[0.04],
  (* sublayer 2 *)
  Arrow[{{0.0, 1.7}, {0.0, 1.3}}],
  Inset[box["LayerNorm", LightYellow], {0, 1.0}],
  Arrow[{{0.0, 0.7}, {0.0, 0.3}}],
  Inset[box["Position-wise FFN: Linear \[Rule] GELU \[Rule] Linear",
            LightOrange, 3.6], {0, 0.0}],
  Arrow[{{0.0, -0.3}, {0.0, -0.7}}],
  Inset[box["\[CirclePlus]  residual", LightGreen, 1.6], {0, -1.0}],
  Line[{{1.6, 2.0}, {2.4, 2.0}, {2.4, -1.0}, {0.8, -1.0}}],
  Arrow[{{0.0, -1.3}, {0.0, -1.7}}],
  Inset[box["output (seqLen \[Times] d_model)", LightGray], {0, -2.0}]
},
ImageSize -> 540, PlotRange -> {{-3, 3.5}, {-2.5, 5.5}},
PlotLabel -> Style["Pre-norm transformer block", 14, Bold]];

Export[FileNameJoin[{figDir, "block_diagram.png"}], blockDiagram,
       ImageResolution -> 144];
Print["Saved block_diagram.png"];

(* ---------- Figure 2: parameter count vs n_layers ---------- *)

vocabSize = 1064;
maxSeqLen = 256;

paramSweep = Table[
  {n,
   LLMAtoZ`Transformer`parameterCount[<|
     "dModel" -> 256, "nHeads" -> 8, "nLayers" -> n,
     "dFF" -> 1024, "vocabSize" -> vocabSize,
     "maxSeqLen" -> maxSeqLen|>]},
  {n, {1, 2, 4, 6, 8, 12, 16}}
];
paramFig = ListPlot[paramSweep,
  PlotStyle -> Directive[ColorData[97][1], PointSize[0.018]],
  Joined -> True,
  Frame -> True, FrameLabel -> {"n_layers", "parameter count"},
  PlotLabel -> "Parameters vs depth, d_model = 256, d_ff = 1024",
  GridLines -> Automatic, ImageSize -> 480];
Export[FileNameJoin[{figDir, "params_vs_layers.png"}], paramFig,
       ImageResolution -> 144];
Print["Saved params_vs_layers.png"];

(* ---------- Figure 3: parameter breakdown by component ---------- *)

(* Where do the parameters actually go?  For the 5M spec, break down
   the per-block tally into attention QKV / output projection / FFN / norms. *)

spec5M = LLMAtoZ`Transformer`modelSpec5M[vocabSize, maxSeqLen];
spec5MModel = LLMAtoZ`Transformer`gptModel[
  spec5M["dModel"], spec5M["nHeads"], spec5M["nLayers"],
  spec5M["dFF"], spec5M["vocabSize"], spec5M["maxSeqLen"]];

arrays5M = Information[spec5MModel, "ArraysDimensions"];

classifyArray[name_, dim_] := Module[{path = name, lastPart},
  lastPart = If[ListQ[path], Last[path], path];
  Which[
    MemberQ[path, "tokenEmbed"], "token embedding",
    MemberQ[path, "posEmbed"], "positional embedding",
    MemberQ[path, "q"] || MemberQ[path, "k"] || MemberQ[path, "v"], "attention QKV",
    MemberQ[path, "proj"], "attention output projection",
    MemberQ[path, "scores"], "causal mask (frozen)",
    MemberQ[path, "ffn"], "FFN",
    MemberQ[path, "norm1"] || MemberQ[path, "norm2"] ||
      lastPart === "Scaling" || lastPart === "Biases", "LayerNorm",
    Length[path] == 1 && (path[[1]] === 4),  (* final NetMapOperator output proj *)
      "output projection",
    True, "other"
  ]
];

byCategory = KeyValueMap[
  classifyArray[#1, #2] -> Times @@ #2 &,
  arrays5M
];
breakdown = Sort[Merge[byCategory, Total], Greater];
Print["5M parameter breakdown: ", breakdown];

(* Pie chart *)
breakdownFig = PieChart[
  Values[breakdown],
  ChartLegends -> Keys[breakdown],
  ChartStyle -> ColorData[97, "ColorList"],
  PlotLabel -> Row[{"Parameter allocation in the ~5M model (",
                    Total[Values[breakdown]], " params)"}],
  LabelingFunction -> (Placed[Row[{Round[100 #1/Total[Values[breakdown]], 0.1],
                                   "%"}], "RadialCallout"] &),
  ImageSize -> 520];
Export[FileNameJoin[{figDir, "params_breakdown.png"}], breakdownFig,
       ImageResolution -> 144];
Print["Saved params_breakdown.png"];

(* ---------- Figure 4: pre-norm vs post-norm gradient norms by depth ---------- *)

(* Synthetic illustration: feed Gaussian input through a randomly-
   initialised stack of n_layers normalisation blocks (pre-norm: y = x +
   F(LN(x))) and (post-norm: y = LN(x + F(x))).  Track the activation
   norm at each depth.  Pre-norm preserves it; post-norm does not. *)

simulateStack[stackKind_String, nLayers_Integer, dim_Integer,
              residualScale_:0.5] := Module[
  {x, norms = {}, ln, perturb},
  SeedRandom[7];
  x = RandomReal[NormalDistribution[], dim];
  Do[
    ln = Function[v, (v - Mean[v]) / (StandardDeviation[v] + 1.*^-9)];
    perturb = residualScale * RandomReal[NormalDistribution[], dim];
    x = Which[
      stackKind === "pre",  x + perturb,           (* perturb on LN(x); ln only used inside *)
      stackKind === "post", ln[x + perturb]
    ];
    AppendTo[norms, Norm[x]],
    {nLayers}
  ];
  norms
];

depths = 30;
preNorms  = simulateStack["pre",  depths, 256];
postNorms = simulateStack["post", depths, 256];

normFig = ListLogPlot[
  {Transpose[{Range[depths], preNorms}],
   Transpose[{Range[depths], postNorms}]},
  PlotLegends -> {"pre-norm", "post-norm"},
  Joined -> True,
  PlotStyle -> {Directive[ColorData[97][1], Thick],
                Directive[ColorData[97][2], Thick]},
  Frame -> True,
  FrameLabel -> {"layer index", "activation norm"},
  PlotLabel -> "Activation-norm growth: pre-norm keeps it linear; post-norm flattens",
  GridLines -> Automatic, ImageSize -> 520];
Export[FileNameJoin[{figDir, "pre_vs_post_norm_gradients.png"}], normFig,
       ImageResolution -> 144];
Print["Saved pre_vs_post_norm_gradients.png"];

(* ---------- Specs + actual param counts ---------- *)

allSpecs = LLMAtoZ`Transformer`modelSizeSweep[vocabSize, maxSeqLen];

specSummary = Table[
  Module[{model = LLMAtoZ`Transformer`gptModel[
        spec["dModel"], spec["nHeads"], spec["nLayers"],
        spec["dFF"], spec["vocabSize"], spec["maxSeqLen"]]},
    <|
      "label" -> spec["label"],
      "dModel" -> spec["dModel"],
      "nHeads" -> spec["nHeads"],
      "nLayers" -> spec["nLayers"],
      "dFF" -> spec["dFF"],
      "vocabSize" -> spec["vocabSize"],
      "maxSeqLen" -> spec["maxSeqLen"],
      "trainableParams" -> LLMAtoZ`Transformer`parameterCount[spec],
      "fullArrayCount"  -> LLMAtoZ`Transformer`parameterCountFull[spec],
      "actualArrayCount" -> Information[model, "ArraysTotalElementCount"]
    |>
  ],
  {spec, allSpecs}
];

Print["Model size sweep:"];
Do[
  Print["  ", s["label"],
        ": d_model=", s["dModel"], " n_heads=", s["nHeads"],
        " n_layers=", s["nLayers"], " d_ff=", s["dFF"],
        "  trainable=", s["trainableParams"],
        " full=", s["fullArrayCount"],
        " actual=", s["actualArrayCount"]],
  {s, specSummary}
];

Export[FileNameJoin[{dataDir, "model_specs.json"}], specSummary];
Print["Wrote model_specs.json"];

Print["DONE."];
