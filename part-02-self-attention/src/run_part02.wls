#!/usr/bin/env wolframscript
(* Build all of Part 2's figures.

   Outputs:
     figures/qkdotproduct_variance.png    -- variance argument for 1/sqrt(d_k)
     figures/softmax_saturation.png       -- softmax shape with/without scaling
     figures/causal_mask_visualisation.png -- triangular mask + softmax demo
     figures/gpt2_heads_gallery.png       -- 12-head gallery at one layer
     figures/gpt2_head_specialisation.png -- three hand-picked heads
*)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];
Get["Attention.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

(* ============================================================
   FIGURE 1: variance of pre-softmax logits with / without 1/Sqrt[d_k]
   ============================================================ *)

dks = {16, 64, 256};
nSamples = 2000;

SeedRandom[1];
unscaledSamples = Association @ Table[
  d -> Table[
    With[{q = RandomReal[NormalDistribution[], d],
          k = RandomReal[NormalDistribution[], d]},
      q . k
    ],
    {nSamples}
  ],
  {d, dks}
];

scaledSamples = Association @ Table[
  d -> unscaledSamples[d] / Sqrt[N[d]],
  {d, dks}
];

Print["Empirical std of q.k     (no scaling), d_k= ",
      KeyValueMap[ToString[#1] <> "->" <> ToString[StandardDeviation[#2], 6] &,
                  unscaledSamples]];
Print["Empirical std of q.k/Sqrt[d_k]:   d_k= ",
      KeyValueMap[ToString[#1] <> "->" <> ToString[StandardDeviation[#2], 6] &,
                  scaledSamples]];

varianceFig = Row[{
  Histogram[Values[unscaledSamples], Automatic, "PDF",
    ChartLegends -> ("d_k = " <> ToString[#] & /@ Keys[unscaledSamples]),
    PlotLabel -> "no scaling: std grows like Sqrt[d_k]",
    Frame -> True, FrameLabel -> {"q\[CenterDot]k", "PDF"},
    ImageSize -> 360, PlotRange -> {{-50, 50}, All}],
  Histogram[Values[scaledSamples], Automatic, "PDF",
    ChartLegends -> ("d_k = " <> ToString[#] & /@ Keys[scaledSamples]),
    PlotLabel -> "divided by Sqrt[d_k]: std \[TildeTilde] 1 for every d_k",
    Frame -> True, FrameLabel -> {"q\[CenterDot]k / Sqrt[d_k]", "PDF"},
    ImageSize -> 360, PlotRange -> {{-5, 5}, All}]
}];
Export[FileNameJoin[{figDir, "qkdotproduct_variance.png"}], varianceFig,
       ImageResolution -> 144];
Print["Saved qkdotproduct_variance.png"];

(* ============================================================
   FIGURE 2: softmax saturation with/without scaling
   ============================================================ *)

softmax[v_] := With[{e = Exp[v - Max[v]]}, e / Total[e]];
seqLen = 8;
SeedRandom[42];
qDemo = RandomReal[NormalDistribution[], {seqLen, 256}];
kDemo = RandomReal[NormalDistribution[], {seqLen, 256}];
scoresUnscaled = qDemo . Transpose[kDemo];
scoresScaled = scoresUnscaled / Sqrt[256.];

saturationFig = GraphicsRow[{
  MatrixPlot[softmax /@ scoresUnscaled,
    PlotLabel -> "softmax(Q\[CenterDot]K\[Transpose])  no scaling, d_k = 256",
    ColorFunction -> "ThermometerColors", FrameTicks -> All,
    ImageSize -> 360, PlotLegends -> Automatic],
  MatrixPlot[softmax /@ scoresScaled,
    PlotLabel -> "softmax(Q\[CenterDot]K\[Transpose] / Sqrt[d_k])",
    ColorFunction -> "ThermometerColors", FrameTicks -> All,
    ImageSize -> 360, PlotLegends -> Automatic]
}, ImageSize -> 760];
Export[FileNameJoin[{figDir, "softmax_saturation.png"}], saturationFig,
       ImageResolution -> 144];
Print["Saved softmax_saturation.png"];

(* ============================================================
   FIGURE 3: causal mask visualisation
   ============================================================ *)

n = 8;
mask = LLMAtoZ`Attention`causalMask[n];
exampleScores = RandomReal[{-2, 2}, {n, n}];
maskedScores = exampleScores + mask;
finalWeights = LLMAtoZ`Attention`softmaxRows[maskedScores];

(* Replace -1e9 in the displayed mask with -2 for visualisation. *)
displayedMask = If[# < -100, -2., #] & /@ mask /. l_List :> If[# < -100, -2., #] & /@ l;

maskFig = GraphicsRow[{
  MatrixPlot[mask /. v_ /; v < -100 -> -2.0,
    PlotLabel -> "the causal mask (displayed as -2)",
    ColorFunction -> "ThermometerColors", ImageSize -> 250,
    FrameTicks -> {Range[n], Range[n]}],
  MatrixPlot[maskedScores /. v_ /; v < -100 -> -10.0,
    PlotLabel -> "scores + mask",
    ColorFunction -> "ThermometerColors", ImageSize -> 250,
    FrameTicks -> {Range[n], Range[n]}],
  MatrixPlot[finalWeights,
    PlotLabel -> "after row-wise softmax",
    ColorFunction -> "ThermometerColors", ImageSize -> 250,
    FrameTicks -> {Range[n], Range[n]}, PlotLegends -> Automatic]
}, ImageSize -> 800];
Export[FileNameJoin[{figDir, "causal_mask_visualisation.png"}], maskFig,
       ImageResolution -> 144];
Print["Saved causal_mask_visualisation.png"];

(* ============================================================
   FIGURE 4 & 5: GPT-2 attention heatmaps
   ============================================================ *)

gptPath = FileNameJoin[{dataDir, "gpt2.wlnet"}];
If[!FileExistsQ[gptPath],
  Print["GPT-2 cache missing -- skipping heatmap figures."];
  Print["DONE (partial)."];
  Exit[]
];

gpt = Import[gptPath];
prompt = "The quick brown fox jumps over the lazy dog";
ids = LLMAtoZ`Attention`gpt2EncodePrompt[gpt, prompt];
Print["Prompt: ", prompt];
Print["Token IDs: ", ids];

(* The encoder also gives us the token strings (for axis labelling). *)
encoder = NetExtract[gpt, "Input"];
vocabTokens = encoder[[1, 1, 2, 1, 1, 1]];  (* fragile -- best-effort *)

(* Robust alternative: just label by position numbers. *)
tokLabels = "t" <> ToString[#] & /@ Range[Length[ids]];

layer = 5;
weights = LLMAtoZ`Attention`gpt2AttentionWeights[gpt, ids, layer];
Print["Layer ", layer, " weights tensor shape: ", Dimensions[weights]];

(* 12-head gallery as a 3x4 grid *)
gallery = Grid[
  Partition[
    Table[
      MatrixPlot[weights[[h]],
        ColorFunction -> "ThermometerColors",
        ColorFunctionScaling -> True,
        FrameTicks -> None,
        PlotLabel -> Style["head " <> ToString[h], 10],
        ImageSize -> 160],
      {h, 12}
    ],
    4
  ],
  Spacings -> {0, 0}
];

gpt2GalleryFig = Labeled[
  gallery,
  Style["GPT-2 small, layer " <> ToString[layer] <>
        ", prompt = \"" <> prompt <> "\"", 12], Top
];
Export[FileNameJoin[{figDir, "gpt2_heads_gallery.png"}], gpt2GalleryFig,
       ImageResolution -> 144];
Print["Saved gpt2_heads_gallery.png"];

(* Save the layer-5 weights tensor for use in the notebook. *)
Export[FileNameJoin[{dataDir, "gpt2_attention_layer" <> ToString[layer] <> ".wxf"}],
       weights];

(* Precompute attention tensors for ALL 12 layers so the notebook's
   interactive layer/head selector can read from a single .wxf. *)
Print["Precomputing attention weights for all 12 layers..."];
allLayerWeights = Association @ Table[
  L -> LLMAtoZ`Attention`gpt2AttentionWeights[gpt, ids, L],
  {L, 12}
];
Export[FileNameJoin[{dataDir, "gpt2_attention_all_layers.wxf"}],
       <|"prompt" -> prompt, "ids" -> ids, "weights" -> allLayerWeights|>];
Print["Saved all-layer attention cache (",
      FileByteCount[FileNameJoin[{dataDir, "gpt2_attention_all_layers.wxf"}]],
      " bytes)."];

(* Specialisation gallery: choose three heads with different patterns.

   Metrics (excluding the universal first-column "attention sink"):
   - prevScore[h]   mean of weights below the main diagonal by 1
                    -- high for a "previous-token tracker"
   - selfScore[h]   mean of the main diagonal entries (excl. token 1
                    which is the sink), -- high for self-attention heads
   - spreadScore[h] minus the entropy-deficit (Gini-like) -- high for
                    heads that put their mass evenly across the
                    accessible past tokens. *)

clipMask[w_] := Module[{n = Length[w], m = w},
  m[[All, 1]] = 0.0;         (* zero the attention-sink column *)
  m / (Total /@ m + 1.*^-9)  (* re-normalise rows, epsilon in denominator *)
];

prevScore[w_]   := Module[{n = Length[w], m = clipMask[w]},
  Mean[Table[m[[i, i - 1]], {i, 2, n}]]];
selfScore[w_]   := Module[{n = Length[w], m = clipMask[w]},
  Mean[Table[m[[i, i]], {i, 2, n}]]];
spreadScore[w_] := Module[{n = Length[w], m = clipMask[w]},
  Mean[Table[
    With[{row = m[[i, ;; i]]},
      -Total[row * Log[row + 1.*^-12]] / Log[N[i]]   (* normalised entropy *)
    ],
    {i, 2, n}
  ]]
];

scores = Table[
  <|"head" -> h,
    "prev"   -> prevScore[weights[[h]]],
    "self"   -> selfScore[weights[[h]]],
    "spread" -> spreadScore[weights[[h]]]
  |>,
  {h, 12}
];

prevHead   = First @ TakeLargestBy[scores, #["prev"] &, 1];
selfHead   = First @ TakeLargestBy[
  Select[scores, #["head"] =!= prevHead["head"] &],
  #["self"] &, 1];
spreadHead = First @ TakeLargestBy[
  Select[scores, #["head"] =!= prevHead["head"] && #["head"] =!= selfHead["head"] &],
  #["spread"] &, 1];

Print["Specialisation picks: prev=", prevHead["head"],
      " self=", selfHead["head"],
      " spread=", spreadHead["head"]];

makeHeatmap[h_, caption_] :=
  Labeled[
    MatrixPlot[weights[[h]],
      ColorFunction -> "ThermometerColors",
      FrameTicks -> {Range[Length[ids]], Range[Length[ids]]},
      PlotLabel -> Style["head " <> ToString[h], 12, Bold],
      PlotLegends -> Placed[Automatic, Right],
      ImageSize -> 240],
    Style[caption, 12], Bottom
  ];

specFig = Pane[
  Grid[{
    {makeHeatmap[prevHead["head"], "previous-token tracker"],
     makeHeatmap[selfHead["head"], "self-attention diagonal"],
     makeHeatmap[spreadHead["head"], "approximately uniform over past"]}
  },
  Spacings -> {2, 2}],
  ImageSize -> 900
];
Export[FileNameJoin[{figDir, "gpt2_head_specialisation.png"}], specFig,
       ImageResolution -> 144];
Print["Saved gpt2_head_specialisation.png"];

(* Save a summary JSON *)
summary = <|
  "varianceDks" -> dks,
  "varianceUnscaledStds" ->
    KeyValueMap[ToString[#1] -> StandardDeviation[#2] &, unscaledSamples],
  "varianceScaledStds" ->
    KeyValueMap[ToString[#1] -> StandardDeviation[#2] &, scaledSamples],
  "prompt" -> prompt,
  "tokenIds" -> ids,
  "featuredLayer" -> layer,
  "headPicks" -> <|"prev" -> prevHead["head"],
                   "self" -> selfHead["head"],
                   "spread" -> spreadHead["head"]|>
|>;
Export[FileNameJoin[{dataDir, "part02_summary.json"}], summary];
Print["Wrote part02_summary.json"];

Print["DONE."];
