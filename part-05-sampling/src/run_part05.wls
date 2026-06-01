#!/usr/bin/env wolframscript
(* Part 5 figures.
   - Distribution shape vs temperature animation
   - Top-k and top-p decision boundaries
   - Speculative-decoding throughput (synthetic) *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];
Get["Sampling.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

(* ---------- Figure 1: temperature sweep on a synthetic logits row ---------- *)

SeedRandom[3];
nVocab = 20;
logits = Sort[RandomReal[NormalDistribution[0, 1.5], nVocab], Greater];

tempSweep = {0.2, 0.5, 1.0, 1.5, 3.0};
tempPlots = Table[
  BarChart[
    LLMAtoZ`Sampling`softmaxTemperature[logits, T],
    PlotLabel -> Row[{"T = ", NumberForm[T, 2]}],
    Frame -> True, FrameLabel -> {None, "p"},
    PlotRange -> {Automatic, {0, 1}},
    ChartStyle -> ColorData["Pastel"][T/3.0],
    ImageSize -> 180],
  {T, tempSweep}
];
tempFig = GraphicsRow[tempPlots, ImageSize -> 900];
Export[FileNameJoin[{figDir, "temperature_sweep.png"}], tempFig,
       ImageResolution -> 144];
Print["Saved temperature_sweep.png"];

(* ---------- Figure 2: top-k vs top-p decision boundaries ---------- *)

probsRow = LLMAtoZ`Sampling`softmaxTemperature[logits, 1.0];

topkMasks = Table[
  Module[{m = ConstantArray[0., nVocab],
          idx = Take[Reverse[Ordering[probsRow]], k]},
    m[[idx]] = probsRow[[idx]];
    BarChart[m,
      Frame -> True, FrameLabel -> {None, "p"},
      PlotLabel -> Row[{"top-k, k = ", k}],
      ChartStyle -> ColorData[97][1],
      PlotRange -> {Automatic, {0, 0.4}},
      ImageSize -> 240]
  ],
  {k, {3, 5, 10}}
];

toppMasks = Table[
  Module[{sorted = Reverse[Ordering[probsRow]], cum, keep, m},
    cum = Accumulate[probsRow[[sorted]]];
    keep = Length[TakeWhile[cum, # < p &]] + 1;
    m = ConstantArray[0., nVocab];
    m[[sorted[[;; keep]]]] = probsRow[[sorted[[;; keep]]]];
    BarChart[m,
      Frame -> True, FrameLabel -> {None, "p"},
      PlotLabel -> Row[{"top-p, p = ", NumberForm[p, 2]}],
      ChartStyle -> ColorData[97][2],
      PlotRange -> {Automatic, {0, 0.4}},
      ImageSize -> 240]
  ],
  {p, {0.5, 0.8, 0.95}}
];

samplingFig = Grid[{topkMasks, toppMasks}, Spacings -> {1, 1}];
Export[FileNameJoin[{figDir, "topk_vs_topp.png"}], samplingFig,
       ImageResolution -> 144];
Print["Saved topk_vs_topp.png"];

(* ---------- Figure 3: speculative-decoding throughput model ---------- *)

(* Synthetic: a draft model proposes k tokens, the main model accepts
   them with probability `acceptRate`.  Expected throughput speedup is
   (1 - acceptRate^(k+1)) / ((1 - acceptRate) * (k + main_cost_ratio)). *)

draftCostRatio = 0.05;  (* draft model is 20x cheaper than main *)
ks = Range[1, 12];
acceptRates = {0.3, 0.5, 0.7, 0.9};

speedup[k_, ar_] := With[
  {accLen = If[ar == 1, k + 1, (1 - ar^(k + 1)) / (1 - ar)]},
  accLen / (1 + k draftCostRatio)
];

specFig = ListLinePlot[
  Table[
    Transpose[{ks, speedup[#, ar] & /@ ks}],
    {ar, acceptRates}
  ],
  PlotLegends -> ("accept rate = " <> ToString[NumberForm[#, 2]] & /@
                  acceptRates),
  Frame -> True,
  FrameLabel -> {"draft k (tokens proposed per main call)",
                 "expected speedup over autoregressive main"},
  PlotStyle -> Thick,
  PlotLabel -> "Speculative-decoding throughput model (draft is 20x cheaper)",
  GridLines -> Automatic,
  ImageSize -> 600];
Export[FileNameJoin[{figDir, "speculative_throughput.png"}], specFig,
       ImageResolution -> 144];
Print["Saved speculative_throughput.png"];

Print["DONE."];
