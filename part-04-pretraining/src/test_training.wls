#!/usr/bin/env wolframscript
(* Smoke tests for Training.wl *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["../../part-03-transformer-block/src/Transformer.wl"];
Get["Training.wl"];

results = {};
push[name_, test_] := AppendTo[results,
  <|"name" -> name, "outcome" -> test["Outcome"]|>];

(* 1. prepareSequenceDataset *)
ids = Range[100];
data = LLMAtoZ`Training`prepareSequenceDataset[ids, 8];
push["prepareSequenceDataset length",
  VerificationTest[Length[data], 12]
];
push["first input/target shape",
  VerificationTest[
    {Length[data[[1, "Input"]]], Length[data[[1, "Target"]]]},
    {8, 8}
  ]
];
push["target shifted by one",
  VerificationTest[data[[1, "Target"]] == data[[1, "Input"]] + 1, True]
];

(* 2. defaultHyperparameters *)
hp = LLMAtoZ`Training`defaultHyperparameters[];
push["defaultHyperparameters returns assoc",
  VerificationTest[KeyExistsQ[hp, "BaseLR"], True]
];

(* 3. cosineSchedule *)
lrs = Table[LLMAtoZ`Training`cosineSchedule[r, 100, 0.01, 0.1], {r, 0, 100, 10}];
push["cosineSchedule produces 11 values", VerificationTest[Length[lrs], 11]];
push["cosineSchedule warmup starts low",
  VerificationTest[lrs[[1]] < lrs[[2]], True]];
push["cosineSchedule end value is 0.1*baseLR (= 0.001)",
  VerificationTest[Abs[Last[lrs] - 0.001] < 0.01, True]];

passed = Count[results, KeyValuePattern["outcome" -> "Success"]];
Print["Passed ", passed, "/", Length[results]];
If[passed < Length[results], Print[results]; Exit[1]];
Print["All training smoke tests passed."];
