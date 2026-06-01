#!/usr/bin/env wolframscript
(* Smoke tests for Sampling.wl *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];
Get["Sampling.wl"];

results = {};
push[name_, test_] := AppendTo[results,
  <|"name" -> name, "outcome" -> test["Outcome"]|>];

(* softmaxTemperature *)
logits = {1., 2., 3., 4., 5.};
sm = LLMAtoZ`Sampling`softmaxTemperature[logits, 1.0];
push["softmax sums to 1",
  VerificationTest[Total[sm], 1.0, Tolerance -> 1.*^-9]];
push["softmax monotonic in logits",
  VerificationTest[sm, Sort[sm]]];

(* greedySample picks the max *)
push["greedySample picks the max",
  VerificationTest[LLMAtoZ`Sampling`greedySample[{1., 5., 3., 2.}], 2]];

(* topK only allows top-k indices *)
SeedRandom[42];
counts = Tally[Table[
  LLMAtoZ`Sampling`topKSample[logits, 2, 1.0],
  {200}]];
allowed = Take[Reverse[Ordering[logits]], 2];
push["topK only samples top-k positions",
  VerificationTest[
    AllTrue[counts, MemberQ[allowed, First[#]] &],
    True]];

(* topP eventually accepts all when p=1 *)
counts = Tally[Table[
  LLMAtoZ`Sampling`topPSample[{1., 1.1, 1.2, 0.9, 1.05}, 0.999, 1.0],
  {300}]];
push["topP with p=1 covers all positions",
  VerificationTest[Length[counts] == 5, True]];

(* Temperature 0+ is concentrated *)
counts = Tally[Table[
  LLMAtoZ`Sampling`temperatureSample[logits, 0.01],
  {200}]];
push["very low T concentrates on max",
  VerificationTest[First[First[Reverse[SortBy[counts, Last]]]] == 5, True]];

passed = Count[results, KeyValuePattern["outcome" -> "Success"]];
Print["Passed ", passed, "/", Length[results]];
If[passed < Length[results], Print[results]; Exit[1]];
Print["All sampling smoke tests passed."];
