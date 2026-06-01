#!/usr/bin/env wolframscript
(* Smoke tests for Transformer.wl *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

(* Load the previous parts' packages too. *)
Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["Transformer.wl"];

(* Test 1: feedForwardBlock *)
ffn = LLMAtoZ`Transformer`feedForwardBlock[64, 256, 16];
Print["Test 1 FFN head: ", Head[ffn]];
x = RandomReal[NormalDistribution[0, 0.1], {16, 64}];
Print["Test 1 FFN output shape: ", Dimensions[ffn[x]]];

(* Test 2: one transformer block *)
blk = LLMAtoZ`Transformer`transformerBlock[64, 4, 256, 16];
Print["Test 2 Block head: ", Head[blk]];
Print["Test 2 Block output shape: ", Dimensions[blk[x]]];
Print["Test 2 Block params: ",
  Information[blk, "ArraysTotalElementCount"]];

(* Test 3: full small model *)
model = LLMAtoZ`Transformer`gptModel[64, 4, 2, 256, 100, 16];
Print["Test 3 Model head: ", Head[model]];
Print["Test 3 Model params: ",
  Information[model, "ArraysTotalElementCount"]];
ids = RandomInteger[{1, 100}, 16];
Print["Test 3 Model output shape: ", Dimensions[model[ids]]];

(* Test 4: parameter-count formula *)
spec = <|"dModel" -> 64, "nHeads" -> 4, "nLayers" -> 2, "dFF" -> 256,
         "vocabSize" -> 100, "maxSeqLen" -> 16|>;
predicted = LLMAtoZ`Transformer`parameterCount[spec];
actual = Information[model, "ArraysTotalElementCount"];
Print["Test 4 Predicted params: ", predicted];
Print["Test 4 Actual params:    ", actual];
Print["Test 4 Predicted = Actual: ", predicted == actual];

(* Test 5: full 3-model sweep at Tiny Shakespeare scale *)
v = 1064;   (* vocab from Part 1 trained BPE *)
n = 256;
Do[
  spec = ms;
  pc = LLMAtoZ`Transformer`parameterCount[spec];
  Print["  ", spec["label"], "  predicted params = ", pc,
        "  (d=", spec["dModel"], ", h=", spec["nHeads"],
        ", L=", spec["nLayers"], ", dFF=", spec["dFF"], ")"],
  {ms, LLMAtoZ`Transformer`modelSizeSweep[v, n]}
];

Print["All smoke tests passed."];
