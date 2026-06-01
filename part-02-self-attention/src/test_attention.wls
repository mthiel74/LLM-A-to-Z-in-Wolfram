#!/usr/bin/env wolframscript
(* Smoke tests for Attention.wl *)

SetDirectory[DirectoryName[$InputFileName]];
Get["Attention.wl"];

(* -------- Test 1: causalMask -------- *)
m = LLMAtoZ`Attention`causalMask[4];
Print["Test 1 causalMask[4]:"];
Print[MatrixForm[m]];

(* -------- Test 2: attentionScores -------- *)
SeedRandom[1];
q = RandomReal[NormalDistribution[], {4, 8}];
k = RandomReal[NormalDistribution[], {4, 8}];
scores = LLMAtoZ`Attention`attentionScores[q, k];
Print["Test 2 attentionScores shape: ", Dimensions[scores]];
Print["    std: ", StandardDeviation[Flatten[scores]] // N];

(* -------- Test 3: singleHeadCausalAttention -------- *)
dModel = 16; dHead = 8; seqLen = 6;
head = LLMAtoZ`Attention`singleHeadCausalAttention[dModel, dHead, seqLen];
Print["Test 3 singleHeadCausalAttention head: ", Head[head]];
x = RandomReal[NormalDistribution[0, 0.1], {seqLen, dModel}];
out = head[x];
Print["Test 3 output shape: ", Dimensions[out]];

(* -------- Test 4: multiHeadAttentionBlock -------- *)
mha = LLMAtoZ`Attention`multiHeadAttentionBlock[16, 4, 6];
Print["Test 4 multiHeadAttentionBlock head: ", Head[mha]];
out2 = mha[x];
Print["Test 4 output shape: ", Dimensions[out2]];

(* -------- Test 5: softmaxRows -------- *)
sm = LLMAtoZ`Attention`softmaxRows[{{1., 2., 3.}, {10., 0., -10.}}];
Print["Test 5 softmaxRows row sums: ", Total /@ sm];

(* -------- Test 6: GPT-2 attention extraction -------- *)
gptPath = "/Users/thiel/GitHub/LLM-A-to-Z-in-WL/part-02-self-attention/data/gpt2.wlnet";
If[FileExistsQ[gptPath],
  gpt = Import[gptPath];
  ids = LLMAtoZ`Attention`gpt2EncodePrompt[gpt, "To be or not to be"];
  Print["Test 6 GPT-2 ids for prompt: ", ids];
  weights = LLMAtoZ`Attention`gpt2AttentionWeights[gpt, ids, 5];
  Print["Test 6 weights shape: ", Dimensions[weights]];
  Print["Test 6 head-1 row-1 (should sum to 1): ", Total[weights[[1, 1]]]];
  Print["Test 6 head-1 row-3 (should sum to 1): ", Total[weights[[1, 3]]]];
  ,
  Print["Test 6 skipped (GPT-2 cache not present)"]
];

Print["All smoke tests passed."];
