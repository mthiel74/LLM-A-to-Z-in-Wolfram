#!/usr/bin/env wolframscript
(* CPU sanity test for NanoChat.wl.  Builds a tiny nano model,
   runs a forward pass on random token IDs, and prints the output
   shape.  Should complete in under 30 seconds on a laptop. *)

repo  = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
Get[FileNameJoin[{repo, "part-02-self-attention", "src", "Attention.wl"}]];
Get[FileNameJoin[{repo, "part-09-from-scratch", "src", "NanoChat.wl"}]];

Print["== rmsNorm =="];
norm = LLMAtoZ`NanoChat`rmsNorm[8, 4];
Print["  net = ", Head[norm]];
Print["  in : ", {4, 8}, " -> out shape: ",
  Dimensions @ norm[RandomReal[NormalDistribution[], {4, 8}]]];

Print["== ropeAngleTables =="];
{ct, st} = LLMAtoZ`NanoChat`ropeAngleTables[4, 8];
Print["  cos dims: ", Dimensions[ct], "  sin dims: ", Dimensions[st]];
Print["  cos[1] (pos 0, should be ones): ", ct[[1]]];
Print["  sin[1] (pos 0, should be zeros): ", st[[1]]];

Print["== applyRoPE =="];
rope = LLMAtoZ`NanoChat`applyRoPE[4, 8];
Print["  out shape: ", Dimensions @ rope[RandomReal[{-1, 1}, {4, 8}]]];

Print["== swiGluMLP =="];
ffn = LLMAtoZ`NanoChat`swiGluMLP[16, 32, 4];
Print["  out shape: ", Dimensions @ ffn[RandomReal[{-1, 1}, {4, 16}]]];

Print["== nanoSingleHeadAttention =="];
h = LLMAtoZ`NanoChat`nanoSingleHeadAttention[16, 4, 4];
Print["  out shape: ", Dimensions @ h[RandomReal[{-1, 1}, {4, 16}]]];

Print["== nanoMultiHeadAttention =="];
mha = LLMAtoZ`NanoChat`nanoMultiHeadAttention[16, 4, 4];
Print["  out shape: ", Dimensions @ mha[RandomReal[{-1, 1}, {4, 16}]]];

Print["== nanoTransformerBlock =="];
blk = LLMAtoZ`NanoChat`nanoTransformerBlock[16, 4, 32, 4];
Print["  out shape: ", Dimensions @ blk[RandomReal[{-1, 1}, {4, 16}]]];

Print["== nanoModel (tiny) =="];
tinySpec = LLMAtoZ`NanoChat`nanoModelSpecTiny[100, 16];
mdl = LLMAtoZ`NanoChat`nanoModel[
  tinySpec["dModel"], tinySpec["nHeads"], tinySpec["nLayers"],
  tinySpec["dFF"], tinySpec["vocabSize"], tinySpec["maxSeqLen"]];
Print["  spec = ", tinySpec];
Print["  closed-form params = ", LLMAtoZ`NanoChat`nanoParameterCount[tinySpec]];
Print["  actual arrays total = ", Information[mdl, "ArraysTotalElementCount"]];
ids = RandomInteger[{1, 100}, 16];
out = mdl[ids];
Print["  forward out shape (should be {16, 100}): ", Dimensions[out]];

Print["== parameter counts =="];
Do[
  spec = sp[50257, 1024];
  Print[ "  ", spec["label"], " : ",
    NumberForm[LLMAtoZ`NanoChat`nanoParameterCount[spec], DigitBlock -> 3]],
  {sp, {LLMAtoZ`NanoChat`nanoModelSpec30M,
        LLMAtoZ`NanoChat`nanoModelSpec100M,
        LLMAtoZ`NanoChat`nanoModelSpec200M}}
];

Print["DONE"];
