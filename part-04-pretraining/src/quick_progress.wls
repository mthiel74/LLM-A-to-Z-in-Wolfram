#!/usr/bin/env wolframscript
(* One-shot snapshot of the in-progress char-level training.
   Reads the periodic checkpoint + history; prints round, loss, a
   short generation.  Designed to be invoked by the monitor every
   N minutes for a richer heartbeat than file-mtime polling. *)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Quiet @ Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Quiet @ Get["../../part-02-self-attention/src/Attention.wl"];
Quiet @ Get["../../part-03-transformer-block/src/Transformer.wl"];
Quiet @ Get["CharTokeniser.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
ckptPath = FileNameJoin[{dataDir, "charlevel_shakespeare_trained.wlnet"}];
histPath = FileNameJoin[{dataDir, "charlevel_training_history.wxf"}];

If[!FileExistsQ[ckptPath] || !FileExistsQ[histPath],
  Print["no checkpoint yet"]; Exit[0]];

hist = Import[histPath];
net  = Import[ckptPath];
fmt[x_, digits_] := ToString[NumberForm[N[x], {Infinity, digits}]];
Print["round=", hist["currentRound"],
      "  loss=", fmt[hist["currentLoss"], 3],
      "  ppl=",  fmt[Exp[N @ hist["currentLoss"]], 2],
      "  wall=", fmt[hist["wallSecondsSoFar"]/60., 1], "min"];

text = Import["../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt", "Text"];
vocab = LLMAtoZ`CharTokeniser`charVocabulary[text];
tokId = LLMAtoZ`CharTokeniser`charTokenToIdAssociation[vocab];
idTok = LLMAtoZ`CharTokeniser`charIdToTokenAssociation[vocab];

(* Short sample: 120 chars at T=0.8 *)
SeedRandom[Hash[hist["currentRound"]]];
gen = LLMAtoZ`CharTokeniser`encodeChars["ROMEO:", tokId];
Do[
  Module[{ctx, pos, l, p},
    ctx = If[Length[gen] >= 128, gen[[-128;;]], PadRight[gen, 128, 1]];
    pos = Min[Length[gen], 128];
    l = net[ctx][[pos]];
    p = Exp[(l - Max[l])/0.8]; p = p/Total[p];
    AppendTo[gen, RandomChoice[p -> Range[Length[p]]]]],
  {120}];
Print["sample: "];
Print[StringReplace[
  LLMAtoZ`CharTokeniser`decodeChars[gen, idTok],
  "\n" -> " | "]];
