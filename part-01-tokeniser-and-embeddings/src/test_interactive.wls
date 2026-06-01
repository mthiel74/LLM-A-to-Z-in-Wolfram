#!/usr/bin/env wolframscript
(* Verify the new interactive cells parse and execute (head check). *)

SetDirectory[ParentDirectory[DirectoryName[$InputFileName]]];
Get["src/Tokeniser.wl"];

Print["[T0] starting"];
text = Import["data/tinyshakespeare.txt", "Text"];
Print["[T1] loaded text ", StringLength[text]];

(* Use the cached trained model instead of retraining. *)
trained = Import["data/bpe_trained.mx"];
mergeRules = Take[trained["MergeRules"], 1000];
Print["[T2] loaded trained model, using first ", Length[mergeRules], " rules"];

(* coloured tokens *)
ct = LLMAtoZ`Tokeniser`colouredTokens[
  "To be or not to be, that is the question.", mergeRules];
Print["colouredTokens: ", Head[ct]];

(* merge progression *)
mp = LLMAtoZ`Tokeniser`mergeProgression["speaking", Take[mergeRules, 500]];
Print["mergeProgression length: ", Length[mp]];
Print["  first state: ", First[mp]];
Print["  last state:  ", Last[mp]];

(* zipf *)
zipf = LLMAtoZ`Tokeniser`tokenZipf[
  LLMAtoZ`Tokeniser`bpeEncode[StringTake[text, 50000], mergeRules]];
Print["zipf top 10: ", Take[zipf, 10]];
Print["zipf length: ", Length[zipf]];

(* Manipulate parses *)
expr = Manipulate[n, {n, 0, 100, 10}];
Print["Manipulate parse head: ", Head[expr]];

(* Animate parses *)
expr2 = Animate[Style[i, 20], {i, 1, 5, 1}];
Print["Animate parse head: ", Head[expr2]];

(* Build embedder and test the cosine widget plumbing *)
vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[trained];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[vocab];
embedder = LLMAtoZ`Tokeniser`tokenPositionEmbedder[
  Length[vocab], 64, 256, RandomSeeding -> 42];

tw = Normal[NetExtract[embedder, {"tokenEmbed", "Weights"}]];
candidates = Take[ReverseSort[Counts[LLMAtoZ`Tokeniser`bpeEncode[
  StringTake[text, 5000], mergeRules]]], UpTo[40]] // Keys;
Print["candidate tokens (first 8): ", Take[candidates, 8]];

cos = Function[{a, b}, (a . b) / (Norm[a] Norm[b])];
Print["cos sim of first two: ",
  cos[tw[[Lookup[tokId, candidates[[1]]]]],
       tw[[Lookup[tokId, candidates[[2]]]]]]];

(* Positional embedding ArrayPlot *)
posMat = Normal[NetExtract[embedder, {"positionEmbed", "Weights"}]];
Print["pos embed shape: ", Dimensions[posMat]];

Print["All interactive smoke tests passed."];
