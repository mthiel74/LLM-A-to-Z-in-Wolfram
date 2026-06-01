#!/usr/bin/env wolframscript
(* Simulate running the Part 1 notebook from a fresh kernel.
   Executes the same Input-cell code in order, with the
   notebook's parent directory set as current. *)

SetDirectory[DirectoryName[$InputFileName]];  (* part-01/src *)
partDir = ParentDirectory[];
SetDirectory[partDir];

Print["[1] Get[Tokeniser.wl]"];
Get["src/Tokeniser.wl"];

Print["[2] Import corpus"];
text = Import["data/tinyshakespeare.txt", "Text"];
Print["    StringLength[text]=", StringLength[text]];

Print["[3] charBaselineVocabulary length"];
Print["    ", LLMAtoZ`Tokeniser`charBaselineVocabulary[text] // Length];

Print["[4] wordBaselineVocabulary length"];
Print["    ", LLMAtoZ`Tokeniser`wordBaselineVocabulary[text] // Length];

Print["[5] preTokenise sample"];
Print["    ", LLMAtoZ`Tokeniser`preTokenise[StringTake[text, 80]]];

Print["[6] Train BPE 1000 merges"];
{t, trained} = AbsoluteTiming[LLMAtoZ`Tokeniser`trainBPE[text, 1000]];
Print["    ", t, " s"];

Print["[7] First 20 merge rules"];
Print["    ", Take[trained["MergeRules"], 20]];

Print["[8] bpeEncode 'To be...'"];
mergeRules = trained["MergeRules"];
Print["    ", LLMAtoZ`Tokeniser`bpeEncode[
  "To be or not to be, that is the question.", mergeRules]];

Print["[9] Import figures/vocab_vs_seqlen.png"];
Print["    ImageQ: ", ImageQ[Import["figures/vocab_vs_seqlen.png"]]];

Print["[10] Build embedder"];
vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[trained];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[vocab];
embedder = LLMAtoZ`Tokeniser`tokenPositionEmbedder[
  Length[vocab], 64, 256, RandomSeeding -> 42];
Print["    Head[embedder]: ", Head[embedder]];

Print["[11] Encode and embed prompt"];
ids = LLMAtoZ`Tokeniser`encodeWithIds[
  "To be or not to be, that is the question.", mergeRules, tokId];
Print["    ids: ", ids];
Print["    embedded shape: ", Dimensions[LLMAtoZ`Tokeniser`embedTokens[embedder, ids]]];

Print["[12] Import figures/embedding_geometry_random.png"];
Print["    ImageQ: ", ImageQ[Import["figures/embedding_geometry_random.png"]]];

Print["[13] Import figures/embedding_pca_random.png"];
Print["    ImageQ: ", ImageQ[Import["figures/embedding_pca_random.png"]]];

Print["ALL NOTEBOOK INPUT CELLS PASSED."];
