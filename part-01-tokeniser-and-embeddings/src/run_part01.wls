#!/usr/bin/env wolframscript
(* End-to-end Part 1 build on Tiny Shakespeare.
   Trains BPE once (with snapshots), computes baselines, builds embedder,
   saves all intermediate artefacts and figures used by the notebook.
*)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];
Get["Tokeniser.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];

corpusPath = FileNameJoin[{dataDir, "tinyshakespeare.txt"}];
If[!FileExistsQ[corpusPath],
  Print["Tiny Shakespeare not cached; downloading..."];
  URLDownload[
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    corpusPath
  ];
  Print["Downloaded ", FileByteCount[corpusPath], " bytes to ", corpusPath];
];
text = Import[corpusPath, "Text"];
Print["Corpus: ", StringLength[text], " characters, ",
      Length[StringSplit[text]], " whitespace-words."];

(* ---------- Baselines ---------- *)
charVocab = LLMAtoZ`Tokeniser`charBaselineVocabulary[text];
wordVocab = LLMAtoZ`Tokeniser`wordBaselineVocabulary[text];
Print["Character vocab size: ", Length[charVocab]];
Print["Word vocab size:      ", Length[wordVocab]];

charSeqLen = StringLength[text];
wordSeqLen = Length[LLMAtoZ`Tokeniser`preTokenise[text]];
Print["Character seq length: ", charSeqLen];
Print["Pre-token seq length: ", wordSeqLen];

(* ---------- BPE: train once with snapshots over merge counts ---------- *)
mergeCounts = {0, 50, 100, 200, 500, 1000, 2000, 4000};
maxMerges = Max[mergeCounts];

Print["Training BPE up to ", maxMerges, " merges, snapshotting at ", mergeCounts];
{trainTime, trained} = AbsoluteTiming[
  LLMAtoZ`Tokeniser`trainBPE[text, maxMerges,
    LLMAtoZ`Tokeniser`Snapshots -> mergeCounts]
];
Print["Trained ", Length[trained["MergeRules"]],
      " merge rules in ", Round[trainTime, 0.1], " s."];

(* The Snapshots gives sequence length over UNIQUE pre-tokens with
   freq weighting -- this includes the </w> markers, so the curve
   compares apples to apples across all merge counts. *)
seqLenAt = trained["Snapshots"];
Print["Snapshot seq lengths: ", Normal[seqLenAt]];

results = Association @ KeyValueMap[
  #1 -> <|
    "VocabSize" -> Length[trained["BaseAlphabet"]] + 1 + #1,  (* base + </w> + merges *)
    "SeqLength" -> #2
  |> &,
  seqLenAt
];
Print["Curve table:"];
KeyValueMap[Print["  merges=", #1, "  vocab=", #2["VocabSize"],
                  "  seqLen=", #2["SeqLength"]] &, results];

(* Save the trained model and curve. *)
Export[FileNameJoin[{dataDir, "bpe_trained.mx"}], trained];
Export[FileNameJoin[{dataDir, "vocab_seqlen_sweep.json"}],
  KeyValueMap[<|"merges" -> #1, "vocab" -> #2["VocabSize"],
                "seqLen" -> #2["SeqLength"]|> &, results]];

(* ---------- Vocabulary-vs-sequence-length figure ---------- *)
points = KeyValueMap[{#2["VocabSize"], #2["SeqLength"]} &, results];
labelledPts = MapThread[Labeled[#1, #2, Above] &,
  {points, "n=" <> ToString[#] & /@ Keys[results]}];

(* Char baseline: vocab=64, seqLen = corpus chars.  Word baseline:
   vocab=14k+, seqLen = pre-token count.  Both expressed in the
   same units as the BPE points (token count over the corpus). *)
charPt = {Length[charVocab], charSeqLen};
wordPt = {Length[wordVocab], wordSeqLen};

curveFig = ListLogLogPlot[
  {labelledPts, {Labeled[charPt, "characters", Below]},
   {Labeled[wordPt, "words", Above]}},
  PlotLegends -> {"BPE (varying merges)", "character-level", "word-level"},
  Joined -> {True, False, False},
  PlotMarkers -> {Automatic, Style["\[FilledSquare]", 20],
                  Style["\[FilledDiamond]", 20]},
  PlotStyle -> {
    Directive[Thick, ColorData[97][1]],
    Directive[PointSize[0.025], ColorData[97][2]],
    Directive[PointSize[0.025], ColorData[97][3]]
  },
  Frame -> True,
  FrameLabel -> {"Vocabulary size", "Sequence length over Tiny Shakespeare"},
  GridLines -> Automatic,
  ImageSize -> 620,
  PlotLabel -> "Vocabulary size vs sequence length (log\[Hyphen]log)"
];
Export[FileNameJoin[{figDir, "vocab_vs_seqlen.png"}], curveFig,
       ImageResolution -> 144];
Print["Saved vocab_vs_seqlen.png"];

(* ---------- Sample BPE output snippet ---------- *)
sample = "To be or not to be, that is the question.";
mergeRules = trained["MergeRules"];

(* For sample sentences we use the cached final encoding for known pre-tokens,
   falling back to applyMerges for novel pre-tokens. *)
encodeQuick[textIn_, finalEnc_Association, rules_] :=
  Flatten[
    Lookup[finalEnc, #, LLMAtoZ`Tokeniser`applyMerges[#, rules]] & /@
      LLMAtoZ`Tokeniser`preTokenise[textIn]
  ];

finalEnc = trained["FinalEncoding"];
sampleTokens = encodeQuick[sample, finalEnc, mergeRules];
Print["Sample text:   ", sample];
Print["Sample tokens: ", sampleTokens];

(* ---------- Vocabulary and ID maps ---------- *)
finalVocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[trained];
Print["Full BPE vocab size at 4000 merges: ", Length[finalVocab]];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[finalVocab];
idTok = LLMAtoZ`Tokeniser`idToTokenAssociation[finalVocab];

(* ---------- Embedder NetGraph ---------- *)
dModel = 64;
maxLen = 256;
embedder = LLMAtoZ`Tokeniser`tokenPositionEmbedder[
  Length[finalVocab], dModel, maxLen, RandomSeeding -> 42];

promptIds = Lookup[tokId, sampleTokens];
Print["Prompt IDs (", Length[promptIds], " tokens): ", promptIds];

embedded = LLMAtoZ`Tokeniser`embedTokens[embedder, promptIds];
Print["Embedded shape: ", Dimensions[embedded]];

Export[FileNameJoin[{dataDir, "embedder.wlnet"}], embedder];
Export[FileNameJoin[{dataDir, "vocab.json"}], finalVocab];

(* ---------- Random-init embedding geometry ---------- *)
tokenWeights = Normal @ NetExtract[embedder, {"tokenEmbed", "Weights"}];
Print["Token weights matrix shape: ", Dimensions[tokenWeights]];

(* Pick eight common BPE tokens by encoding a chunk of the corpus and tallying. *)
chunk = StringTake[text, 50000];
chunkTokens = encodeQuick[chunk, finalEnc, mergeRules];
freqTokens = TakeLargestBy[Counts[chunkTokens], # &, 12];
chosen = Keys[freqTokens][[1 ;; 8]];
Print["Chosen tokens: ", chosen];

chosenIds = Lookup[tokId, chosen];
chosenVecs = tokenWeights[[chosenIds]];
cosines = Outer[
  (#1 . #2) / (Norm[#1] Norm[#2]) &,
  chosenVecs, chosenVecs, 1
];

geomFig = MatrixPlot[cosines,
  FrameTicks -> {
    {Transpose[{Range[Length[chosen]], chosen}], None},
    {None, Transpose[{Range[Length[chosen]], chosen}]}
  },
  ColorFunction -> "RedBlueTones",
  ColorFunctionScaling -> True,
  PlotLegends -> Automatic,
  PlotLabel -> "Cosine similarity among 8 common tokens at random init",
  ImageSize -> 520
];
Export[FileNameJoin[{figDir, "embedding_geometry_random.png"}], geomFig,
       ImageResolution -> 144];
Print["Saved embedding_geometry_random.png"];

(* PCA 2D projection. *)
pca = Normal @ PrincipalComponents[tokenWeights];
pcaProj = pca[[All, 1 ;; 2]];
chosenProj = pcaProj[[chosenIds]];

geomFig2 = Show[
  ListPlot[pcaProj,
    PlotStyle -> Directive[GrayLevel[0.6], PointSize[0.004]],
    ImageSize -> 520, AspectRatio -> 1,
    Frame -> True, FrameLabel -> {"PC1", "PC2"},
    PlotLabel -> "Token embeddings, PCA(2) -- random init"
  ],
  Graphics[{
    Red, PointSize[0.018],
    Point /@ chosenProj,
    Black,
    MapThread[Text[Style[#1, 12, Bold],
                   #2 + {0, 0.05 Mean[Abs[pcaProj[[All, 2]]]]}] &,
              {chosen, chosenProj}]
  }]
];
Export[FileNameJoin[{figDir, "embedding_pca_random.png"}], geomFig2,
       ImageResolution -> 144];
Print["Saved embedding_pca_random.png"];

(* ---------- Summary JSON the notebook can quote ---------- *)
summary = <|
  "corpusChars" -> StringLength[text],
  "charVocab" -> Length[charVocab],
  "wordVocab" -> Length[wordVocab],
  "charSeqLen" -> charSeqLen,
  "wordSeqLen" -> wordSeqLen,
  "bpeMaxMerges" -> maxMerges,
  "bpeTrainSeconds" -> trainTime,
  "bpeFinalVocab" -> Length[finalVocab],
  "samplePrompt" -> sample,
  "samplePromptTokens" -> sampleTokens,
  "embedderDModel" -> dModel,
  "embedderMaxLen" -> maxLen,
  "curve" -> KeyValueMap[
    <|"merges" -> #1, "vocab" -> #2["VocabSize"],
      "seqLen" -> #2["SeqLength"]|> &, results]
|>;
Export[FileNameJoin[{dataDir, "part01_summary.json"}], summary];
Print["Wrote part01_summary.json"];

Print["DONE."];
