(* ::Package:: *)

(* ============================================================
   Tokeniser.wl  ---  Part 1 of "LLM A to Z in Wolfram Language"
   ============================================================

   A from-scratch byte-pair-encoding (BPE) tokeniser in pure WL,
   plus character- and word-level baselines for comparison, plus
   a small NetGraph that combines token and positional embeddings.

   No external libraries are used. The functions are deliberately
   written to be inspectable: every intermediate quantity is a
   plain WL expression (list, association, rule) the reader can
   print and pick apart.
*)

BeginPackage["LLMAtoZ`Tokeniser`"];

ClearAll[
  preTokenise, wordSymbolList, pairCounts, mergeOnce, trainBPE,
  applyMerges, bpeEncode, bpeDecode,
  charBaselineVocabulary, wordBaselineVocabulary,
  buildBPEVocabulary, tokenToIdAssociation, idToTokenAssociation,
  encodeWithIds, decodeFromIds,
  tokenPositionEmbedder, embedTokens,
  colouredTokens, mergeProgression, tokenZipf
];

preTokenise::usage =
  "preTokenise[text] splits a string into pre-tokens (words and \
runs of punctuation). The output is a flat list of strings.";

wordSymbolList::usage =
  "wordSymbolList[word] returns the initial symbol list for a \
pre-token: its characters followed by the special end-of-word \
marker \"</w>\".";

pairCounts::usage =
  "pairCounts[state] returns an Association mapping each adjacent \
symbol pair {a,b} to the total number of times it occurs in the \
corpus, weighted by word frequency. `state` is a list of \
{symbolList, frequency} pairs.";

mergeOnce::usage =
  "mergeOnce[state, pair] applies the merge rule pair -> StringJoin[pair] \
across every entry in the state, returning the updated state.";

trainBPE::usage =
  "trainBPE[text, numMerges] learns numMerges BPE merge rules from text. \
Returns <|\"MergeRules\" -> {...}, \"BaseAlphabet\" -> {...}|>. \
Pass Snapshots -> {n1, n2, ...} to additionally collect the corpus \
sequence length at the requested merge counts: \
<|..., \"Snapshots\" -> <|n -> seqLen|>|>.";

applyMerges::usage =
  "applyMerges[word, mergeRules] applies the learned merge rules to a \
single pre-token, returning a list of BPE token strings.";

bpeEncode::usage =
  "bpeEncode[text, mergeRules] tokenises text using the learned merge \
rules. Returns a flat list of BPE token strings.";

bpeDecode::usage =
  "bpeDecode[tokens] reconstructs the original text from a list of \
BPE token strings.";

charBaselineVocabulary::usage =
  "charBaselineVocabulary[text] returns the sorted character vocabulary.";

wordBaselineVocabulary::usage =
  "wordBaselineVocabulary[text] returns the sorted whitespace-split \
word vocabulary.";

buildBPEVocabulary::usage =
  "buildBPEVocabulary[text, trainedModel] returns the full BPE token \
vocabulary: base alphabet plus all merged tokens, plus the </w> marker.";

tokenToIdAssociation::usage =
  "tokenToIdAssociation[vocab] returns an Association mapping each \
token string to a 1-based integer ID suitable for EmbeddingLayer.";

idToTokenAssociation::usage =
  "idToTokenAssociation[vocab] returns the inverse map.";

encodeWithIds::usage =
  "encodeWithIds[text, mergeRules, tokToId] tokenises text and \
returns the list of integer IDs.";

decodeFromIds::usage =
  "decodeFromIds[ids, idToTok] reconstructs the text from integer IDs.";

tokenPositionEmbedder::usage =
  "tokenPositionEmbedder[vocabSize, dModel, maxSeqLen] returns a \
NetGraph that takes a sequence of token IDs at port \"tokens\" and a \
sequence of position IDs at port \"positions\" and outputs their \
elementwise sum: the input tensor a transformer expects.";

embedTokens::usage =
  "embedTokens[net, ids] is a convenience wrapper that generates the \
positions automatically and runs the embedder on the given token IDs.";

colouredTokens::usage =
  "colouredTokens[text, mergeRules] tokenises text and returns a Row \
of coloured boxes, one per BPE token, so the reader can see the \
token boundaries inline.";

mergeProgression::usage =
  "mergeProgression[word, mergeRules] returns the list of symbol-list \
states that arise when the merge rules are applied to the word one at \
a time. Useful for animating the BPE merge process on a single word.";

tokenZipf::usage =
  "tokenZipf[tokens] returns a sorted-descending list of token counts \
suitable for plotting the Zipf-shaped frequency curve.";

Begin["`Private`"];

(* ------------------------------------------------------------
   Pre-tokenisation.

   We split text into alphabetic words and runs of non-alphabetic
   characters (punctuation, digits, whitespace blocks).  This is
   coarser than GPT-2's byte-level pre-tokeniser but easier to
   read.  Whitespace itself is dropped here; word boundaries are
   later marked with the explicit "</w>" symbol inside each
   pre-token, so the information is not lost.
   ------------------------------------------------------------ *)

preTokenise[text_String] :=
  Select[
    StringSplit[text, RegularExpression["(\\s+|(?<=[^A-Za-z'])(?=[A-Za-z'])|(?<=[A-Za-z'])(?=[^A-Za-z']))"]],
    StringMatchQ[#, RegularExpression["\\S+"]] &
  ];

wordSymbolList[word_String] := Append[Characters[word], "</w>"];

(* ------------------------------------------------------------
   Pair counting and merging.

   `state` is a list of pairs {symbolList, frequency}.  We never
   keep the original word strings once we have built the state:
   all subsequent work is on symbol lists.
   ------------------------------------------------------------ *)

pairCounts[state_List] :=
  Merge[
    Flatten[
      Map[
        Function[entry,
          With[{syms = First[entry], freq = Last[entry]},
            (# -> freq & /@ Partition[syms, 2, 1])
          ]
        ],
        state
      ],
      1
    ],
    Total
  ];
(* Returns Association[{a,b} -> totalFrequency, ...].  The keys are
   2-element lists, which WL associations handle natively. *)

mergeOnce[state_List, pair : {a_String, b_String}] :=
  With[{joined = a <> b},
    Map[
      Function[entry,
        {SequenceReplace[First[entry], {a, b} -> joined], Last[entry]}
      ],
      state
    ]
  ];

(* trainBPE uses incremental pair-count maintenance.

   The naive approach (pairCounts + mergeOnce applied each iteration)
   is conceptually clean but quadratic-ish on Tiny Shakespeare: it
   rebuilds the entire pair-count table O(numMerges) times.

   For the actual training run we maintain two living indices:

     pc   : pair -> running total frequency
     p2w  : pair -> Association of word indices that currently contain it

   Each merge step then only visits the words that actually contained
   the merged pair, decrementing counts for the pairs broken by the
   merge and incrementing counts for the pairs created by it.  On
   Tiny Shakespeare with 1000 merges this turns ~minutes into ~seconds. *)

Options[trainBPE] = {Snapshots -> {}};
trainBPE[text_String, numMerges_Integer, OptionsPattern[]] := Module[
  {wordFreqs, wordKeys, syms, freqs, n,
   pc, p2w, mergeRules = {}, best, ab, touched,
   f, oldSyms, oldPairs, newSyms, newPairs,
   snapshotSet, snapshots = <||>, step, totalLen},
  snapshotSet = AssociationThread[OptionValue[Snapshots] -> True];

  wordFreqs = Counts[preTokenise[text]];
  wordKeys = Keys[wordFreqs];
  freqs    = Values[wordFreqs];
  n        = Length[wordKeys];
  syms     = wordSymbolList /@ wordKeys;

  pc  = <||>;
  p2w = <||>;
  Do[
    With[{f0 = freqs[[w]]},
      Do[
        With[{p = {syms[[w, i]], syms[[w, i + 1]]}},
          pc[p] = Lookup[pc, Key[p], 0] + f0;
          If[! KeyExistsQ[p2w, p], p2w[p] = <||>];
          p2w[p, w] = True
        ],
        {i, Length[syms[[w]]] - 1}
      ]
    ],
    {w, n}
  ];

  totalLen = Total[Length /@ syms * freqs];
  If[KeyExistsQ[snapshotSet, 0], snapshots[0] = totalLen];

  Do[
    If[Length[pc] == 0, Break[]];
    best = First @ Keys @ MaximalBy[pc, Identity, 1];
    AppendTo[mergeRules, best];
    ab = best[[1]] <> best[[2]];
    touched = Keys[p2w[best]];

    Do[
      f         = freqs[[w]];
      oldSyms   = syms[[w]];
      oldPairs  = Partition[oldSyms, 2, 1];
      newSyms   = SequenceReplace[oldSyms, {best[[1]], best[[2]]} -> ab];
      newPairs  = Partition[newSyms, 2, 1];
      syms[[w]] = newSyms;
      totalLen -= (Length[oldSyms] - Length[newSyms]) * f;

      Do[
        pc[p] = pc[p] - f;
        If[pc[p] <= 0, KeyDropFrom[pc, {p}]];
        If[KeyExistsQ[p2w, p],
          KeyDropFrom[p2w[p], w];
          If[Length[p2w[p]] == 0, KeyDropFrom[p2w, {p}]]
        ],
        {p, oldPairs}
      ];

      Do[
        pc[p] = Lookup[pc, Key[p], 0] + f;
        If[! KeyExistsQ[p2w, p], p2w[p] = <||>];
        p2w[p, w] = True,
        {p, newPairs}
      ],
      {w, touched}
    ];
    step = Length[mergeRules];
    If[KeyExistsQ[snapshotSet, step], snapshots[step] = totalLen],
    {numMerges}
  ];

  <|
    "MergeRules"   -> mergeRules,
    "BaseAlphabet" -> Union[Flatten[Characters /@ wordKeys]],
    "Snapshots"    -> KeySort[snapshots],
    "FinalEncoding" -> AssociationThread[wordKeys -> syms]
  |>
];

(* ------------------------------------------------------------
   Encoding new text with a learned BPE.
   ------------------------------------------------------------ *)

applyMerges[word_String, mergeRules_List] := Module[{syms},
  syms = wordSymbolList[word];
  Do[
    syms = SequenceReplace[syms, {rule[[1]], rule[[2]]} -> rule[[1]] <> rule[[2]]],
    {rule, mergeRules}
  ];
  syms
];

bpeEncode[text_String, mergeRules_List] :=
  Flatten[applyMerges[#, mergeRules] & /@ preTokenise[text]];

bpeDecode[tokens_List] := Module[{groups, words},
  (* Group consecutive tokens together until one ends with "</w>",
     which marks a word boundary.  Then strip the marker and join
     the remaining pieces of each group into a word. *)
  groups = Split[tokens, ! StringEndsQ[#1, "</w>"] &];
  words  = Map[StringReplace[StringJoin[#], "</w>" -> ""] &, groups];
  StringJoin[Riffle[words, " "]]
];
(* Decode is approximate by design: it recovers the words (each group
   ends at a token whose tail is "</w>") but discards the original
   whitespace and punctuation positions, which the BPE pre-tokeniser
   does not preserve. *)

(* ------------------------------------------------------------
   Baseline vocabularies (character- and word-level) for comparison.
   ------------------------------------------------------------ *)

charBaselineVocabulary[text_String] := Union[Characters[text]];

wordBaselineVocabulary[text_String] := Union[preTokenise[text]];

(* ------------------------------------------------------------
   Vocabulary assembly and token <-> ID maps.

   IDs are 1-based to match WL convention and EmbeddingLayer's
   "Class" port semantics.
   ------------------------------------------------------------ *)

buildBPEVocabulary[trainedModel_Association] :=
  DeleteDuplicates @ Join[
    trainedModel["BaseAlphabet"],
    {"</w>"},
    StringJoin /@ trainedModel["MergeRules"]
  ];

tokenToIdAssociation[vocab_List] :=
  AssociationThread[vocab -> Range[Length[vocab]]];

idToTokenAssociation[vocab_List] :=
  AssociationThread[Range[Length[vocab]] -> vocab];

encodeWithIds[text_String, mergeRules_List, tokToId_Association] :=
  Lookup[tokToId, bpeEncode[text, mergeRules]];

decodeFromIds[ids_List, idToTok_Association] :=
  bpeDecode[Lookup[idToTok, ids]];

(* ------------------------------------------------------------
   Token and positional embedding NetGraph.

   Two parallel EmbeddingLayers, one indexed by token ID and one
   indexed by position, summed elementwise.  This is the input
   tensor a transformer block consumes.  Positions are passed
   explicitly so the network has no maximum-length surprise:
   the caller supplies Range[seqLen] (1-based) as the position
   sequence.
   ------------------------------------------------------------ *)

tokenPositionEmbedder[vocabSize_Integer, dModel_Integer, maxSeqLen_Integer,
    opts : OptionsPattern[{RandomSeeding -> Automatic}]] :=
  NetInitialize[
    NetGraph[
      <|
        "tokenEmbed"    -> EmbeddingLayer[dModel, vocabSize],
        "positionEmbed" -> EmbeddingLayer[dModel, maxSeqLen],
        "sum"           -> ThreadingLayer[Plus]
      |>,
      {
        NetPort["tokens"]    -> "tokenEmbed",
        NetPort["positions"] -> "positionEmbed",
        {"tokenEmbed", "positionEmbed"} -> "sum"
      }
    ],
    RandomSeeding -> OptionValue[RandomSeeding]
  ];

embedTokens[net_NetGraph, ids_List] :=
  net[<|"tokens" -> ids, "positions" -> Range[Length[ids]]|>];

(* ------------------------------------------------------------
   Interactive visualisation helpers used by the notebook. *)

(* Stable per-token colour: hash token string to one of 12 palette
   entries.  Different tokens get different colours, same token
   always the same colour. *)
$tokenPalette = {
  RGBColor[0.85, 0.78, 0.61],  (* sand   *)
  RGBColor[0.61, 0.78, 0.78],  (* teal   *)
  RGBColor[0.93, 0.71, 0.61],  (* salmon *)
  RGBColor[0.74, 0.83, 0.65],  (* sage   *)
  RGBColor[0.85, 0.65, 0.80],  (* rose   *)
  RGBColor[0.68, 0.72, 0.85],  (* periwinkle *)
  RGBColor[0.88, 0.85, 0.55],  (* mustard *)
  RGBColor[0.70, 0.85, 0.78],  (* mint   *)
  RGBColor[0.85, 0.62, 0.55],  (* coral  *)
  RGBColor[0.62, 0.75, 0.85],  (* sky    *)
  RGBColor[0.78, 0.68, 0.85],  (* lilac  *)
  RGBColor[0.85, 0.78, 0.78]   (* dusty rose *)
};
tokenColour[tok_String] := $tokenPalette[[Mod[Hash[tok], Length[$tokenPalette]] + 1]];

colouredTokens[text_String, mergeRules_List] := Module[{tokens, display},
  tokens = bpeEncode[text, mergeRules];
  display = If[StringEndsQ[#, "</w>"],
                 Style[StringReplace[#, "</w>" -> "\[FilledSquare]"], Bold],
                 Style[#, Bold]] & /@ tokens;
  Row[
    MapThread[
      Framed[#1, Background -> #2, FrameMargins -> 2,
        FrameStyle -> Directive[Thin, GrayLevel[0.85]]] &,
      {display, tokenColour /@ tokens}
    ],
    " "
  ]
];

mergeProgression[word_String, mergeRules_List] := Module[{syms = wordSymbolList[word], states},
  states = {syms};
  Do[
    With[{newSyms = SequenceReplace[
            Last[states], {rule[[1]], rule[[2]]} -> rule[[1]] <> rule[[2]]]},
      If[newSyms =!= Last[states], AppendTo[states, newSyms]]
    ],
    {rule, mergeRules}
  ];
  states
];

tokenZipf[tokens_List] := ReverseSort[Values[Counts[tokens]]];

End[];
EndPackage[];
