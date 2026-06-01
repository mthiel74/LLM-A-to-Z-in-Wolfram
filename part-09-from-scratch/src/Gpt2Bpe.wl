(* ::Package:: *)

ClearAll[gpt2Encode, gpt2Decode, gpt2LoadBpeData];

$gpt2BpeLoaded = False;
$gpt2BpeDataPath = FileNameJoin[{DirectoryName[$InputFileName], "..", "data", "gpt2_bpe.json"}];
$gpt2PairSeparator = "\t";
$gpt2Pat = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

gpt2LoadBpeData[path_String : $gpt2BpeDataPath] := Module[
  {data, merges},
  If[TrueQ[$gpt2BpeLoaded] && path === $gpt2LoadedPath, Return[Null]];

  data = Import[path, "RawJSON"];
  If[!AssociationQ[data], Print["Could not load GPT-2 BPE JSON: ", path]; Abort[]];

  $gpt2BytesToUnicode = data["bytes_to_unicode"];
  $gpt2UnicodeToBytes = AssociationThread[$gpt2BytesToUnicode -> Range[0, 255]];

  merges = data["merge_rules"];
  $gpt2MergeRanks = AssociationThread[
    (#[[1]] <> $gpt2PairSeparator <> #[[2]] & /@ merges) -> Range[0, Length[merges] - 1]
  ];

  $gpt2Vocab = data["vocab"];
  $gpt2IdToToken = AssociationThread[Values[$gpt2Vocab] -> Keys[$gpt2Vocab]];

  $gpt2LoadedPath = path;
  $gpt2BpeLoaded = True;
  Null
];

gpt2ByteEncodeChunk[chunk_String] :=
  StringJoin[$gpt2BytesToUnicode[[# + 1]] & /@ ToCharacterCode[chunk, "UTF-8"]];

gpt2MergePair[tokens_List, pair_List] := Module[
  {out = {}, i = 1, n = Length[tokens]},
  While[i <= n,
    If[i < n && tokens[[i]] === pair[[1]] && tokens[[i + 1]] === pair[[2]],
      AppendTo[out, pair[[1]] <> pair[[2]]];
      i += 2,
      AppendTo[out, tokens[[i]]];
      i += 1
    ]
  ];
  out
];

gpt2BpeTokens[token_String] := Module[
  {parts = Characters[token], ranks, minRank, pos, pair},
  If[Length[parts] <= 1, Return[parts]];
  While[Length[parts] > 1,
    ranks = Lookup[
      $gpt2MergeRanks,
      (#[[1]] <> $gpt2PairSeparator <> #[[2]] & /@ Partition[parts, 2, 1]),
      Infinity
    ];
    minRank = Min[ranks];
    If[minRank === Infinity, Break[]];
    pos = First@FirstPosition[ranks, minRank];
    pair = parts[[{pos, pos + 1}]];
    parts = gpt2MergePair[parts, pair];
  ];
  parts
];

gpt2Encode[string_String] := Module[
  {chunks, tokens, ids},
  gpt2LoadBpeData[];
  chunks = StringCases[string, RegularExpression[$gpt2Pat]];
  tokens = Flatten[gpt2BpeTokens /@ (gpt2ByteEncodeChunk /@ chunks)];
  ids = Lookup[$gpt2Vocab, tokens, Missing["NotFound"]];
  If[!FreeQ[ids, _Missing], Print["Unknown GPT-2 BPE token encountered."]; Abort[]];
  ids
];

gpt2Decode[ids_List] := Module[
  {tokens, chars, bytes},
  gpt2LoadBpeData[];
  tokens = Lookup[$gpt2IdToToken, ids, Missing["NotFound"]];
  If[!FreeQ[tokens, _Missing], Print["Unknown GPT-2 token id encountered."]; Abort[]];
  chars = Characters[StringJoin[tokens]];
  bytes = Lookup[$gpt2UnicodeToBytes, chars, Missing["NotFound"]];
  If[!FreeQ[bytes, _Missing],
    If[ids === {50256}, Return["<|endoftext|>"]];
    Print["Token id sequence contains bytes outside the GPT-2 byte alphabet."];
    Abort[]
  ];
  FromCharacterCode[bytes, "UTF-8"]
];
