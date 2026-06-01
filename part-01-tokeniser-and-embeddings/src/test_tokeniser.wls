#!/usr/bin/env wolframscript
(* Asserting smoke tests for Tokeniser.wl.
   Uses VerificationTest so a regression fails loudly. *)

SetDirectory[DirectoryName[$InputFileName]];
Get["Tokeniser.wl"];

Needs["MUnit`"] (* not strictly needed, but kept for future use *);

results = {};

push[name_String, test_TestResultObject] := AppendTo[results,
  <|"name" -> name, "outcome" -> test["Outcome"],
    "actual" -> test["ActualOutput"], "expected" -> test["ExpectedOutput"]|>
];

(* 1. preTokenise on punctuation *)
push["preTokenise drops whitespace, separates punctuation",
  VerificationTest[
    LLMAtoZ`Tokeniser`preTokenise["Hello, world!"],
    {"Hello", ",", "world", "!"}
  ]
];

(* 2. trainBPE on a toy corpus produces deterministic first merges *)
toy = "the cat sat on the mat. the cat ate the rat.";
trained = LLMAtoZ`Tokeniser`trainBPE[toy, 5];
push["trainBPE first merge is the highest-frequency adjacent pair",
  VerificationTest[
    Take[trained["MergeRules"], 1],
    {{"a", "t"}}
  ]
];

(* 3. bpeEncode "the" produces a single token at the right vocab size *)
encoded = LLMAtoZ`Tokeniser`bpeEncode["the", trained["MergeRules"]];
push["bpeEncode 'the' \[Rule] one token",
  VerificationTest[Length[encoded], 1]
];

(* 4. bpeDecode reverses bpeEncode for tokens in the trained corpus *)
push["bpeDecode round-trip on a trained word",
  VerificationTest[
    LLMAtoZ`Tokeniser`bpeDecode[
      LLMAtoZ`Tokeniser`bpeEncode["the cat", trained["MergeRules"]]],
    "the cat"
  ]
];

(* 5. embedder shape *)
vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[trained];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[vocab];
emb = LLMAtoZ`Tokeniser`tokenPositionEmbedder[Length[vocab], 16, 32, RandomSeeding -> 42];
ids = LLMAtoZ`Tokeniser`encodeWithIds["the cat", trained["MergeRules"], tokId];
out = LLMAtoZ`Tokeniser`embedTokens[emb, ids];
push["embedder shape: (seqLen, dModel)",
  VerificationTest[Dimensions[out], {Length[ids], 16}]
];

(* 6. colouredTokens returns a Row *)
ct = LLMAtoZ`Tokeniser`colouredTokens["the cat", trained["MergeRules"]];
push["colouredTokens returns Row",
  VerificationTest[Head[ct], Row]
];

(* 7. mergeProgression contains the initial character list as state 1 *)
mp = LLMAtoZ`Tokeniser`mergeProgression["the", trained["MergeRules"]];
push["mergeProgression starts at character list",
  VerificationTest[First[mp], {"t", "h", "e", "</w>"}]
];

(* 8. tokenZipf returns a sorted-descending list *)
zz = LLMAtoZ`Tokeniser`tokenZipf[LLMAtoZ`Tokeniser`bpeEncode[toy, trained["MergeRules"]]];
push["tokenZipf is sorted descending",
  VerificationTest[zz, ReverseSort[zz]]
];

(* Report *)
passed = Count[results, KeyValuePattern["outcome" -> "Success"]];
failed = Length[results] - passed;
Print["Passed: ", passed, "/", Length[results]];
If[failed > 0,
  Print["Failures:"];
  Do[
    If[r["outcome"] =!= "Success",
      Print["  ", r["name"], " :: actual=", r["actual"], " expected=", r["expected"]]
    ],
    {r, results}
  ];
  Exit[1]
];
Print["All tests passed."];
