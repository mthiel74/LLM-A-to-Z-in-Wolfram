#!/usr/bin/env wolframscript

scriptDir = DirectoryName[$InputFileName];
repo = ExpandFileName[FileNameJoin[{scriptDir, "..", ".."}]];

Get[FileNameJoin[{scriptDir, "Gpt2Bpe.wl"}]];

cmdLine = If[ListQ[$ScriptCommandLine], $ScriptCommandLine, {}];
args = If[Length[cmdLine] >= 2, Rest[cmdLine], {}];
defaultModelPath = FileNameJoin[{repo, "part-09-from-scratch", "data", "models", "nano30M_v3_sft.wlnet"}];

resolveModelPath[arg_String] := Which[
  FileExistsQ[arg], arg,
  FileExistsQ[FileNameJoin[{repo, "part-09-from-scratch", "data", "models", arg}]],
    FileNameJoin[{repo, "part-09-from-scratch", "data", "models", arg}],
  True, arg
];

hasModelArg = TrueQ[Length[args] > 0 && (
  StringEndsQ[First[args], ".wlnet"] ||
  FileExistsQ[First[args]] ||
  FileExistsQ[FileNameJoin[{repo, "part-09-from-scratch", "data", "models", First[args]}]]
)];

modelPath = If[hasModelArg, resolveModelPath[First[args]], defaultModelPath];
questionArgs = If[hasModelArg, Rest[args], args];
oneShotQuestion = If[Length[questionArgs] > 0, StringRiffle[questionArgs, " "], None];

If[!FileExistsQ[modelPath],
  Print["Model not found: ", modelPath];
  Exit[1]
];

net = Import[modelPath, "WLNet"];
maxLen = NetExtract[net, "Input"][[1]];

eotWL = 50257;
padWL = 1;
maxNew = 200;
topK = 40;
temperature = 0.8;
SeedRandom[];

alpacaPrompt[question_String] :=
  "Below is an instruction that describes a task. Write a response that completes the request.\n\n" <>
  "### Instruction:\n" <> question <> "\n\n### Response:\n";

sampleNext[logits_] := Module[{l = logits/temperature, idx, probs},
  idx = Take[Ordering[l, -Min[topK, Length[l]]], -Min[topK, Length[l]]];
  probs = SoftmaxLayer[][l[[idx]]];
  RandomChoice[probs -> idx]
];

generateResponse[question_String] := Module[
  {promptIds, gen, response = {}, fed, out, logits, nxt, pos},
  promptIds = gpt2Encode[alpacaPrompt[question]] + 1;
  If[Length[promptIds] == 0, promptIds = {padWL}];
  gen = promptIds;

  Do[
    fed = Take[gen, -Min[Length[gen], maxLen]];
    pos = Length[fed];
    fed = PadRight[fed, maxLen, padWL];
    out = net[fed];
    logits = out[[pos]];
    nxt = sampleNext[logits];
    If[nxt === eotWL, Break[]];
    AppendTo[gen, nxt];
    AppendTo[response, nxt],
    {maxNew}
  ];

  gpt2Decode[response - 1]
];

answerQuestion[question_String] := Module[{answer},
  answer = generateResponse[question];
  Print[answer];
];

If[StringQ[oneShotQuestion],
  answerQuestion[oneShotQuestion],
  (* Interactive loop. InputString reads a line from stdin in wolframscript and
     returns EndOfFile at end of input (or $Failed under some front ends);
     guard against both so a piped/redirected stdin cannot spin forever. *)
  Module[{line},
    While[True,
      line = Quiet @ Check[InputString["Question> "], EndOfFile];
      If[line === EndOfFile || line === $EndOfFile || line === $Failed ||
         ! StringQ[line] || StringLength[StringTrim[line]] == 0, Break[]];
      answerQuestion[StringTrim[line]]
    ]
  ]
];
