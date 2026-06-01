#!/usr/bin/env wolframscript
(* Watchdog that runs alongside the char-level training.

   Every minute, polls the periodic-checkpoint file.  When a new
   round number is detected in the history file, loads the
   checkpoint, generates short samples at three temperatures from a
   fixed set of prompts, and appends them to a running log:

       data/charlevel_progress_log.wxf

   This is the data the Part 4 post will use to show how the model's
   output evolves as training progresses.
*)

$HistoryLength = 0;
SetDirectory[DirectoryName[$InputFileName]];

Get["../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl"];
Get["../../part-02-self-attention/src/Attention.wl"];
Get["../../part-03-transformer-block/src/Transformer.wl"];
Get["CharTokeniser.wl"];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];

ckptPath    = FileNameJoin[{dataDir, "charlevel_shakespeare_trained.wlnet"}];
historyPath = FileNameJoin[{dataDir, "charlevel_training_history.wxf"}];
logPath     = FileNameJoin[{dataDir, "charlevel_progress_log.wxf"}];

text = Import["../../part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt", "Text"];
vocab = LLMAtoZ`CharTokeniser`charVocabulary[text];
tokId = LLMAtoZ`CharTokeniser`charTokenToIdAssociation[vocab];
idTok = LLMAtoZ`CharTokeniser`charIdToTokenAssociation[vocab];

prompts = {"ROMEO:", "HAMLET:\n", "First Citizen:\n"};
temperatures = {0.5, 0.8, 1.0};
nGenerate = 200;

trainingAlive[] := Module[{out},
  out = RunProcess[{"pgrep", "-f", "run_charlevel.wls"},
                   "StandardOutput"];
  StringLength[StringTrim[out]] > 0
];

generate[net_, prompt_String, n_, T_] := Module[
  {pIds = LLMAtoZ`CharTokeniser`encodeChars[prompt, tokId],
   gen, ctx, pos, logits, p, sampled},
  gen = pIds;
  Do[
    ctx = If[Length[gen] >= 128, gen[[-128 ;;]], PadRight[gen, 128, 1]];
    pos = Min[Length[gen], 128];
    logits = net[ctx][[pos]];
    p = Exp[(logits - Max[logits])/T]; p = p/Total[p];
    sampled = RandomChoice[p -> Range[Length[p]]];
    AppendTo[gen, sampled],
    {n}];
  LLMAtoZ`CharTokeniser`decodeChars[gen, idTok]
];

snapshotCheckpoint[] := Module[
  {net, hist, round, loss, secs, samples},
  net = Quiet[Import[ckptPath]];
  hist = Quiet[Import[historyPath]];
  If[Head[net] =!= NetChain || !AssociationQ[hist], Return[$Failed]];
  round = Lookup[hist, "currentRound", -1];
  loss  = Lookup[hist, "currentLoss", Missing[]];
  secs  = Lookup[hist, "wallSecondsSoFar", Missing[]];
  samples = Association @ Table[
    p -> Association @ Table[
      ("T=" <> ToString[T]) -> generate[net, p, nGenerate, T],
      {T, temperatures}],
    {p, prompts}];
  <|
    "round"           -> round,
    "loss"            -> loss,
    "wallSeconds"     -> secs,
    "wallClockUTC"    -> DateString["ISODateTime"],
    "samples"         -> samples
  |>
];

existingLog = If[FileExistsQ[logPath], Import[logPath], <||>];
log = If[AssociationQ[existingLog], existingLog, <||>];
Print["Existing log entries: ", Length[log]];

SeedRandom[10];
Print["Watchdog start: ", DateString[]];

lastRound = -1;
While[True,
  If[!trainingAlive[],
    Print[DateString["Time"], " training gone, exiting."];
    Break[]
  ];
  If[FileExistsQ[ckptPath] && FileExistsQ[historyPath],
    Module[{hist},
      hist = Quiet[Import[historyPath]];
      If[AssociationQ[hist] && KeyExistsQ[hist, "currentRound"],
        If[hist["currentRound"] =!= lastRound,
          Module[{snap},
            snap = snapshotCheckpoint[];
            If[AssociationQ[snap],
              log["r" <> ToString[snap["round"]]] = snap;
              Export[logPath, log];
              lastRound = snap["round"];
              Print[DateString["Time"], " round=", snap["round"],
                    " loss=", N @ snap["loss"]]
            ]
          ]
        ]
      ]
    ]
  ];
  Pause[60];
];

Print["Watchdog done."];
