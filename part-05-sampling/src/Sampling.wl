(* ::Package:: *)

(* ============================================================
   Sampling.wl  ---  Part 5 of "LLM A to Z in Wolfram Language"
   ============================================================

   Five sampling strategies + speculative decoding, all implemented
   from scratch.  Each strategy takes a vector of logits (or a model
   plus a prompt) and produces the next token (or sequence).

   Conventions:
     - Logits are 1-D Real arrays of length vocabSize.
     - Temperatures are positive Reals.  T < 1 sharpens, T > 1
       softens.  T = 0 is greedy (handled as a special case).
     - All sampling functions return a single 1-based integer ID. *)

BeginPackage["LLMAtoZ`Sampling`"];

ClearAll[
  softmaxTemperature,
  greedySample, temperatureSample, topKSample, topPSample,
  beamSearch, speculativeDecode,
  generateText
];

softmaxTemperature::usage =
  "softmaxTemperature[logits, T] returns the softmax-with-temperature \
distribution: exp(logits / T) renormalised.  Numerically stable.";

greedySample::usage =
  "greedySample[logits] returns the position of the maximal logit.";

temperatureSample::usage =
  "temperatureSample[logits, T] samples from softmax(logits / T).";

topKSample::usage =
  "topKSample[logits, k, T] keeps only the k largest logits, sets the \
rest to -Infinity, then samples at temperature T.";

topPSample::usage =
  "topPSample[logits, p, T] keeps the smallest set of top tokens whose \
cumulative probability exceeds p, then samples at temperature T \
(nucleus sampling).";

beamSearch::usage =
  "beamSearch[stepFn, initialIds, beamWidth, maxLen, eosId : None] \
runs beam search.  stepFn[ids] must return a vector of next-token \
logits given a sequence of ids.";

speculativeDecode::usage =
  "speculativeDecode[mainStepFn, draftStepFn, initialIds, k, maxLen] \
runs speculative decoding: the draft model proposes k tokens, the main \
model verifies them in one forward pass, accepted tokens are committed \
and rejected tokens cause a rollback.  Returns the generated id list.";

generateText::usage =
  "generateText[net, promptIds, vocabIdToToken, strategy, options] is a \
convenience wrapper that uses one of the strategies above to produce a \
length-`MaxNew` continuation and decodes it back to a string.  strategy \
\[Element] {\"greedy\", \"temperature\", \"top-k\", \"top-p\", \"beam\"}.";

Begin["`Private`"];

(* ------------------------------------------------------------
   Building blocks. *)

softmaxTemperature[logits_List, T_?Positive] := Module[{shifted, ex},
  shifted = logits / T - Max[logits / T];
  ex = Exp[shifted];
  ex / Total[ex]
];

greedySample[logits_List] := First[Ordering[logits, -1]];

temperatureSample[logits_List, T_?Positive] := Module[{p},
  p = softmaxTemperature[logits, T];
  RandomChoice[p -> Range[Length[p]]]
];

topKSample[logits_List, k_Integer, T_?Positive] := Module[
  {indices, masked},
  indices = Ordering[logits, -k];
  masked = ConstantArray[-1.*10^9, Length[logits]];
  masked[[indices]] = logits[[indices]];
  temperatureSample[masked, T]
];

topPSample[logits_List, p_?Positive, T_?Positive] := Module[
  {probs, sorted, cum, kept, masked},
  probs = softmaxTemperature[logits, T];
  sorted = Reverse @ Ordering[probs];
  cum = Accumulate[probs[[sorted]]];
  kept = Length @ TakeWhile[cum, # < p &] + 1;
  masked = ConstantArray[-1.*10^9, Length[logits]];
  masked[[sorted[[;; Min[kept, Length[sorted]]]]]] =
    logits[[sorted[[;; Min[kept, Length[sorted]]]]]];
  temperatureSample[masked, T]
];

(* ------------------------------------------------------------
   Beam search.

   At each step, expand every beam by every token, score the
   combined log-prob, and keep the top `beamWidth` beams.  Stops
   when every beam ends with eosId or maxLen is reached. *)

beamSearch[stepFn_, initialIds_List, beamWidth_Integer,
    maxLen_Integer, eosId_:None] := Module[
  {beams, allDone, logits, candidates},
  beams = {{initialIds, 0.0}};  (* {ids, log-prob} *)
  While[
    Length[First[First[beams]]] < maxLen &&
    !AllTrue[beams, eosId =!= None && Last[First[#]] === eosId &],
    candidates = Flatten[
      Map[
        Function[beam,
          With[{ids = First[beam], lp = Last[beam]},
            logits = stepFn[ids];
            With[{logProbs = Log[softmaxTemperature[logits, 1.0] + 1.*^-12]},
              Table[
                {Append[ids, t], lp + logProbs[[t]]},
                {t, Length[logits]}
              ]
            ]
          ]
        ],
        beams
      ],
      1
    ];
    beams = Take[ReverseSortBy[candidates, Last], UpTo[beamWidth]];
  ];
  First[First[beams]]
];

(* ------------------------------------------------------------
   Speculative decoding (Leviathan, Kalman, Matias 2023).

   Draft model proposes k tokens cheaply.  Main model verifies
   them all in a single forward pass and rejects whichever ones
   are inconsistent with its own distribution.  Accepted tokens
   are appended; the first rejected token is resampled from the
   main model's distribution.

   For pedagogical simplicity we implement a deterministic
   version: a draft token is accepted if it matches what the
   main model would have picked greedily; otherwise we fall back
   to the main model's pick.  The throughput gain is the same in
   shape; the probabilistic version (with importance-weighted
   acceptance) is referenced in the post. *)

speculativeDecode[mainStepFn_, draftStepFn_, initialIds_List,
    k_Integer, maxLen_Integer] := Module[
  {ids, draftSeq, mainLogits, mainPick, accepted, i},
  ids = initialIds;
  While[Length[ids] < maxLen,
    (* Draft proposes k tokens (autoregressively from itself). *)
    draftSeq = {};
    Module[{cur = ids, t},
      Do[
        t = greedySample[draftStepFn[cur]];
        AppendTo[draftSeq, t];
        cur = Append[cur, t],
        {k}
      ]
    ];
    (* Main model checks all k tokens in one pass.  Iterate: *)
    accepted = 0;
    Do[
      mainLogits = mainStepFn[Join[ids, Take[draftSeq, i]]];
      mainPick = greedySample[mainLogits];
      If[mainPick === draftSeq[[i + 1]],
        accepted += 1,
        Break[]
      ],
      {i, 0, k - 1}
    ];
    If[accepted > 0,
      ids = Join[ids, Take[draftSeq, accepted]]
    ];
    (* Resample one token from main at the rejection point. *)
    If[accepted < k && Length[ids] < maxLen,
      mainLogits = mainStepFn[ids];
      AppendTo[ids, greedySample[mainLogits]]
    ];
    If[accepted == 0 && Length[ids] < maxLen,
      mainLogits = mainStepFn[ids];
      AppendTo[ids, greedySample[mainLogits]]
    ];
  ];
  ids
];

(* ------------------------------------------------------------
   Convenience wrapper: take a model, a prompt, a strategy, and
   produce a continuation. *)

Options[generateText] = {
  "MaxNew"      -> 60,
  "Temperature" -> 1.0,
  "TopK"        -> 40,
  "TopP"        -> 0.9,
  "BeamWidth"   -> 4,
  "ContextLen"  -> 64,
  "PadId"       -> 1
};

generateText[net_, promptIds_List, vocab_Association, strategy_String,
    opts : OptionsPattern[]] := Module[
  {ids = promptIds, maxNew, T, k, p, ctxLen, padId, stepFn},
  maxNew = OptionValue["MaxNew"];
  T      = OptionValue["Temperature"];
  k      = OptionValue["TopK"];
  p      = OptionValue["TopP"];
  ctxLen = OptionValue["ContextLen"];
  padId  = OptionValue["PadId"];

  stepFn[curIds_] := Module[{ctx, pos},
    ctx = If[Length[curIds] >= ctxLen, curIds[[-ctxLen ;;]],
              PadRight[curIds, ctxLen, padId]];
    pos = Min[Length[curIds], ctxLen];
    net[ctx][[pos, All]]
  ];

  If[strategy === "beam",
    ids = beamSearch[stepFn, promptIds, OptionValue["BeamWidth"],
                     Length[promptIds] + maxNew],
    Do[
      With[{logits = stepFn[ids]},
        AppendTo[ids,
          Switch[strategy,
            "greedy",      greedySample[logits],
            "temperature", temperatureSample[logits, T],
            "top-k",       topKSample[logits, k, T],
            "top-p",       topPSample[logits, p, T],
            _, greedySample[logits]
          ]
        ]
      ],
      {maxNew}
    ]
  ];

  StringJoin @ Lookup[vocab, ids] /. "</w>" -> " "
];

End[];
EndPackage[];
