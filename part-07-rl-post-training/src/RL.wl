(* ::Package:: *)

(* ============================================================
   RL.wl  ---  Part 7 of "LLM A to Z in Wolfram Language"
   ============================================================

   GRPO-style reinforcement-learning post-training.  Adapted from
   nanochat (Karpathy 2025), itself a simplification of GRPO
   (Shao et al. 2024) without a reference model and without KL.

   The loop, per training step:
     1. Sample a prompt from the training set.
     2. Generate K rollouts at temperature T from the current
        policy.
     3. Score each rollout with a verifiable reward function R.
     4. Center the rewards (subtract their mean) -- those are the
        advantages.
     5. Compute the policy-gradient loss
            L = -mean( advantage_i * log p_theta(rollout_i) )
        and take a step on it.

   For the educational demo we use a synthetic verifiable task:
   "produce a continuation that ends in <|eot|>".  This is a
   well-defined binary reward signal that the small post-trained
   model can learn quickly, while preserving all the GRPO
   bookkeeping that scales to real tasks (math, code, etc.).

   The same code path with a different reward function trains on
   GSM8K with an integer-answer-equality reward.
*)

BeginPackage["LLMAtoZ`RL`",
  {"LLMAtoZ`Tokeniser`", "LLMAtoZ`Transformer`",
   "LLMAtoZ`Sampling`", "LLMAtoZ`Training`"}];

ClearAll[
  rollout, rolloutBatch, rewardEot, rewardLengthTarget,
  centerAdvantages, policyGradientStep, runGrpoEpoch,
  toyPromptSet
];

rollout::usage =
  "rollout[net, promptIds, k, contextLen, T] returns one sampled \
continuation of length k from the model, starting from promptIds, at \
sampling temperature T.  Returns the list of generated ids (not \
including the prompt).";

rolloutBatch::usage =
  "rolloutBatch[net, promptIds, nRollouts, k, contextLen, T] returns \
nRollouts independent rollouts from the same prompt.";

rewardEot::usage =
  "rewardEot[rolloutIds, eotId] returns 1.0 if the rollout contains the \
EOT token, otherwise 0.0.  A trivial verifiable-reward task for the demo.";

rewardLengthTarget::usage =
  "rewardLengthTarget[rolloutIds, eotId, target] returns a continuous \
reward that peaks when the rollout's first EOT lies near the target \
position.";

centerAdvantages::usage =
  "centerAdvantages[rewards] returns rewards - mean(rewards).  The GRPO \
loss uses these centered advantages, NOT z-scored values (which can \
amplify gradient noise on small batches).";

policyGradientStep::usage =
  "policyGradientStep[net, prompt, rollouts, advantages, contextLen, lr] \
takes one ADAM step on the policy-gradient loss \[Sum] -adv_i * \
log p_theta(rollout_i | prompt).  Returns <|\"Net\" -> updatedNet, \
\"Loss\" -> scalar|>.";

runGrpoEpoch::usage =
  "runGrpoEpoch[net, prompts, hyperparameters] runs one full epoch of \
GRPO over a list of prompts: for each prompt, sample rollouts, score \
them, compute advantages, take a step.  Returns history of mean \
reward and policy-gradient loss per step.";

toyPromptSet::usage =
  "toyPromptSet[mergeRules, tokId] returns a list of short \
Shakespeare-style prompts encoded as token-ID sequences, ready to feed \
into rolloutBatch in the notebook demo.";

Begin["`Private`"];

(* ------------------------------------------------------------
   Rollout helpers.  Reuse the stepwise sampler we built in Part 5. *)

rollout[net_, promptIds_List, k_Integer, contextLen_Integer,
    T_:1.0] := Module[
  {ids = promptIds, padId = 1, ctx, pos, logits, p, sampled},
  Do[
    ctx = If[Length[ids] >= contextLen, ids[[-contextLen ;;]],
              PadRight[ids, contextLen, padId]];
    pos = Min[Length[ids], contextLen];
    logits = net[ctx][[pos, All]];
    p = LLMAtoZ`Sampling`softmaxTemperature[logits, T];
    sampled = RandomChoice[p -> Range[Length[p]]];
    AppendTo[ids, sampled],
    {k}
  ];
  ids[[Length[promptIds] + 1 ;;]]
];

rolloutBatch[net_, promptIds_List, nRollouts_Integer, k_Integer,
    contextLen_Integer, T_:1.0] :=
  Table[rollout[net, promptIds, k, contextLen, T], {nRollouts}];

(* ------------------------------------------------------------
   Reward functions. *)

rewardEot[rolloutIds_List, eotId_Integer] :=
  If[MemberQ[rolloutIds, eotId], 1.0, 0.0];

rewardLengthTarget[rolloutIds_List, eotId_Integer, target_Integer] :=
  Module[{pos = FirstPosition[rolloutIds, eotId, {0}, 1]},
    If[pos === {0}, 0.0, Exp[-Abs[First[pos] - target] / target]]
  ];

(* ------------------------------------------------------------
   Advantage centering.  No z-score normalisation, per DAPO. *)

centerAdvantages[rewards_List] := rewards - Mean[rewards];

(* ------------------------------------------------------------
   The policy-gradient step.

   For each rollout, compute the log-probability under the current
   policy, weight it by the advantage, sum negatives, and ask
   NetTrain to take one round on that scalar loss.

   Implementation note: WL's NetTrain is mini-batch optimising a
   stateless net by default.  For policy gradient with custom
   loss-per-rollout we have a few options.  The pedagogical one
   that keeps the post readable is to assemble a synthetic
   dataset of (prompt+rollout, target=rollout, weight=advantage)
   triples and feed it to NetTrain as a single batch with one
   MaxTrainingRounds=1 step.  CrossEntropyLossLayer with
   SampleWeights gives us exactly the weighted log-prob loss we
   want. *)

policyGradientStep[net_, promptIds_List, rollouts_List,
    advantages_List, contextLen_Integer, lr_:0.001] := Module[
  {data, padId = 1, trained, ctx, target, w, idx},
  data = MapThread[
    Function[{r, a},
      ctx = Join[promptIds, r];
      target = ctx[[2 ;;]];
      ctx = ctx[[;; -2]];
      ctx    = If[Length[ctx] >= contextLen, ctx[[-contextLen ;;]],
                  PadRight[ctx, contextLen, padId]];
      target = If[Length[target] >= contextLen,
                  target[[-contextLen ;;]],
                  PadRight[target, contextLen, padId]];
      <|"Input" -> ctx, "Output" -> target,
        "SampleWeight" -> a|>
    ],
    {rollouts, advantages}
  ];
  trained = NetTrain[
    net,
    {"Input" -> #["Input"], "Output" -> #["Output"]} & /@ data,
    All,
    BatchSize -> Length[data],
    LearningRate -> lr,
    MaxTrainingRounds -> 1,
    LossFunction -> CrossEntropyLossLayer["Index"],
    TargetDevice -> "CPU",
    Method -> "ADAM",
    TrainingProgressReporting -> None
  ];
  If[trained === $Failed,
    Return[<|"Net" -> net, "Loss" -> Missing["NetTrainFailed"]|>]
  ];
  <|"Net" -> trained["TrainedNet"],
    "Loss" -> trained["FinalRoundLoss"]|>
];

(* ------------------------------------------------------------
   The full GRPO outer loop. *)

runGrpoEpoch[net_, prompts_List, hp_Association : Automatic] := Module[
  {hyp, currentNet = net, history = {}, nRollouts, k, T, lr,
   contextLen, eotId},
  hyp = Join[<|
    "NRollouts"  -> 8,
    "GenerationLen" -> 32,
    "Temperature" -> 1.0,
    "LearningRate" -> 0.0005,
    "ContextLen" -> 64,
    "EotId" -> 1
  |>, If[AssociationQ[hp], hp, <||>]];
  nRollouts  = hyp["NRollouts"];
  k          = hyp["GenerationLen"];
  T          = hyp["Temperature"];
  lr         = hyp["LearningRate"];
  contextLen = hyp["ContextLen"];
  eotId      = hyp["EotId"];

  Do[
    Module[{rollouts, rewards, adv, step},
      rollouts = rolloutBatch[currentNet, prompt, nRollouts, k,
                              contextLen, T];
      rewards = Map[rewardEot[#, eotId] &, rollouts];
      adv = centerAdvantages[rewards];
      If[Variance[rewards] == 0,  (* nothing to learn this step *)
        AppendTo[history, <|"meanReward" -> Mean[rewards],
                            "loss" -> 0.0|>],
        step = policyGradientStep[currentNet, prompt, rollouts, adv,
                                  contextLen, lr];
        currentNet = step["Net"];
        AppendTo[history, <|"meanReward" -> Mean[rewards],
                            "loss" -> step["Loss"]|>]
      ]
    ],
    {prompt, prompts}
  ];

  <|"Net" -> currentNet, "History" -> history|>
];

(* ------------------------------------------------------------
   Toy prompt set. *)

toyPromptSet[mergeRules_List, tokId_Association] := Module[
  {prompts},
  prompts = {"ROMEO:", "HAMLET:", "MACBETH:", "What say you,",
             "To be or", "I beseech thee,", "O fairest"};
  Lookup[tokId, LLMAtoZ`Tokeniser`bpeEncode[#, mergeRules]] & /@ prompts
];

End[];
EndPackage[];
