#!/usr/bin/env wolframscript
SetDirectory[DirectoryName[$InputFileName]];
partDir = ParentDirectory[];
nbDir = FileNameJoin[{partDir, "notebook"}];
If[!DirectoryQ[nbDir], CreateDirectory[nbDir]];

title[s_]      := Cell[s, "Title"];
subtitle[s_]   := Cell[s, "Subtitle"];
section[s_]    := Cell[s, "Section"];
text[s_String] := Cell[s, "Text"];
text[parts_List] := Cell[TextData[parts], "Text"];
input[s_String] := Cell[s, "Input", FormatType -> InputForm];
math[box_] := Cell[BoxData[box], "Text", FontSize -> 14];

cells = Flatten @ {
  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["Part 7: Reinforcement learning post-training (GRPO)"],

  text[{
    "Supervised fine-tuning in Part 6 made the model do question-and-",
    "answer. RL post-training is the next step: take the SFT model, sample ",
    "completions on a task with a verifiable reward, and update the policy ",
    "to make high-reward completions more likely. Following nanochat, we ",
    "use ",
    StyleBox["GRPO", FontWeight -> Bold],
    " in its simplest form \[LongDash] no reference model, no KL term, ",
    "centered advantages over a group of rollouts."
  }],

  section["1.  Setup"],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl\"]
Get[\"../../part-02-self-attention/src/Attention.wl\"]
Get[\"../../part-03-transformer-block/src/Transformer.wl\"]
Get[\"../../part-04-pretraining/src/Training.wl\"]
Get[\"../../part-05-sampling/src/Sampling.wl\"]
Get[\"src/RL.wl\"]"],

  section["2.  The GRPO recipe in five lines"],
  text[{
    "Per training step, given a prompt:"
  }],
  text[{
    "1. ", StyleBox["Sample", FontWeight -> Bold],
    " ", math["K"], " rollouts at temperature ",
    math["T"], " from the current policy.\n",
    "2. ", StyleBox["Score", FontWeight -> Bold],
    " each rollout with a reward function ",
    math["R"], " (binary for verifiable tasks).\n",
    "3. ", StyleBox["Center", FontWeight -> Bold],
    " the rewards: advantages = rewards \[LongDash] mean(rewards).\n",
    "4. ", StyleBox["Compute", FontWeight -> Bold],
    " policy-gradient loss ",
    math[RowBox[{"L", "=", "-", "\[Sum]"}]], " advantage_i \[CenterDot] log p_\[Theta](rollout_i).\n",
    "5. ", StyleBox["Step", FontWeight -> Bold],
    " ADAM on L."
  }],
  text[{
    "What's deliberately absent: no reference model, no KL divergence, ",
    "no value head, no importance-weighting. The reward IS the ground ",
    "truth, the variance across the K rollouts IS the advantage signal. ",
    "It is shocking how simple this is once you accept that you don't ",
    "need PPO's machinery for tasks with verifiable correctness."
  }],

  section["3.  A verifiable toy task: end with <|eot|>"],
  text[{
    "For the educational demo we use the trivial task ",
    StyleBox["\"end your continuation with the <|eot|> token within k tokens\"",
            FontSlant -> Italic],
    ". Binary reward: 1.0 if the rollout contains <|eot|>, else 0.0. ",
    "Small models post-trained from random init learn to do this in a few ",
    "minutes of CPU time, while preserving all the GRPO bookkeeping that ",
    "scales to real verifiable tasks (GSM8K, code, format-following)."
  }],

  section["4.  One full GRPO step, walked through"],
  input["(* Pretend we have the SFT model from Part 6.  For the demo we
     just use a fresh small model so the notebook runs standalone. *)
SeedRandom[1];
bpe = Import[\"../../part-01-tokeniser-and-embeddings/data/bpe_trained.mx\"];
baseVocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[bpe];
chatVocab = Join[baseVocab, {\"<|user|>\", \"<|assistant|>\", \"<|eot|>\"}];
tokId = AssociationThread[chatVocab -> Range[Length[chatVocab]]];
net = LLMAtoZ`Transformer`gptModel[64, 4, 2, 128, Length[chatVocab], 64];"],
  input["promptIds = Lookup[tokId,
  LLMAtoZ`Tokeniser`bpeEncode[\"ROMEO:\", bpe[\"MergeRules\"]]];
rollouts = LLMAtoZ`RL`rolloutBatch[net, promptIds, 8, 24, 64, 1.0];
rewards = Map[LLMAtoZ`RL`rewardEot[#, tokId[\"<|eot|>\"]] &, rollouts];
{Length[rollouts], rewards}"],
  input["advantages = LLMAtoZ`RL`centerAdvantages[rewards];
advantages"],

  section["5.  Run a small GRPO epoch"],
  input["prompts = LLMAtoZ`RL`toyPromptSet[bpe[\"MergeRules\"], tokId];
hp = <|\"NRollouts\" -> 6, \"GenerationLen\" -> 20,
       \"Temperature\" -> 1.2, \"LearningRate\" -> 0.001,
       \"ContextLen\" -> 64, \"EotId\" -> tokId[\"<|eot|>\"]|>;
res = LLMAtoZ`RL`runGrpoEpoch[net, prompts, hp];
Length[res[\"History\"]]"],
  input["ListLinePlot[
  res[\"History\"][[All, \"meanReward\"]],
  Frame -> True,
  FrameLabel -> {\"step\", \"mean reward over the rollout group\"},
  PlotLabel -> \"GRPO learning curve on the EOT-ending task\",
  PlotStyle -> Directive[ColorData[97][1], Thick],
  ImageSize -> 540, GridLines -> Automatic]"],

  section["6.  Summary"],
  text[{
    "\[FilledSquare] ",
    StyleBox["A full GRPO loop in less than 100 lines.", FontWeight -> Bold],
    " ",
    StyleBox["rolloutBatch", FontFamily -> "Courier"], ", ",
    StyleBox["centerAdvantages", FontFamily -> "Courier"], ", ",
    StyleBox["policyGradientStep", FontFamily -> "Courier"], ", ",
    StyleBox["runGrpoEpoch", FontFamily -> "Courier"], " \[LongDash] no ",
    "reference model, no KL term, no value head, no PPO clipping.\n",
    "\[FilledSquare] ",
    StyleBox["A trivial verifiable task", FontWeight -> Bold],
    " (end-with-EOT) that exercises every step of the loop while ",
    "finishing in minutes of CPU time. The same code with a different ",
    StyleBox["rewardFn", FontFamily -> "Courier"],
    " trains on GSM8K \[LongDash] swap one function."
  }],
  text[{
    "In ",
    StyleBox["Part 8", FontWeight -> Bold],
    " we put the post-trained model behind an interactive chat widget ",
    "inside the notebook \[LongDash] the final artefact of the series."
  }],

  section["References"],
  text[{
    "Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. (GRPO.)\n",
    "Yu, F. et al. (2024). DAPO: An Open-Source LLM Reinforcement Learning System at Scale.\n",
    "Karpathy, A. (2025). nanochat."
  }]
};
nb = Notebook[cells, WindowSize -> {900, 750}, Saveable -> True];
out = FileNameJoin[{nbDir, "Part07.nb"}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes)"];
