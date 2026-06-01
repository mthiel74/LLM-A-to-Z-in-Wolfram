#!/usr/bin/env wolframscript
(* Chat with an SFT'd Part-9 nano model.

   Loads an instruction-tuned logit-head NetChain, formats each question with
   the SAME Alpaca template prepare_alpaca.py used, and generates the response
   autoregressively, STOPPING at the EOT token (GPT-2 50256 -> WL id 50257) the
   SFT taught it to emit.  Writes token ids to generations.json for
   sample_decode.py to detokenise.

   Questions come from data/questions.json {questions:[str]} (encoded to ids by
   encode_chat.py).  Greedy-ish top-k sampling. *)

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
(* optional CLI arg: model filename under data/models (default = v2 SFT) *)
modelName = First[$ScriptCommandLine[[2 ;;]], "nano30M_sft.wlnet"];
modelPath = FileNameJoin[{repo, "part-09-from-scratch", "data", "models", modelName}];
qFile = FileNameJoin[{repo, "part-09-from-scratch", "data", "questions_encoded.json"}];

net = Import[modelPath, "WLNet"];
maxLen = NetExtract[net, "Input"][[1]];

qData = Import[qFile, "RawJSON"];
questions = qData["questions"];
qIds = qData["ids"];           (* GPT-2 0-indexed ids of the FULL templated prompt *)

eotWL = 50257;                  (* GPT-2 50256 + 1 *)
nNew = 120; topK = 40; temp = 0.8;
SeedRandom[1];

sampleNext[logits_] := Module[{l = logits/temp, idx, probs},
  idx = Take[Ordering[l, -topK], -topK];
  probs = SoftmaxLayer[][l[[idx]]];
  RandomChoice[probs -> idx]];

answers = {};
Do[
  Module[{gen, fed, out, logits, nxt, resp},
    gen = (qIds[[q]] + 1);                 (* WL 1-indexed prompt *)
    If[Length[gen] == 0, gen = {1}];
    resp = {};
    Do[
      fed = Take[gen, -Min[Length[gen], maxLen]];
      fed = PadRight[fed, maxLen, 1];
      out = net[fed];
      logits = out[[Min[Length[gen], maxLen]]];
      nxt = sampleNext[logits];
      If[nxt === eotWL, Break[]];           (* learned stop *)
      AppendTo[gen, nxt]; AppendTo[resp, nxt],
      {nNew}];
    AppendTo[answers, (resp - 1)];          (* response only, GPT-2 0-indexed *)
  ],
  {q, Length[questions]}];

Export[FileNameJoin[{repo, "part-09-from-scratch", "data", "generations.json"}],
  <|"prompts" -> questions, "generations" -> answers|>, "JSON"];
Print["wrote ", Length[answers], " answers"];
