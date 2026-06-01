#!/usr/bin/env wolframscript
(* Autoregressive top-k sampler for the Part-9 nano-30M model.

   The model is a logit-head NetChain: length-`maxLen` integer input (token ids
   in WL's 1-indexed space, i.e. GPT-2 id + 1) -> (maxLen, vocab) logits.

   We seed with a BPE-encoded prompt (produced by encode_prompts.py), generate
   `nNew` tokens by repeatedly feeding the last `maxLen` tokens and sampling the
   next-token distribution at the last filled position, then write the GPT-2 ids
   (un-shifted back to 0-indexed) to JSON for sample_decode.py to detokenise. *)

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
modelPath = FileNameJoin[{repo, "part-09-from-scratch", "data", "models", "nano30M_v2.wlnet"}];
promptFile = FileNameJoin[{repo, "part-09-from-scratch", "data", "prompts_encoded.json"}];

net = Import[modelPath, "WLNet"];
maxLen = NetExtract[net, "Input"][[1]];   (* 512 *)
Print["model maxLen=", maxLen];

promptData = Import[promptFile, "RawJSON"];  (* {prompts:[str], ids:[[int(0-idx)]]} *)
prompts = promptData["prompts"];
promptIds = promptData["ids"];

nNew = 60; topK = 40; temp = 0.9;
SeedRandom[7];

sampleNext[logits_] := Module[{l = logits/temp, idx, probs, choice},
  idx = Take[Ordering[l, -topK], -topK];     (* indices (1-based) of top-k logits *)
  probs = SoftmaxLayer[][l[[idx]]];
  choice = RandomChoice[probs -> idx];
  choice];                                     (* 1-based WL index *)

generations = {};
Do[
  Module[{ctx, gen, fed, out, logits, nxt},
    (* ctx: WL 1-indexed ids (gpt2 id + 1) *)
    ctx = (promptIds[[p]] + 1);
    If[Length[ctx] == 0, ctx = {1}];           (* guard empty prompt *)
    gen = ctx;
    Do[
      fed = Take[gen, -Min[Length[gen], maxLen]];
      (* input port is Restricted[Integer,50257] -> ids must be 1..50257.
         Pad on the RIGHT with a valid id (1); causal masking means these
         trailing pad tokens cannot affect the logits at earlier positions,
         and we read the logit at the last REAL position. *)
      fed = PadRight[fed, maxLen, 1];
      out = net[fed];                          (* (maxLen, vocab) *)
      logits = out[[Min[Length[gen], maxLen]]];(* logits at last real position *)
      nxt = sampleNext[logits];
      AppendTo[gen, nxt],
      {nNew}];
    AppendTo[generations, (gen - 1)];          (* back to GPT-2 0-indexed ids *)
  ],
  {p, Length[prompts]}];

Export[FileNameJoin[{repo, "part-09-from-scratch", "data", "generations.json"}],
  <|"prompts" -> prompts, "generations" -> generations|>, "JSON"];
Print["wrote ", Length[generations], " generations"];
