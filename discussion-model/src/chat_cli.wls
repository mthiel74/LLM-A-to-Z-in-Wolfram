#!/usr/bin/env wolframscript
(* Interactive streaming terminal chat with the trained GPT-2 medium SFT
   model.  Loads once; reads turns from stdin; streams each generated
   token to stdout immediately (no waiting for the whole reply).

   Commands inside the chat:
     /reset   clear transcript
     /quit    exit
*)

SetDirectory["/Users/thiel/GitHub/LLM-A-to-Z-in-WL/discussion-model"];
Get["src/Discussion.wl"];

modelPath = "data/discussion_medium_20k.wlnet";
If[!FileExistsQ[modelPath],
  Print["Trained model not found at ", modelPath];
  Print["Either run discussion-model/src/run_discussion_sft.wls (CPU, slow)"];
  Print["or CloudGet the one at:"];
  Print["  https://www.wolframcloud.com/obj/m.thiel/discussion-model/discussion-medium-20k.wlnet"];
  Quit[]];

Print["Loading GPT-2 small (for BPE encoder/decoder; 498 MB)..."];
gpt = LLMAtoZ`Discussion`loadGpt2Small[];
encoder = NetExtract[gpt, "Input"];
rawDecoder = NetExtract[gpt, "Output"]; (* takes 50257-vector one-hots *)

Print["Loading trained medium SFT model (1.35 GB)..."];
t0 = AbsoluteTime[];
net = Import[modelPath];
Print["  loaded in ", Round[AbsoluteTime[] - t0, 0.1], " s; ",
      Information[net, "ArraysTotalElementCount"], " params."];

V = 50257;
(* Single-token decoder: dense one-hot through bundled NetDecoder.
   ~200 KB transient per token, much cheaper than the batched all-at-once. *)
decodeOne[id_Integer] := Module[{v = ConstantArray[0., V]},
  v[[id]] = 1.; rawDecoder[v]];

(* Heuristic fix for the byte-level BPE Mojibake (curly apostrophes etc.
   getting rendered as â€™). Reinterpret the WL string as UTF-8 bytes. *)
fixUtf8[s_String] := Module[{bytes},
  bytes = ToCharacterCode[s, "ISO8859-1"];
  Quiet @ Check[FromCharacterCode[bytes, "UTF-8"], s]];

(* Decode the full token list at the end with the same fix. *)
fullDecode[ids_List] := fixUtf8 @ StringJoin[decodeOne /@ ids];

(* Streaming generation. Prints each token as it's produced; checks for
   a stop string in the accumulated text after each token.  Returns the
   final cleaned reply. *)
streamReply[promptStr_String, opts_Association] := Module[
  {ids, T = opts["Temperature"], topK = opts["TopK"],
   maxNew = opts["MaxNew"], ctxLen = 1024, j, scores, last, probs, nextId,
   generated = {}, decoded = "", chunk, runningText = "",
   stopStrings, hitStop, replyOut},
  stopStrings = {" You:", " User:", " Assistant:", "\nUser:", "\nYou:",
                 "\nAssistant:"};
  ids = encoder[promptStr];
  Do[
    scores = net[Take[ids, -Min[Length[ids], ctxLen]]];
    last = Last[scores] / T;
    If[topK < Length[last],
      Module[{cut = Sort[last, Greater][[topK]]},
        last = Map[If[# < cut, -Infinity, #] &, last]]];
    probs = Exp[last - Max[last]];
    probs = probs / Total[probs];
    nextId = RandomChoice[probs -> Range[Length[probs]]];
    AppendTo[ids, nextId];
    AppendTo[generated, nextId];

    (* Decode just this one token, fix Mojibake, write to terminal. *)
    chunk = fixUtf8[decodeOne[nextId]];
    runningText = runningText <> chunk;
    WriteString["stdout", chunk];

    (* Stop conditions *)
    hitStop = SelectFirst[stopStrings, StringContainsQ[runningText, #] &,
              None];
    If[hitStop =!= None,
      (* truncate runningText at the stop string *)
      runningText = First @ StringSplit[runningText, hitStop, 2];
      Break[]];

    (* repetition stop: 3 identical IDs in a row *)
    If[Length[generated] >= 3 &&
       MatchQ[Take[generated, -3], {x_, x_, x_}],
      Break[]];
    (* low diversity in last 6 tokens *)
    If[Length[generated] >= 6 &&
       Length[Union @ Take[generated, -6]] <= 2,
      Break[]],
    {j, maxNew}];
  WriteString["stdout", "\n"];
  StringTrim[runningText]];

Print[""];
Print["================================================================"];
Print["  Chat with GPT-2 medium SFT (20k Alpaca, 1 epoch)              "];
Print["  Streams tokens as they generate.  Commands:                   "];
Print["    /reset   clear transcript                                   "];
Print["    /quit    exit                                               "];
Print["================================================================"];
Print[""];

history = {};
While[True,
  WriteString["stdout", "You: "];
  line = InputString[""];
  If[line === EndOfFile || line === $Failed, Print[""]; Break[]];
  line = StringTrim[line];
  Which[
    line === "/quit" || line === "/exit",
      Break[],
    line === "/reset",
      history = {};
      Print["[transcript cleared]"];
      Continue[],
    line === "",
      Continue[]];

  prompt = StringJoin[
    Map[Function[turn,
      "User: " <> turn["user"] <> "\nAssistant: " <> turn["assistant"] <> "\n"],
      history],
    "User: ", line, "\nAssistant:"];

  WriteString["stdout", "AI: "];
  t0 = AbsoluteTime[];
  reply = streamReply[prompt, <|"MaxNew" -> 100, "Temperature" -> 0.7,
                                 "TopK" -> 40|>];
  elapsed = AbsoluteTime[] - t0;
  Print["[", Round[elapsed, 0.1], " s, ", StringLength[reply], " chars]"];
  Print[""];

  AppendTo[history, <|"user" -> line, "assistant" -> reply|>];
  If[Length[history] > 4, history = history[[-4 ;;]]]];

Print["bye."];
