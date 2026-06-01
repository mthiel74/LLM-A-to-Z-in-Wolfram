#!/usr/bin/env wolframscript
(* Dump the WL GPT-2 vocabulary as a JSON list: position i (0-indexed) gives
   the string for WL token ID (i+1). Chunked to avoid 10 GB peaks. *)

SetDirectory["/Users/thiel/GitHub/LLM-A-to-Z-in-WL/discussion-model"];

Print["Loading bundled GPT-2 small (for the NetDecoder)..."];
(* The small variant is just as good for the BPE vocabulary and is 6x smaller
   to download than 345M. *)
gpt = NetModel[{"GPT-2 Transformer Trained on WebText Data",
                "Size" -> "117M",
                "Task" -> "LanguageModeling"}];
dec = NetExtract[gpt, "Output"];
V = 50257;

Print["Decoding ", V, " tokens in chunks of 1000..."];
chunkSize = 1000;
chunks = Partition[Range[V], chunkSize, chunkSize, 1, {}];
allDecoded = {};
t0 = AbsoluteTime[];
Do[
  Module[{ids = chunks[[c]], oneHots, decoded},
    (* build dense one-hots for this chunk only *)
    oneHots = Table[
      Module[{v = ConstantArray[0., V]}, v[[i]] = 1.; v],
      {i, ids}];
    decoded = dec[oneHots];
    allDecoded = Join[allDecoded, decoded];
    If[Mod[c, 5] == 0,
      Print["  chunk ", c, "/", Length[chunks], "  (",
            Round[AbsoluteTime[] - t0, 0.1], " s elapsed, ",
            Length[allDecoded], " decoded)"]];
  ],
  {c, Length[chunks]}];
Print["Decoded ", Length[allDecoded], " tokens in ",
      Round[AbsoluteTime[] - t0, 0.1], " s"];

Print["First 5: ", allDecoded[[1;;5]]];
Print["914 (WL token for \"the\"): ", allDecoded[[914]]];
Print["12727 (WL token for \"User\"): ", allDecoded[[12727]]];
Print["50257 (last): ", allDecoded[[50257]]];

out = "data/hf_weights/wl_vocab.json";
Export[out, allDecoded, "JSON"];
Print["Wrote ", out];
