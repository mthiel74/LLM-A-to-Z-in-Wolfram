#!/usr/bin/env wolframscript
(* Build the "map of the whole notebook" flowchart for LLM_A_to_Z.nb.

   Shows every model / path discussed, grouped by what it starts from
   (architecture built from scratch vs OpenAI's pretrained GPT-2 weights),
   with the chapter number where each step lives.  Native Mathematica
   Graphics so it is reproducible and the section numbers are exact text. *)

(* ---- palette ---- *)
cBuild  = RGBColor[0.20, 0.40, 0.65];   (* architecture / from-scratch *)
cTrain  = RGBColor[0.16, 0.52, 0.34];   (* training steps *)
cPretr  = RGBColor[0.70, 0.45, 0.13];   (* uses pretrained GPT-2 *)
cModel  = RGBColor[0.45, 0.20, 0.55];   (* deliverable model *)
cUse    = RGBColor[0.55, 0.16, 0.22];   (* sampling / chat / use *)

(* a rounded box: pos = center, returns Graphics primitives *)
box[pos_, w_, h_, color_, chap_, label_] := {
  EdgeForm[{color, Thickness[0.0016]}], FaceForm[Lighter[color, 0.82]],
  Rectangle[pos - {w, h}/2, pos + {w, h}/2, RoundingRadius -> 0.07],
  Text[Style[chap, 9, Bold, color], pos + {0, h/2 - 0.16}],
  Text[Style[label, 8.3, Darker[color, 0.15],
    FontFamily -> "Helvetica"], pos, {0, 0}]
};

arrow[a_, b_, col_: GrayLevel[0.45]] :=
  {col, Arrowheads[0.014], Thickness[0.0016], Arrow[{a, b}]};

(* ---- layout (x grows right, y grows up) ---- *)
W = 2.55; H = 0.92; col1 = 0; col2 = 3.4; col3 = 6.8; col4 = 10.2;

prims = {
  (* row labels / lanes via background tint *)
  FaceForm[GrayLevel[0.97]], EdgeForm[],
  Rectangle[{-1.6, -8.9}, {11.6, 1.1}, RoundingRadius -> 0.1],

  (* ===== top: shared foundation ===== *)
  box[{col2 + 1.7, 0.3}, 5.4, H, cBuild, "Chapters 1\[Dash]4",
    "Build the architecture from scratch\n(tokeniser \[Rule] embeddings \[Rule] attention \[Rule] transformer block)"],

  (* ===== PATH A: original GPT model, Tiny Shakespeare ===== *)
  box[{col1, -1.7}, W, H, cTrain, "Chapter 5",
    "Pre-train GPT on\nTiny Shakespeare (CPU)"],
  box[{col1, -3.2}, W, H, cUse, "Chapter 6",
    "Sampling: temperature,\ntop-k/p, beam, speculative"],

  (* ===== PATH B: pretrained GPT-2 -> SFT/RL/chat ===== *)
  box[{col3, -1.7}, W, H, cPretr, "Chapter 7",
    "Load OpenAI GPT-2,\nsupervised fine-tune (SFT)"],
  box[{col3, -3.2}, W, H, cTrain, "Chapter 8",
    "RL post-training\n(GRPO on a verifiable task)"],
  box[{col3, -4.7}, W, H, cUse, "Chapter 9",
    "Interactive chat widget"],
  box[{col4, -1.7}, W, H, cModel, "Chapter 10",
    "Discussion model:\nGPT-2 medium SFT on\n20k Alpaca (cloud)"],

  (* ===== PATH C (NEW): from-scratch nano on cloud GPUs ===== *)
  box[{col1, -6.5}, W, H, cBuild, "Chapter 11",
    "Nano architecture\n(RMSNorm, RoPE, SwiGLU)"],
  box[{col2, -6.5}, W, H, cTrain, "Chapter 11",
    "Pre-train from scratch\n600M FineWeb-Edu tok, T4\nv2 ppl 55 \[Bullet] v3 ppl 52"],
  box[{col3, -6.5}, W, H, cModel, "Chapter 11",
    "Base model:\nfluent English, ~$3/run"],
  box[{col4, -6.5}, W, H, cTrain, "Chapter 11",
    "Instruction tune (SFT)\non Alpaca \[Rule] chat model"],
  box[{col4, -8.0}, W, H, cUse, "Chapter 11",
    "Chat: right form,\nshaky facts (30M limit)"],

  (* ===== arrows ===== *)
  (* foundation feeds all three paths *)
  arrow[{col2 + 1.7, 0.3 - H/2}, {col1, -1.7 + H/2}],
  arrow[{col2 + 1.7, 0.3 - H/2}, {col3, -1.7 + H/2}],
  arrow[{col2 + 1.7, 0.3 - H/2}, {col1, -6.5 + H/2}],
  (* path A *)
  arrow[{col1, -1.7 - H/2}, {col1, -3.2 + H/2}],
  (* path B *)
  arrow[{col3, -1.7 - H/2}, {col3, -3.2 + H/2}],
  arrow[{col3, -3.2 - H/2}, {col3, -4.7 + H/2}],
  arrow[{col3 + W/2, -1.7}, {col4 - W/2, -1.7}, cPretr],
  (* path C *)
  arrow[{col1 + W/2, -6.5}, {col2 - W/2, -6.5}, cBuild],
  arrow[{col2 + W/2, -6.5}, {col3 - W/2, -6.5}, cTrain],
  arrow[{col3 + W/2, -6.5}, {col4 - W/2, -6.5}, cModel],
  arrow[{col4, -6.5 - H/2}, {col4, -8.0 + H/2}, cTrain],

  (* ===== lane captions ===== *)
  Text[Style["PATH A  \[LongDash]  original GPT, trained on Tiny Shakespeare",
     8.5, Italic, cTrain], {col1 - W/2 + 0.1, -0.75}, {-1, 0}],
  Text[Style["PATH B  \[LongDash]  reuse OpenAI's pretrained GPT-2 weights",
     8.5, Italic, cPretr], {col3 - W/2 + 0.1, -0.75}, {-1, 0}],
  Text[Style["PATH C  \[LongDash]  pretrain from scratch on cloud GPUs (Part 9)",
     8.5, Italic, cBuild], {col1 - W/2 + 0.1, -5.55}, {-1, 0}]
};

flowchartGraphics = Graphics[prims,
  PlotRange -> {{-1.7, 11.7}, {-9.0, 1.2}},
  ImageSize -> 980, Background -> White, AspectRatio -> Automatic];

(* Only export when run directly (not when Get-loaded by the insert script). *)
If[StringContainsQ[ToString[$ScriptCommandLine], "build_flowchart"],
  Export["/tmp/flowchart.png", flowchartGraphics, ImageResolution -> 110];
  Print["wrote /tmp/flowchart.png  ", FileByteCount["/tmp/flowchart.png"], " bytes"]];
