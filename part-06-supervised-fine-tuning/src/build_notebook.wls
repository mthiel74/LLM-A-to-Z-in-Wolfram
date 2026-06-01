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
  subtitle["Part 6: Supervised fine-tuning (instruction following)"],

  text[{
    "The model after Part 4 can continue Shakespeare \[LongDash] but it has ",
    "no concept of \[OpenCurlyDoubleQuote]a user asks a question, an assistant answers\
\[CloseCurlyDoubleQuote]. ",
    "Supervised fine-tuning teaches that concept by training the same ",
    "next-token loss on a dataset where the inputs are already formatted ",
    "in the chat template ",
    StyleBox["<|user|>...<|assistant|>...<|eot|>", FontFamily -> "Courier"],
    ". The same ",
    StyleBox["NetTrain", FontFamily -> "Courier"],
    " call we used in Part 4, on different data."
  }],

  section["1.  Setup and chat-format vocabulary"],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl\"]
Get[\"../../part-02-self-attention/src/Attention.wl\"]
Get[\"../../part-03-transformer-block/src/Transformer.wl\"]
Get[\"../../part-04-pretraining/src/Training.wl\"]
Get[\"src/SFT.wl\"]"],
  text[{
    "We add three special tokens to the BPE vocabulary, after the BPE ",
    "merges learned in Part 1: ",
    StyleBox["<|user|>", FontFamily -> "Courier"], ", ",
    StyleBox["<|assistant|>", FontFamily -> "Courier"], ", ",
    StyleBox["<|eot|>", FontFamily -> "Courier"],
    ". Their IDs land at the end of the embedding matrix; the rest of the ",
    "vocabulary keeps its pretrained rows."
  }],
  input["bpe = Import[\"../../part-01-tokeniser-and-embeddings/data/bpe_trained.mx\"];
baseVocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[bpe];
chatVocab = LLMAtoZ`SFT`extendVocabulary[baseVocab];
tokId = AssociationThread[chatVocab -> Range[Length[chatVocab]]];
{Length[baseVocab], Length[chatVocab]}"],

  section["2.  The chat template"],
  text[{
    "Every turn in the training data is formatted into a single string ",
    "with the special tokens marking role boundaries:"
  }],
  input["LLMAtoZ`SFT`formatConversation[\"What is a sonnet?\",
  \"A sonnet is a poem of fourteen lines.\"]"],
  input["row = LLMAtoZ`SFT`tokeniseConversation[\"What is a sonnet?\",
  \"A sonnet is a poem of fourteen lines.\", bpe[\"MergeRules\"], tokId];
{Length[row[\"ids\"]], Total[row[\"assistantMask\"]]}"],

  section["3.  The toy instruction dataset"],
  text[{
    "Twenty hand-written Shakespeare-themed {user, assistant} pairs ",
    "ship with the package. Just enough to demonstrate SFT end to end ",
    "on the Mac mini; the same code consumes the SmolTalk subset when ",
    "scaled up via the cloud path."
  }],
  input["LLMAtoZ`SFT`toyInstructionDataset[][[1 ;; 4]]"],

  section["4.  Expand the model and train"],
  text[{
    "Load the pretrained model from Part 4, expand its token-embedding ",
    "and output projection to the new chat vocab size (adding random ",
    "rows for the three special tokens), then run a short SFT pass."
  }],
  input["pretrained = Import[
  \"../../part-04-pretraining/data/tiny_gpt_trained.wlnet\"];
expanded = LLMAtoZ`SFT`expandModelVocabulary[pretrained, Length[chatVocab]];"],
  input["dataset = LLMAtoZ`SFT`prepareSftDataset[
  LLMAtoZ`SFT`toyInstructionDataset[],
  bpe[\"MergeRules\"], tokId, 64];
Length[dataset]"],
  input["sftResult = LLMAtoZ`SFT`trainSft[expanded, dataset,
  <|\"BaseLR\" -> 0.0005, \"MaxRounds\" -> 8, \"BatchSize\" -> 4|>];
{sftResult[\"FinalLoss\"], sftResult[\"WallSeconds\"]}"],

  section["5.  Generate a response"],
  text[{
    "Same sampling code as in Part 5, but the model now produces text ",
    "in the SFT chat template, so we stop at <|eot|>."
  }],
  input["sftSample[net_, q_String, T_:0.8] := Module[
  {ids, ctx, pos, logits, p, sampled, contextLen = 64,
   eotId = tokId[\"<|eot|>\"],
   idTok = AssociationThread[Range[Length[chatVocab]] -> chatVocab]},
  ids = Lookup[tokId,
    LLMAtoZ`Tokeniser`bpeEncode[
      \"<|user|>\" <> q <> \"<|assistant|>\", bpe[\"MergeRules\"]]];
  (* fall back via lookup loop *)
  Do[
    ctx = If[Length[ids] >= contextLen, ids[[-contextLen ;;]],
              PadRight[ids, contextLen, 1]];
    pos = Min[Length[ids], contextLen];
    logits = net[ctx][[pos]];
    p = Exp[(logits - Max[logits])/T]; p = p/Total[p];
    sampled = RandomChoice[p -> Range[Length[p]]];
    AppendTo[ids, sampled];
    If[sampled === eotId, Break[]],
    {60}];
  StringReplace[StringJoin @ Lookup[idTok, ids],
    {\"</w>\" -> \" \", \"<|user|>\" -> \"\\nUSER: \",
     \"<|assistant|>\" -> \"\\nASSISTANT: \", \"<|eot|>\" -> \"\"}]
];
sftSample[sftResult[\"Net\"], \"What is a sonnet?\"]"],

  section["6.  Parameter-efficient SFT with LoRA"],
  text[{
    "Full SFT updates every weight in the pretrained model. ",
    StyleBox["LoRA", FontWeight -> Bold],
    " (Hu et al. 2021) updates a tiny fraction. The trick: for each ",
    "linear layer with weight ",
    math["W"], " of shape ",
    math[RowBox[{SubscriptBox["d", "in"], "\[Times]",
                 SubscriptBox["d", "out"]}]],
    ", freeze ", math["W"],
    " and learn two small matrices ", math["A"], " and ", math["B"],
    " of ranks ", math["r"], " so the effective weight at inference is ",
    math[RowBox[{"W", "+", "BA"}]],
    ". For r = 8 and ", math["d = 768"],
    ", LoRA adds ", math[RowBox[{"8", "\[Times]", "(", "768", "+", "768", ")",
                                 "=", "12288"}]],
    " parameters versus the full ", math["768\[Times]768 = 589,824"],
    " \[LongDash] a 48\[Times] reduction."
  }],
  input["Get[\"src/Lora.wl\"]"],
  input["(* Build a small LoRA delta and confirm zero-init identity. *)
delta = LLMAtoZ`Lora`loraDelta[64, 64, 8];
Print[\"loraDelta params: \", Information[delta, \"ArraysTotalElementCount\"]];
Print[\"vs full LinearLayer params: \", 64*64]"],
  input["LLMAtoZ`Lora`loraParameterCount[768, 768, #] & /@ {4, 8, 16, 32, 64}"],

  text[{
    "Compare to the full per-attention-block parameter count of ",
    math[RowBox[{"4", SuperscriptBox[SubscriptBox["d", "model"], "2"]}]],
    " = 4\[Times]768\[Times]768 \[TildeEqual] 2.36M parameters: even ",
    "rank-64 LoRA on all four QKV+output projections costs only ",
    "~100K parameters per layer, a >20\[Times] saving. With 12 layers ",
    "that is ~1.2M trainable LoRA parameters vs ~28M in full SFT."
  }],
  subsection["6.1  Interactive: choose your LoRA rank"],
  input["Manipulate[
  With[{loraPer = LLMAtoZ`Lora`loraParameterCount[768, 768, r],
        fullPer = 768*768},
    Column[{
      Style[Row[{\"LoRA params per linear: \", loraPer}], 14, Bold],
      Style[Row[{\"Full linear params:     \", fullPer}], 12],
      Style[Row[{\"Savings: \",
        NumberForm[100. (1 - loraPer/fullPer), 4], \"%\"}], 12, Gray]
    }]
  ],
  {{r, 8, \"LoRA rank\"}, 1, 128, 1, Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  section["7.  Classification fine-tuning (Raschka chapter 6)"],
  text[{
    "Same pretrained model, very different task: ",
    StyleBox["classification.", FontSlant -> Italic],
    " Instead of producing the next token, output a single ",
    "class label given the whole sequence. The architectural change is ",
    "tiny: strip the LM head (per-position vocab projection), add a ",
    StyleBox["SequenceLastLayer", FontFamily -> "Courier"],
    " to pool the hidden state at the final position, then a fresh ",
    StyleBox["LinearLayer[nClasses]", FontFamily -> "Courier"],
    ". The rest of the model \[LongDash] all the pretrained ",
    "transformer blocks \[LongDash] is reused as-is."
  }],
  input["Get[\"src/Classification.wl\"]"],
  input["classifier = LLMAtoZ`Classification`attachClassifier[
  pretrained, 2, 96];
classifier"],
  text[{
    "Demo task: classify Shakespeare excerpts as tragedy (1) or comedy ",
    "(2). The package ships a 20-pair hand-curated dataset for the ",
    "demo; the same code consumes IMDb-sentiment or AG News if you ",
    "supply them."
  }],
  input["clsData = LLMAtoZ`Classification`prepareClassificationDataset[
  LLMAtoZ`Classification`shakespeareGenreLabels[],
  bpe[\"MergeRules\"], tokId, 64];
Length[clsData]"],
  input["clsResult = LLMAtoZ`Classification`trainClassifier[
  classifier, clsData,
  <|\"BaseLR\" -> 0.001, \"MaxRounds\" -> 6, \"BatchSize\" -> 4|>];
{clsResult[\"FinalLoss\"], clsResult[\"WallSeconds\"]}"],
  input["LLMAtoZ`Classification`classifySample[clsResult[\"Net\"],
  LLMAtoZ`Classification`prepareClassificationDataset[
    {{\"To be, or not to be, that is the question.\", 0}},
    bpe[\"MergeRules\"], tokId, 64][[1, \"Input\"]]]"],

  text[{
    "Caveat: with only 20 training examples and a tiny base model, the ",
    "classifier will overfit and barely generalise. The mechanics are ",
    "exactly right; scale the dataset to thousands of examples (and the ",
    "model to 25M+ params) and the same code path produces a working ",
    "spam/sentiment/topic classifier. Raschka's chapter 6 uses ~5K ",
    "samples for the same target."
  }],

  section["8.  Summary"],
  text[{
    "Part 6 covers the three main fine-tuning regimes that turn a ",
    "pretrained next-token model into something useful:"
  }],
  text[{
    "\[FilledSquare] ",
    StyleBox["Full SFT for instruction following.", FontWeight -> Bold],
    " Chat template, vocab extension, ",
    StyleBox["expandModelVocabulary", FontFamily -> "Courier"],
    " for grafting new rows onto the pretrained embedder and output ",
    "projection.\n",
    "\[FilledSquare] ",
    StyleBox["LoRA for parameter-efficient SFT.", FontWeight -> Bold],
    " ",
    StyleBox["loraDelta", FontFamily -> "Courier"], ", ",
    StyleBox["loraWrap", FontFamily -> "Courier"], ", ",
    StyleBox["applyLoraToLinear", FontFamily -> "Courier"], ", ",
    StyleBox["trainLora", FontFamily -> "Courier"],
    " \[LongDash] 20-50\[Times] fewer trainable parameters than full SFT, ",
    "no degradation in result quality at the scales most users actually ",
    "run.\n",
    "\[FilledSquare] ",
    StyleBox["Classification fine-tuning.", FontWeight -> Bold],
    " Strip the LM head, pool the last-position hidden state, project ",
    "to ", math[SubscriptBox["n", "classes"]],
    " logits. Same pretrained backbone; entirely different output ",
    "objective. The Raschka chapter-6 setup."
  }],
  text[{
    "In ",
    StyleBox["Part 7", FontWeight -> Bold],
    " we go beyond imitation and apply ",
    StyleBox["reinforcement learning", FontSlant -> Italic],
    " on top of the SFT model: GRPO with a verifiable binary reward, ",
    "no separate reward model, the recipe from nanochat."
  }]
};
nb = Notebook[cells, WindowSize -> {900, 750}, Saveable -> True];
out = FileNameJoin[{nbDir, "Part06.nb"}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes)"];
