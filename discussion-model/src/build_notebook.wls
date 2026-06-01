#!/usr/bin/env wolframscript
(* Build discussion-model/LLM_Discussion.nb -- the standalone scale-up
   notebook.  Loads GPT-2 small via NetModel, performs the strip-and-
   wrap surgery, SFTs on Alpaca-cleaned, and exposes a chat widget.
   Designed to slot into the master series later as a new chapter.       *)

SetDirectory[DirectoryName[$InputFileName]];
partDir = ParentDirectory[];

title[s_]      := Cell[s, "Title"];
subtitle[s_]   := Cell[s, "Subtitle"];
chapter[s_]    := Cell[s, "Chapter"];
section[s_]    := Cell[s, "Section"];
subsection[s_] := Cell[s, "Subsection"];
text[s_String] := Cell[s, "Text"];
text[parts_List] := Cell[TextData[parts], "Text"];
input[s_String] := Cell[s, "Input", FormatType -> InputForm];
program[s_String] := Cell[s, "Program"];

cells = Flatten @ {
  title["Scaling Up: A GPT-2 Discussion Model"],
  subtitle["From the 20-pair toy SFT to a model that actually chats"],

  text[{
    "The master notebook ended with a 124M-parameter GPT-2 model SFT'd on ",
    "20 hand-written Q&A pairs.  It learned the chat ",
    StyleBox["template", FontSlant -> Italic],
    " but its factual answers were obviously wrong (\"Hamlet is a ",
    "playwright who wrote the play in 1763\").  In this notebook we keep ",
    "the same architecture and the same training loop -- and just give ",
    "the model real data: 10,000 instruction\[Dash]response pairs from the ",
    "Apache-2.0 Alpaca-cleaned corpus (Stanford 2023).  Three SFT epochs ",
    "take 3-4 hours on a Mac mini CPU.  The result is a small but ",
    "genuinely conversational model."
  }],

  text[{
    "Two design choices worth flagging:"
  }],
  text[{
    "1. ", StyleBox["GPT-2 small (124M), not medium (345M)", FontWeight -> Bold],
    ".  Medium gives a more capable base, but in our Mac-mini setup it ",
    "hits an MXNet operator-shape bug when we strip the inference tail ",
    "and re-chain for training, and per-example wall time would put any ",
    "meaningful training run beyond 8 days.  Cloud compute fixes both ",
    "issues; for the educational story we keep small and treat ",
    "RemoteBatchSubmit as the scaling switch.\n",
    "2. ", StyleBox["Plain-text chat template, no new vocabulary tokens",
                    FontWeight -> Bold],
    ".  The master notebook taught the canonical ",
    StyleBox["<|user|>...<|assistant|>...<|eot|>", FontFamily -> "Courier"],
    " trick.  Here we use plain ",
    StyleBox["\"User: ...\\nAssistant: ...\\n\"", FontFamily -> "Courier"],
    ".  GPT-2's BPE tokeniser already represents \"User\", \"Assistant\", ",
    "the colon, the newline; we keep the 50,257-entry vocabulary and ",
    "the weight tying intact.  Less elegant theoretically; more robust ",
    "practically."
  }],

  chapter["1.  Setup"],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"src/Discussion.wl\"]"],

  chapter["2.  Load GPT-2 small and inspect"],
  text[{
    "WL ships GPT-2 in three sizes (117M, 345M, 774M) via the Neural Net ",
    "Repository.  Loading downloads ~498 MB on first call and caches the ",
    "weights on disk."
  }],
  input["gpt = LLMAtoZ`Discussion`loadGpt2Small[];
Information[gpt, \"ArraysTotalElementCount\"]"],

  text[{
    "The pretrained net has five top-level layers:"
  }],
  input["Keys @ NetExtract[gpt, All]"],

  text[{
    "The structure is:"
  }],
  text[{
    StyleBox["embedding", FontFamily -> "Courier"],
    " (token + positional) \[RightArrow] ",
    StyleBox["decoder", FontFamily -> "Courier"],
    " (12 transformer blocks for small) \[RightArrow] ",
    StyleBox["last", FontFamily -> "Courier"],
    " (SequenceLastLayer: takes only the last position) \[RightArrow] ",
    StyleBox["classifier", FontFamily -> "Courier"],
    " (LinearLayer: 768 \[Rule] 50,257) \[RightArrow] ",
    StyleBox["probabilities", FontFamily -> "Courier"],
    " (SoftmaxLayer).  This wiring is for ",
    StyleBox["inference", FontSlant -> Italic],
    ": one prompt in, one next-token distribution out."
  }],

  chapter["3.  Surgery: from inference net to training trunk"],
  text[{
    "For SFT we need per-position logits at every step.  Three changes:"
  }],
  text[{
    "1. Drop ", StyleBox["SequenceLastLayer", FontFamily -> "Courier"],
    " so we keep all positions of the (seqLen, dModel) decoder output.\n",
    "2. Wrap the ", StyleBox["classifier", FontFamily -> "Courier"], " in ",
    StyleBox["NetMapOperator", FontFamily -> "Courier"],
    " so it applies row-wise rather than once.\n",
    "3. Drop the final ", StyleBox["SoftmaxLayer", FontFamily -> "Courier"],
    " -- the training loop adds its own softmax via ",
    StyleBox["NetMapOperator[SoftmaxLayer[]]", FontFamily -> "Courier"],
    " so that ", StyleBox["CrossEntropyLossLayer[\"Index\"]", FontFamily -> "Courier"],
    " sees probabilities (the bug from Section 5.3 of the master)."
  }],
  input["trunk = LLMAtoZ`Discussion`buildTrainableTrunk[gpt];
Information[trunk, \"ArraysTotalElementCount\"]"],
  text[{
    "Same parameter count as the original (we just rearranged), and the ",
    "embedding-classifier weight tying is preserved through the bundled ",
    "SharedArrays."
  }],
  input["(* Sanity: feed a 4-token sequence and verify shape *)
encoder = NetExtract[gpt, \"Input\"];
ids = encoder[\"Hello world. How are you today?\"];
Print[\"ids: \", ids];
out = trunk[ids];
Print[\"output dims: \", Dimensions[out]]"],
  text[{
    "(seqLen, 50257) -- per-position logits over the full vocabulary.  ",
    "That is the shape NetTrain needs."
  }],

  chapter["4.  The dataset: Alpaca-cleaned"],
  text[{
    "Alpaca-cleaned (Yahma 2023, Apache 2.0) is 51,760 instruction-",
    "response pairs in three fields: ",
    StyleBox["instruction", FontFamily -> "Courier"], ", ",
    StyleBox["input", FontFamily -> "Courier"], " (optional context), ",
    StyleBox["output", FontFamily -> "Courier"], ".  Format each as plain ",
    "text with markers that GPT-2's BPE already tokenises naturally."
  }],
  input["examples = LLMAtoZ`Discussion`loadAlpacaExamples[5];
TableForm[
  Map[KeyTake[#, {\"instruction\", \"output\"}] &, examples],
  TableHeadings -> {None, {\"instruction\", \"output\"}}]"],
  input["LLMAtoZ`Discussion`formatPair[examples[[1]]]"],

  text[{
    "Encode + truncate / pad to a fixed context length of 128 tokens, ",
    "then split each sequence into shifted input / target pairs."
  }],
  input["dataset = LLMAtoZ`Discussion`buildSftDataset[
  LLMAtoZ`Discussion`formatPair /@ examples,
  encoder, 128];
Print[\"examples: \", Length[dataset]];
Print[\"first example shape: input=\", Length[dataset[[1]][\"Input\"]],
      \"  output=\", Length[dataset[[1]][\"Output\"]]]"],

  chapter["5.  Training (the actual SFT)"],
  text[{
    "On a Mac mini CPU at batch_size 4, GPT-2 small SFT runs at about ",
    StyleBox["0.42 s/example", FontWeight -> Bold],
    " in steady state.  10,000 examples \[Times] 3 epochs = 30,000 forward+",
    "backward passes; about 3.5 hours wall.  ",
    StyleBox["trainOvernight", FontFamily -> "Courier"],
    " checkpoints to disk every 30 minutes so an early ",
    StyleBox["Ctrl-C", FontFamily -> "Courier"], " still leaves you with a ",
    "usable model."
  }],
  text[{
    "Heavy run -- skip in interactive notebook use.  Kick it off from a ",
    "shell instead and read the progress log:"
  }],
  program["wolframscript -file discussion-model/src/run_discussion_sft.wls 10000 3
tail -f discussion-model/data/discussion_progress.txt"],
  text[{
    "The cells below load a previously-trained checkpoint and exercise ",
    "it.  Until that file exists, the chat widget at the end will show a ",
    "warning placeholder."
  }],

  chapter["6.  Load the trained model"],
  text[{
    "The notebook expects to be opened from ",
    StyleBox["discussion-model/", FontFamily -> "Courier"],
    ", so paths below are anchored at ",
    StyleBox["NotebookDirectory[]", FontFamily -> "Courier"],
    ".  Pick the best available trained checkpoint: the cloud-trained ",
    "GPT-2 medium (1.35 GB) if you have it, else the local GPT-2 small ",
    "SFT (498 MB), else fall back to the untrained trunk so the rest ",
    "of the chapter still runs."
  }],
  input["SetDirectory[NotebookDirectory[]];
candidates = {
  \"data/discussion_medium_20k.wlnet\",
  \"data/discussion_sft.wlnet\",
  \"data/discussion_sft_ckpt.wlnet\"};
sftPath = SelectFirst[candidates, FileExistsQ, None];
sftNet = If[sftPath === None,
  Print[Style[\"No trained checkpoint found in data/.  Using untrained \" <>
              \"trunk; chat widget will produce nonsense.\", Red]];
  trunk,
  Print[\"Loading \", sftPath, \" ...\"];
  Import[sftPath]];
Information[sftNet, \"ArraysTotalElementCount\"]"],

  chapter["7.  Generation"],
  text[{
    "Sampling is the same recipe as Chapter 6 of the master notebook: ",
    "temperature + top-k.  ",
    StyleBox["generate", FontFamily -> "Courier"],
    " also stops cleanly when the model produces ",
    StyleBox["\\n", FontFamily -> "Courier"],
    StyleBox["User: ", FontFamily -> "Courier"],
    " -- it would otherwise babble onwards into a new turn."
  }],
  input["decoderFn = LLMAtoZ`Discussion`makeIdDecoder[gpt];
SeedRandom[2026];
LLMAtoZ`Discussion`generate[sftNet, encoder, decoderFn,
  \"User: What is a sonnet?\\nAssistant:\",
  <|\"MaxNew\" -> 80, \"Temperature\" -> 0.7, \"TopK\" -> 40|>]"],
  input["LLMAtoZ`Discussion`generate[sftNet, encoder, decoderFn,
  \"User: How are you today?\\nAssistant:\",
  <|\"MaxNew\" -> 60, \"Temperature\" -> 0.7|>]"],
  input["LLMAtoZ`Discussion`generate[sftNet, encoder, decoderFn,
  \"User: Give three tips for staying healthy.\\nAssistant:\",
  <|\"MaxNew\" -> 120, \"Temperature\" -> 0.7|>]"],

  chapter["8.  Interactive chat widget"],
  text[{
    "Multi-turn chat via ", StyleBox["DynamicModule", FontFamily -> "Courier"],
    ".  Each Send concatenates the running transcript with the new user ",
    "message in the User/Assistant template and feeds it to ",
    StyleBox["chatTurn", FontFamily -> "Courier"], "."
  }],
  text[{
    "If after clicking ", StyleBox["Send", FontWeight -> Bold],
    " you see ", StyleBox["Thinking\[Ellipsis]", FontSlant -> Italic],
    " for 10\[Dash]60 seconds: that's expected.  GPT-2 small inference ",
    "in Wolfram Language on a Mac mini CPU runs around 5 tokens/sec, ",
    "so an 80-token reply takes 15\[Dash]20 s.  GPT-2 medium is even ",
    "slower (\[TildeTilde]60 s per reply).  When the model finishes, ",
    "the new turn appears in the transcript pane above the input."
  }],

  input["DynamicModule[{history = {}, userInput = \"\", thinking = False},
  Framed[
    Column[{
      Style[\"Discussion model \[Bullet] \" <> sftPath, Bold, 14,
            FontFamily -> \"Helvetica\"],
      Spacer[8],
      (* Transcript pane.  Wrap in a Framed Pane so the conversation
         scrolls.  We rebuild from `history` on every change. *)
      Framed[
        Dynamic[
          If[Length[history] == 0 && !thinking,
            Style[\"(no messages yet \[Dash] type below and click Send)\",
                  FontSlant -> Italic, FontColor -> Gray,
                  FontFamily -> \"Helvetica\"],
            Column[Join[
              Map[Function[turn,
                Column[{
                  Row[{Style[\"User: \", Bold, FontFamily -> \"Helvetica\"],
                       Style[turn[\"user\"], FontFamily -> \"Helvetica\"]}],
                  Row[{Style[\"AI:   \", Bold,
                             FontFamily -> \"Helvetica\",
                             FontColor -> RGBColor[0.2, 0.4, 0.7]],
                       Style[turn[\"assistant\"],
                             FontFamily -> \"Helvetica\"]}],
                  Spacer[6]}]], history],
              If[thinking,
                {Style[\"AI:   thinking\[Ellipsis]\", Bold,
                       FontFamily -> \"Helvetica\",
                       FontColor -> RGBColor[0.5, 0.5, 0.5],
                       FontSlant -> Italic]}, {}]]]]],
        Background -> RGBColor[0.97, 0.97, 1.0],
        FrameStyle -> RGBColor[0.85, 0.85, 0.9],
        RoundingRadius -> 6, ImageSize -> {640, Automatic},
        FrameMargins -> 10],
      Spacer[6],
      InputField[Dynamic[userInput], String,
        FieldSize -> {60, 2}, FieldHint -> \"Type your message...\"],
      Spacer[4],
      Row[{
        Button[\"Send\",
          If[StringQ[userInput] && StringLength[StringTrim[userInput]] > 0,
            Module[{msg = StringTrim[userInput], reply},
              userInput = \"\";
              thinking = True;
              reply = LLMAtoZ`Discussion`chatTurn[
                sftNet, history, msg, encoder, decoderFn,
                <|\"MaxNew\" -> 80, \"Temperature\" -> 0.7, \"TopK\" -> 40|>];
              AppendTo[history, <|\"user\" -> msg,
                                  \"assistant\" -> StringTrim[reply]|>];
              thinking = False]],
          Method -> \"Queued\",
          Background -> RGBColor[0.2, 0.55, 0.85],
          BaseStyle -> {FontFamily -> \"Helvetica\", Bold, White}],
        Spacer[6],
        Button[\"Reset\", history = {}; userInput = \"\";
                          thinking = False,
          Background -> RGBColor[0.9, 0.9, 0.9],
          BaseStyle -> {FontFamily -> \"Helvetica\"}]
      }]
    }],
    Background -> White,
    FrameStyle -> RGBColor[0.6, 0.6, 0.65],
    RoundingRadius -> 8, FrameMargins -> 14]
]"],

  chapter["9.  Limits and where to go"],
  text[{
    "This 124M model SFT'd for 3 epochs holds the chat format and gives ",
    "plausible responses to general questions.  It will not pass a Turing ",
    "test, and its factual accuracy is only as good as Alpaca's: many ",
    "details are wrong.  The next levers, in order of impact per hour:"
  }],
  text[{
    "1. ", StyleBox["More data + a chattier dataset.", FontWeight -> Bold],
    " SmolTalk (HuggingFace, ~150k multi-turn dialogues) is a much better ",
    "fit for chat than Alpaca's single-turn instructions.\n",
    "2. ", StyleBox["Bigger base via cloud.", FontWeight -> Bold],
    " GPT-2 medium (345M) or large (774M) SFT'd via RemoteBatchSubmit on ",
    "the same data; budget ~\[Sterling]30-100.\n",
    "3. ", StyleBox["RL post-training.", FontWeight -> Bold],
    " Apply the GRPO step from Chapter 8 of the master with a quality ",
    "reward (e.g., a small judge model that scores helpfulness).\n",
    "4. ", StyleBox["Modern instruct base.", FontWeight -> Bold],
    " For a model that genuinely chats, the only fast path is pulling a ",
    "post-2024 instruction-tuned model (SmolLM2-360M-Instruct, Qwen2.5-",
    "0.5B-Instruct) via ONNX.  Tradeoff: breaks the from-scratch story."
  }],

  chapter["References"],
  text[{
    "Radford, A. et al. (2019). Language Models are Unsupervised Multitask ",
    "Learners. (GPT-2.)\n",
    "Taori, R. et al. (2023). Stanford Alpaca: An Instruction-Following ",
    "LLaMA Model.\n",
    "Yahma. (2023). alpaca-cleaned (HuggingFace dataset, Apache 2.0)."
  }]
};

nb = Notebook[cells, WindowSize -> {900, 750}, Saveable -> True];
out = FileNameJoin[{partDir, "LLM_Discussion.nb"}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes, ", Length[cells], " cells)"];
