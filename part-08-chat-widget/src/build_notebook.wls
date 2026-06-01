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
  subtitle["Part 8: An interactive chat widget"],

  text[{
    "We close the series with the artefact every reader actually wants: a ",
    "chat interface, inside the notebook, talking to the model we built. ",
    "An input field, a transcript pane, a streaming generator. The model ",
    "is the one Parts 1\[Dash]7 produced. Drop the widget cell into a notebook, ",
    "evaluate it, type, get a reply."
  }],

  section["1.  Setup"],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl\"]
Get[\"../../part-02-self-attention/src/Attention.wl\"]
Get[\"../../part-03-transformer-block/src/Transformer.wl\"]
Get[\"../../part-04-pretraining/src/Training.wl\"]
Get[\"../../part-05-sampling/src/Sampling.wl\"]
Get[\"../../part-06-supervised-fine-tuning/src/SFT.wl\"]
Get[\"src/Chat.wl\"]"],

  section["2.  Load the post-trained model"],
  text[{
    "For the demo we use the small pretrained checkpoint from Part 4 ",
    "(in a real run you'd load the GRPO-post-trained checkpoint from ",
    "Part 7)."
  }],
  input["bpe = Import[\"../../part-01-tokeniser-and-embeddings/data/bpe_trained.mx\"];
baseVocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[bpe];
chatVocab = LLMAtoZ`SFT`extendVocabulary[baseVocab];
tokId  = AssociationThread[chatVocab -> Range[Length[chatVocab]]];
idTok  = AssociationThread[Range[Length[chatVocab]] -> chatVocab];
pretrained = Import[
  \"../../part-04-pretraining/data/tiny_gpt_trained.wlnet\"];
net = LLMAtoZ`SFT`expandModelVocabulary[pretrained, Length[chatVocab]];"],

  section["3.  The chat widget"],
  text[{
    "Drop the cell below into the notebook and evaluate it. The widget ",
    "is a ",
    StyleBox["DynamicModule", FontFamily -> "Courier"],
    " holding a running transcript, an input field, and Send / Reset ",
    "buttons. Each Send triggers ",
    StyleBox["chatTurn", FontFamily -> "Courier"],
    ", which formats the existing transcript and the new user message ",
    "in the SFT chat template, runs the model autoregressively until ",
    "<|eot|> or the max-tokens limit, and appends the assistant turn."
  }],
  input["LLMAtoZ`Chat`chatWidget[net, bpe[\"MergeRules\"], tokId, idTok,
  \"MaxNew\" -> 60, \"Temperature\" -> 0.8, \"TopK\" -> 40]"],

  text[{
    "Caveat: this 1M-parameter model is laughably small. Even with the ",
    "SFT step from Part 6 and the GRPO step from Part 7 stacked on top, ",
    "it will not pass a Turing test \[LongDash] but it ",
    StyleBox["will", FontSlant -> Italic],
    " produce coherent Shakespeare-like exchanges, and every detail of ",
    "the pipeline from BPE through GRPO is something you wrote (and ",
    "could change). That is the point of the series."
  }],

  section["4.  Streaming generation"],
  text[{
    "For watching tokens land in real time, ",
    StyleBox["streamChatTurn", FontFamily -> "Courier"],
    " takes a callback that fires on every emitted token. The notebook ",
    "front end can use this to update a Dynamic in place."
  }],
  input["DynamicModule[{streamed = \"\"},
  Column[{
    Style[\"Streaming demo\", Bold, 14],
    Dynamic[Pane[streamed, {500, 100}, Scrollbars -> True]],
    Button[\"start\",
      streamed = \"\";
      LLMAtoZ`Chat`streamChatTurn[net, {}, \"Hello\",
        bpe[\"MergeRules\"], tokId, idTok,
        Function[t, streamed = streamed <> t],
        \"MaxNew\" -> 40, \"Temperature\" -> 0.7]
    ]
  }]
]"],

  section["5.  Deploying the chat as a web service"],
  text[{
    "The same chat function can be deployed to the Wolfram Cloud and ",
    "given a permanent URL. With ",
    StyleBox["CloudConnect", FontFamily -> "Courier"],
    " set up:"
  }],
  input["(* CloudConnect[\"you@example.com\"];
   CloudDeploy[
     APIFunction[
       {\"message\" -> \"String\"},
       LLMAtoZ`Chat`chatTurn[net, {}, #message,
         bpe[\"MergeRules\"], tokId, idTok][[-1, \"content\"]] &
     ],
     Permissions -> \"Public\"
   ]
*)"],

  section["6.  Where we end"],
  text[{
    "Across eight notebooks we built, from scratch in Wolfram Language:"
  }],
  text[{
    "1. A byte-pair-encoding tokeniser + embedder.\n",
    "2. Single-head and multi-head causal self-attention, with the ",
    math[RowBox[{"1", "/", SqrtBox[SubscriptBox["d", "k"]]}]],
    " factor derived rather than asserted.\n",
    "3. The full pre-norm transformer block + a closed-form parameter ",
    "count, with three reference model sizes.\n",
    "4. Pre-training on Tiny Shakespeare, with the cloud path for ",
    "scaling on a real corpus.\n",
    "5. Five sampling strategies plus speculative decoding.\n",
    "6. Supervised fine-tuning on instruction-following data.\n",
    "7. GRPO reinforcement learning post-training on a verifiable task.\n",
    "8. An interactive chat widget. This notebook."
  }],
  text[{
    "The final artefact, in front of you, is a small but ",
    StyleBox["functional", FontSlant -> Italic],
    " chat-style LLM where every component is something the reader can ",
    "open, change, and re-evaluate. The model is not big enough to be ",
    "useful as a product, but the pipeline is the same one production ",
    "labs use; everything in nanochat or in the Llama or GPT-2 paper has ",
    "a corresponding cell in this series."
  }],
  text[{
    "From here, the obvious next steps: scale (RemoteBatchSubmit on a ",
    "larger corpus), better tokeniser (byte-level BPE matching GPT-2 ",
    "exactly), better positional encoding (RoPE instead of learned ",
    "absolute), multi-task SFT (mix in code, math, dialogue), reasoning-",
    "style RL (GSM8K, longer chains-of-thought). Every one of those is ",
    "a single-package edit away."
  }],

  section["References"],
  text[{
    "Vaswani et al. (2017). Attention Is All You Need.\n",
    "Radford et al. (2019). GPT-2.\n",
    "Raschka, S. (2024). Build a Large Language Model from Scratch.\n",
    "Karpathy, A. (2025). nanochat."
  }]
};
nb = Notebook[cells, WindowSize -> {900, 750}, Saveable -> True];
out = FileNameJoin[{nbDir, "Part08.nb"}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes)"];
