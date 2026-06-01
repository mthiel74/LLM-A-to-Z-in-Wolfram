#!/usr/bin/env wolframscript
(* Assemble the Wolfram Community notebook for Part 3. *)

SetDirectory[DirectoryName[$InputFileName]];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];
nbDir   = FileNameJoin[{partDir, "notebook"}];

If[!DirectoryQ[nbDir], CreateDirectory[nbDir]];

(* ---------- helpers ---------- *)

title[s_]      := Cell[s, "Title"];
subtitle[s_]   := Cell[s, "Subtitle"];
section[s_]    := Cell[s, "Section"];
subsection[s_] := Cell[s, "Subsection"];
text[s_String] := Cell[s, "Text"];
text[parts_List] := Cell[TextData[parts], "Text"];
input[s_String] := Cell[s, "Input", FormatType -> InputForm];
inputBox[box_]  := Cell[BoxData[box], "Input"];
math[box_] := Cell[BoxData[box], "Text", FontSize -> 14];

mathDmodel  := math[SubscriptBox["d", "model"]];
mathDFF     := math[SubscriptBox["d", "ff"]];
mathDhead   := math[SubscriptBox["d", "head"]];
mathNHeads  := math[SubscriptBox["n", "heads"]];
mathNLayers := math[SubscriptBox["n", "layers"]];
mathV       := math[RowBox[{"|", "V", "|"}]];

mainCells = Flatten @ {

  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["Part 3: The full transformer block + parameter accounting"],

  text[{
    "Parts 1 and 2 produced two pieces: a tokeniser/embedder that turns ",
    "text into a (seqLen, ",
    mathDmodel,
    ") tensor (Part 1), and a multi-head causal self-attention block ",
    "that takes such a tensor and returns a tensor of the same shape ",
    "(Part 2). Part 3 wraps that attention block inside the canonical ",
    "transformer envelope and stacks the result ",
    mathNLayers,
    " times. We pick model sizes for Part 4's training runs, derive a ",
    "closed-form parameter-count formula, and verify it empirically."
  }],

  inputBox[RowBox[{"Import", "[",
    "\"figures/illustration_block_diagram.png\"", "]"}]],

  section["1.  Setup"],
  text[{
    "We need the Tokeniser package from Part 1 and the Attention package ",
    "from Part 2. The new Transformer package on top of them adds the ",
    "feed-forward block, the full transformer block, and the model ",
    "constructor."
  }],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"../../part-01-tokeniser-and-embeddings/src/Tokeniser.wl\"]"],
  input["Get[\"../../part-02-self-attention/src/Attention.wl\"]"],
  input["Get[\"src/Transformer.wl\"]"],

  section["2.  Position-wise feed-forward block"],
  text[{
    "The attention block lets every position look at every other position. ",
    "But within each position, all it does is take a weighted average of ",
    "value vectors. There is no per-position transformation that mixes the ",
    "features of a single token nonlinearly \[LongDash] no place for the ",
    "model to actually ",
    StyleBox["compute", FontSlant -> Italic],
    " on what it has gathered. The feed-forward block fixes that: a ",
    "two-layer position-wise MLP with a non-linearity in the middle."
  }],
  text[{
    "Two linear projections sandwich a GELU non-linearity: ",
    mathDmodel, " \[Rule] ", mathDFF, " \[Rule] ", mathDmodel,
    ". Conventionally ",
    mathDFF, " = 4 \[Times] ", mathDmodel,
    "; this is where most of the model's parameter budget lives. The ",
    "block is applied independently to every position of the sequence, ",
    "hence wrapped in ",
    StyleBox["NetMapOperator", FontFamily -> "Courier"],
    "."
  }],
  input["ffn = LLMAtoZ`Transformer`feedForwardBlock[64, 256, 16];
SeedRandom[1]; x = RandomReal[NormalDistribution[0, 0.1], {16, 64}];
Dimensions[ffn[x]]"],

  section["3.  The transformer block, pre-norm style"],
  text[{
    "Two sub-layers, each in its own residual connection:"
  }],
  text[{
    "    x' = x + Attention(LayerNorm(x))\n",
    "    y  = x' + FFN(LayerNorm(x'))"
  }],
  text[{
    "Two design choices visible here. First, the LayerNorm sits ",
    StyleBox["before", FontSlant -> Italic],
    " each sub-layer, not after. This is the ",
    StyleBox["pre-norm", FontWeight -> Bold],
    " ordering, which differs from the post-norm ordering in the original ",
    "Vaswani et al. paper. We will look at the gradient-flow argument for ",
    "pre-norm in a moment. Second, each sub-layer is wrapped in an ",
    "additive residual \[LongDash] the sub-layer learns what to ",
    StyleBox["add", FontSlant -> Italic],
    " to the residual stream, not what to replace it with. The original ",
    "ResNet argument applies: when a sub-layer's right answer is \
\[OpenCurlyDoubleQuote]do nothing\[CloseCurlyDoubleQuote], it can learn ",
    "to output zero rather than to learn the identity transformation."
  }],
  input["blk = LLMAtoZ`Transformer`transformerBlock[64, 4, 256, 16];
blk[x] // Dimensions"],
  input["Information[blk, \"ArraysTotalElementCount\"]"],

  section["4.  Pre-norm vs post-norm"],

  text[{
    "Why pre-norm? In a deep stack, what matters is whether the residual ",
    "stream is ever destroyed. Pre-norm normalises the ",
    StyleBox["input", FontSlant -> Italic],
    " to each sub-layer, then adds the sub-layer's output to the unnormalised ",
    "residual. The residual stream is never normalised, so a gradient flowing ",
    "backward through ",
    mathNLayers,
    " layers is the product of (1 + small_correction) factors, which is a ",
    "well-behaved O(1) number. Post-norm normalises ",
    StyleBox["after", FontSlant -> Italic],
    " the residual addition, so each layer rescales the residual stream by ",
    "1/std, and a gradient flowing backward is a product of repeatedly-rescaled ",
    "numbers \[LongDash] which can compress to zero quickly."
  }],
  text[{
    "A synthetic illustration of the difference on randomly-initialised ",
    "stacks (no training, just forward-pass activation norms):"
  }],
  input["Import[\"figures/pre_vs_post_norm_gradients.png\"]"],
  text[{
    "Pre-norm grows linearly in depth (because each residual addition adds ",
    "a small random vector to an unnormalised stream); post-norm flattens ",
    "because the LayerNorm at the end of every block keeps the stream's ",
    "magnitude pinned to 1. The reverse situation in backprop is what makes ",
    "post-norm hard to train at depth without warmup tricks. Modern ",
    "transformers (GPT-2/3, LLaMA, Mistral, nanochat) all use pre-norm."
  }],

  section["5.  The full GPT model"],
  text[{
    "Putting it together: token embedder + ",
    mathNLayers,
    " transformer blocks + a final LayerNorm + an output projection of ",
    "shape (", mathDmodel, ", ", mathV, ") that produces a logit at every ",
    "position for every token in the vocabulary."
  }],
  text[{
    "One small WL detail: the embedder from Part 1 has two input ports ",
    "(\"tokens\" and \"positions\"), which a ",
    StyleBox["NetChain", FontFamily -> "Courier"],
    " cannot consume directly. The Transformer package exports a ",
    StyleBox["chainEmbedder", FontFamily -> "Courier"],
    " that uses ",
    StyleBox["SequenceIndicesLayer", FontFamily -> "Courier"],
    " to generate the positions internally, so we can drop the embedder ",
    "into a chain along with the blocks."
  }],
  input["model = LLMAtoZ`Transformer`gptModel[256, 8, 6, 1024, 1064, 256];
ids = RandomInteger[{1, 1064}, 256];
Dimensions[model[ids]]"],
  text[{
    "(seqLen, |V|) out, as promised. Every row of the output is the ",
    "pre-softmax logit vector at one position. During training we shift it ",
    "by one and ask the model to predict the next token. During inference ",
    "we apply a softmax to the last row and sample. Both come in Parts 4 ",
    "and 5."
  }],

  section["6.  Counting parameters"],
  text[{
    "A pre-norm transformer block of size (", mathDmodel, ", ",
    mathDFF, ", ", mathNHeads, "):"
  }],
  text[{
    "\[FilledSmallCircle] Q, K, V projections summed over heads: ",
    math[RowBox[{"3", SuperscriptBox[SubscriptBox["d", "model"], "2"]}]],
    " (no biases by convention).\n",
    "\[FilledSmallCircle] Output projection: ",
    math[SuperscriptBox[SubscriptBox["d", "model"], "2"]],
    ".\n",
    "\[FilledSmallCircle] Two LayerNorms (scale + bias each): ",
    math[RowBox[{"4", SubscriptBox["d", "model"]}]],
    ".\n",
    "\[FilledSmallCircle] FFN linear 1 (with bias): ",
    math[RowBox[{SubscriptBox["d", "model"], SubscriptBox["d", "ff"],
                 " + ", SubscriptBox["d", "ff"]}]],
    ".\n",
    "\[FilledSmallCircle] FFN linear 2 (with bias): ",
    math[RowBox[{SubscriptBox["d", "ff"], SubscriptBox["d", "model"],
                 " + ", SubscriptBox["d", "model"]}]], "."
  }],
  text[{
    "Per block: ",
    math[RowBox[{
      "4", SuperscriptBox[SubscriptBox["d", "model"], "2"], " + ",
      "2", SubscriptBox["d", "model"], SubscriptBox["d", "ff"], " + ",
      SubscriptBox["d", "ff"], " + ", "5", SubscriptBox["d", "model"]
    }]],
    ". And on top of that, embeddings (",
    math[RowBox[{SubscriptBox["d", "model"],
                 RowBox[{"(", "|V|", " + ", SubscriptBox["n", "max"], ")"}]}]],
    "), a final LayerNorm (2", mathDmodel, "), and an output projection ",
    "(", math[RowBox[{SubscriptBox["d", "model"], "|V|", " + ", "|V|"}]],
    ")."
  }],
  text[{
    "The exported function ",
    StyleBox["parameterCount[spec]", FontFamily -> "Courier"],
    " computes this closed form; ",
    StyleBox["parameterCountFull[spec]", FontFamily -> "Courier"],
    " adds the frozen causal-mask arrays we bake into each attention head ",
    "(one ", mathNHeads, " \[Times] (maxSeqLen)\[Times](maxSeqLen) tensor ",
    "per layer), which is what WL's ",
    StyleBox["Information[net, \"ArraysTotalElementCount\"]",
            FontFamily -> "Courier"],
    " counts."
  }],
  input["spec = LLMAtoZ`Transformer`modelSpec5M[1064, 256];
trainable = LLMAtoZ`Transformer`parameterCount[spec];
full = LLMAtoZ`Transformer`parameterCountFull[spec];
actual = Information[
  LLMAtoZ`Transformer`gptModel[spec[\"dModel\"], spec[\"nHeads\"], spec[\"nLayers\"],
    spec[\"dFF\"], spec[\"vocabSize\"], spec[\"maxSeqLen\"]],
  \"ArraysTotalElementCount\"];
{trainable, full, actual}"],
  text[{
    "Trainable, full (with frozen masks), and actual agree exactly. The ",
    "frozen masks are non-trainable arrays the optimizer never touches; ",
    "they exist only because we stored the causal mask as a constant ",
    "inside the attention NetGraph. By convention we name our models by ",
    "their ",
    StyleBox["trainable", FontSlant -> Italic],
    " parameter count, so the 5M model truly has 5.3M trainable params."
  }],
  text["Where do those parameters live?"],
  input["Import[\"figures/params_breakdown.png\"]"],
  text[{
    "Most of the trainable parameters (37%) live in the FFN; the frozen ",
    "causal masks contribute another 37% of the array count even though ",
    "they are not trainable. The attention QKV projections account for ",
    "14%. The embedding tables, output projection, and LayerNorms ",
    "contribute the remaining low single-digit percentages. This is the ",
    "shape one finds across most transformer sizes; the picture changes ",
    "for very small models where the embedding tables dominate, and for ",
    "very large models where attention QKV catches up to FFN."
  }],

  section["7.  Three sizes for Part 4"],
  text[{
    "The training run in Part 4 needs a small ladder of model sizes so we ",
    "can recover a Chinchilla-style loss-vs-compute signature. The package ",
    "exports three reference specs at the Tiny Shakespeare scale ",
    "(|V| = 1064, maxSeqLen = 256):"
  }],
  input["Grid[
  Prepend[
    Table[
      With[{s = ms},
        {s[\"label\"], s[\"dModel\"], s[\"nHeads\"], s[\"nLayers\"], s[\"dFF\"],
         LLMAtoZ`Transformer`parameterCount[s],
         LLMAtoZ`Transformer`parameterCountFull[s]}],
      {ms, LLMAtoZ`Transformer`modelSizeSweep[1064, 256]}],
    {\"label\", \"d_model\", \"n_heads\", \"n_layers\", \"d_ff\",
     \"trainable\", \"with frozen masks\"}],
  Frame -> All, Background -> {None, {LightGray, None, None}},
  Alignment -> Center, Spacings -> {1.5, 0.5}]"],

  subsection["7.1  Interactive: dial your own model"],
  text[{
    "Pick any (", mathDmodel, ", ", mathDFF, ", ", mathNLayers, ") combination ",
    "and read off the parameter count on the fly. Common ratios: ",
    mathDFF, " = 4", mathDmodel, " for standard transformers; ",
    mathDFF, " between 2", mathDmodel, " and 4", mathDmodel,
    " for nanochat and recent compact architectures."
  }],
  input["Manipulate[
  With[{spec = <|\"dModel\" -> d, \"nHeads\" -> h, \"nLayers\" -> L,
                 \"dFF\" -> f, \"vocabSize\" -> v, \"maxSeqLen\" -> n|>},
    Column[{
      Style[Row[{\"trainable params: \",
        Style[NumberForm[LLMAtoZ`Transformer`parameterCount[spec], DigitBlock -> 3],
              Bold, 18, ColorData[97][1]]}], 14],
      Style[Row[{\"with frozen masks: \",
        Style[NumberForm[LLMAtoZ`Transformer`parameterCountFull[spec], DigitBlock -> 3],
              Bold, 14, ColorData[97][2]]}], 12],
      Style[Row[{\"size ratio FFN/attention: \",
        NumberForm[N[(2 d f + f + d) / (4 d^2 + 4 d)], 3]}], 12, Gray]
    }]
  ],
  {{d, 256, \"d_model\"}, 64, 1024, 64, Appearance -> \"Labeled\"},
  {{h, 8, \"n_heads\"}, 4, 16, 4, Appearance -> \"Labeled\"},
  {{L, 6, \"n_layers\"}, 1, 24, 1, Appearance -> \"Labeled\"},
  {{f, 1024, \"d_ff\"}, 256, 4096, 256, Appearance -> \"Labeled\"},
  {{v, 1064, \"|V|\"}, 256, 50000, 256, Appearance -> \"Labeled\"},
  {{n, 256, \"max_seq_len\"}, 64, 2048, 64, Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  section["8.  Reconstructing GPT-2 from our primitives"],

  text[{
    "A natural question: is our ",
    StyleBox["gptModel", FontFamily -> "Courier"],
    " constructor really general, or is it a toy that only handles the ",
    "1M/5M/25M reference sizes? The answer: plug in GPT-2 small's ",
    "architectural spec and you get GPT-2 small \[Dash] the same ",
    "architecture OpenAI shipped in 2019, parameter-count exact."
  }],
  input["spec = LLMAtoZ`Transformer`gpt2SmallSpec[];
spec"],
  text[{
    "Without weight tying the parameter formula gives ~163M; with weight ",
    "tying (sharing the input-embedding matrix with the output ",
    "projection, transposed) it gives 124.45M, matching real GPT-2 small ",
    "(124.44M) to within ~13K parameters. The residual is a tiny ",
    "convention gap in how LayerNorm biases are counted."
  }],
  input["{LLMAtoZ`Transformer`parameterCount[spec],
 LLMAtoZ`Transformer`parameterCount[spec, \"WeightTying\" -> True]}"],

  text[{
    "Two equivalent ways forward exist for the rest of the series:"
  }],
  text[{
    "\[FilledCircle] ",
    StyleBox["Path A: build from scratch.", FontWeight -> Bold],
    " ",
    StyleBox["buildGpt2Architecture[\"Small\"]", FontFamily -> "Courier"],
    " constructs an empty 124M-parameter net at GPT-2's spec. You then ",
    "train it from random init (Part 4 onward). On a Mac mini this is ",
    "infeasible \[LongDash] but the architectural code is yours.\n",
    "\[FilledCircle] ",
    StyleBox["Path B: load the pretrained NetModel.", FontWeight -> Bold],
    " ",
    StyleBox["loadPrebuiltGpt2[]", FontFamily -> "Courier"],
    " returns OpenAI's trained checkpoint via WL's Neural Network ",
    "Repository (498 MB on first download). You skip pretraining and ",
    "fine-tune from real pretrained weights (Part 6 onward), which is ",
    "what the cloud-cheap and cloud-real tiers actually do."
  }],

  text[{
    "We will choose between these in ",
    StyleBox["Tiers.wl", FontFamily -> "Courier"],
    " (introduced in Part 4): a single named tier sets the model spec, ",
    "the training corpus, and whether to use ",
    StyleBox["buildGpt2Architecture", FontFamily -> "Courier"],
    " or ",
    StyleBox["loadPrebuiltGpt2", FontFamily -> "Courier"],
    " as the starting point."
  }],

  text[{
    "A sanity check we ",
    StyleBox["can", FontSlant -> Italic],
    " do on a Mac mini: compare the two architectures' top-level ",
    "structure without instantiating the full 124M net."
  }],
  input["(* Don't build the full 124M net at notebook-eval time;
   it's slow and memory-hungry.  Inspect the spec instead. *)
specSummary = <|
  \"OurArchitecture\" -> \"gptModel[gpt2SmallSpec]: 12-layer pre-norm,
     12 heads, d_model 768, d_ff 3072, vocab 50257, max_seq 1024.\",
  \"OpenAIsArchitecture\" -> \"GPT-2 small: 12-layer post-norm or
     pre-norm (replications vary), 12 heads, d_model 768, d_ff 3072,
     vocab 50257, max_seq 1024.\",
  \"Differences\" -> {
    \"Our attention uses 12 separate (d_head x d_model) Q/K/V projections;
     OpenAI's checkpoint stores them as a single (d_model x d_model)
     matrix.  Mathematically equivalent; the reshape happens at load.\",
    \"Our output emits per-position logits (seqLen x |V|) for parallel
     training; the NetModel emits softmax probabilities at the last
     position only.  Either form is a one-line wrap of the other.\"
  }|>;
specSummary"],

  section["9.  Summary and what comes next"],
  text[{
    "We finish Part 3 with seven deliverables, each reusable in the ",
    "remaining parts:"
  }],
  text[{
    "\[FilledSquare] ",
    StyleBox["A pre-norm transformer block.", FontWeight -> Bold],
    " ",
    StyleBox["transformerBlock[dModel, nHeads, dFF, maxSeqLen]",
             FontFamily -> "Courier"],
    " returns a six-node NetGraph: norm \[Rule] attention \[Rule] add \
\[Rule] norm \[Rule] FFN \[Rule] add. ",
    "Pre-norm is the canonical 2020+ choice for stable deep training.\n",
    "\[FilledSquare] ",
    StyleBox["A position-wise feed-forward block.", FontWeight -> Bold],
    " ",
    StyleBox["feedForwardBlock[dModel, dFF, maxSeqLen]", FontFamily -> "Courier"],
    " is the standard ",
    mathDmodel, " \[Rule] ", mathDFF,
    " \[Rule] (GELU) \[Rule] ", mathDmodel,
    " sandwich, applied per position.\n",
    "\[FilledSquare] ",
    StyleBox["A full GPT model constructor.", FontWeight -> Bold],
    " ",
    StyleBox["gptModel[dModel, nHeads, nLayers, dFF, vocabSize, maxSeqLen]",
             FontFamily -> "Courier"],
    " returns a NetChain from token IDs to per-position logits, ready to ",
    "train.\n",
    "\[FilledSquare] ",
    StyleBox["A closed-form parameter count.", FontWeight -> Bold],
    " ",
    StyleBox["parameterCount[spec]", FontFamily -> "Courier"],
    " and ",
    StyleBox["parameterCountFull[spec]", FontFamily -> "Courier"],
    " agree exactly with the empirical ",
    StyleBox["Information", FontFamily -> "Courier"],
    " count for all three of our reference model sizes.\n",
    "\[FilledSquare] ",
    StyleBox["A model-size ladder.", FontWeight -> Bold],
    " ",
    StyleBox["modelSpec1M", FontFamily -> "Courier"], ", ",
    StyleBox["modelSpec5M", FontFamily -> "Courier"], ", ",
    StyleBox["modelSpec25M", FontFamily -> "Courier"],
    " return ready-to-use specs for the Part 4 training experiments. ",
    "(Trainable parameter counts: 1.10M, 5.34M, 18.65M.)\n",
    "\[FilledSquare] ",
    StyleBox["An interactive size dial.", FontWeight -> Bold],
    " Section 7.1's ",
    StyleBox["Manipulate", FontFamily -> "Courier"],
    " lets the reader pick any (",
    mathDmodel, ", ", mathNHeads, ", ", mathNLayers, ", ", mathDFF,
    ", ", mathV, ", maxSeqLen) and see the parameter count, the with-mask ",
    "count, and the FFN/attention size ratio update in real time.\n",
    "\[FilledSquare] ",
    StyleBox["GPT-2 reconstruction.", FontWeight -> Bold],
    " ",
    StyleBox["gpt2SmallSpec[]", FontFamily -> "Courier"],
    " plugged into our same ",
    StyleBox["gptModel", FontFamily -> "Courier"],
    " constructor yields a parameter count exact to within 13K of ",
    "real GPT-2 small (with weight tying). The same constructor at our ",
    "1M spec gives the toy model Part 4 actually trains.\n",
    "\[FilledSquare] ",
    StyleBox["Use-the-prebuilt-or-build-your-own switch.", FontWeight -> Bold],
    " ",
    StyleBox["loadPrebuiltGpt2[]", FontFamily -> "Courier"],
    " returns the trained NetModel checkpoint for fine-tuning ",
    "starts; ",
    StyleBox["buildGpt2Architecture[]", FontFamily -> "Courier"],
    " returns an empty net at the same spec for train-from-scratch ",
    "tiers."
  }],
  text[{
    "In ",
    StyleBox["Part 4", FontWeight -> Bold],
    " we train the 1M and 5M models on Tiny Shakespeare with a CPU-only ",
    "loop running on a Mac mini, plot the loss curve live via ",
    StyleBox["TrainingProgressFunction", FontFamily -> "Courier"],
    ", and offer the cloud path: submit the 25M model on a FineWeb-Edu ",
    "slice via ",
    StyleBox["RemoteBatchSubmit", FontFamily -> "Courier"],
    " for readers with a Wolfram Cloud account. The headline figure is ",
    "the loss vs FLOPs overlay across all three sizes, recovering a ",
    "small-scale Chinchilla signature."
  }],

  section["References"],
  text[{
    "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.\n",
    "Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML.\n",
    "Hendrycks, D., Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).\n",
    "Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models. (Chinchilla.)\n",
    "Karpathy, A. (2025). nanochat. github.com/karpathy/nanochat.\n",
    "Raschka, S. (2024). Build a Large Language Model from Scratch. Manning."
  }]
};

nb = Notebook[
  mainCells,
  WindowSize -> {900, 750},
  StyleDefinitions -> "Default.nb",
  Saveable -> True
];

outPath = FileNameJoin[{nbDir, "Part03.nb"}];
Export[outPath, nb, "NB"];
Print["Wrote ", outPath, "  (", FileByteCount[outPath], " bytes)"];
