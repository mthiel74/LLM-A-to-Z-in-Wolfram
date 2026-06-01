#!/usr/bin/env wolframscript
(* Assemble the Wolfram Community notebook for Part 2.

   Same pattern as Part 1: TextData/BoxData math, Import["...png"]
   cells for figures, runnable Input cells for code. *)

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

mathSqrtDk      := math[FractionBox["1", SqrtBox[SubscriptBox["d", "k"]]]];
mathDk          := math[SubscriptBox["d", "k"]];
mathDhead       := math[SubscriptBox["d", "head"]];
mathDmodel      := math[SubscriptBox["d", "model"]];
mathQKT         := math[RowBox[{"Q", "\[CenterDot]", SuperscriptBox["K", "\[Transpose]"]}]];
mathSoftmaxAtt  := math[RowBox[{
  "softmax", "(",
  FractionBox[RowBox[{"Q", "\[CenterDot]", SuperscriptBox["K", "\[Transpose]"]}],
              SqrtBox[SubscriptBox["d", "k"]]],
  ")", "\[CenterDot]", "V"}]];

(* ---------- cells ---------- *)

mainCells = Flatten @ {

  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["Part 2: Self-attention and the QKV picture"],

  text[{
    "Part 1 built a tokeniser and an embedder; we ended with a ",
    StyleBox["NetGraph", FontFamily -> "Courier"],
    " that turns a sequence of BPE token IDs into a (seqLen, ",
    mathDmodel, ") input tensor.  In Part 2 we feed that tensor into ",
    "the mechanism the transformer is actually named for: scaled ",
    "dot-product attention, derived from first principles."
  }],

  text[{
    "The two pieces of work for this post:"
  }],

  text[{
    "\[FilledCircle] Derive ",
    mathSoftmaxAtt, " step by step, including the ",
    mathSqrtDk, " factor that keeps the pre-softmax logits at unit ",
    "variance.  Most expositions assert this scaling; we will show ",
    "where it comes from with an empirical histogram you can ",
    "reproduce.\n",
    "\[FilledCircle] Build a single-head and a multi-head causal ",
    "attention block as inspectable ",
    StyleBox["NetGraph", FontFamily -> "Courier"],
    "s, then load a pretrained GPT-2 small and visualise its real, ",
    "learned attention heatmaps \[LongDash] before we train our own ",
    "from scratch in Part 4."
  }],

  inputBox[RowBox[{"Import", "[",
    "\"figures/illustration_attention_hero.png\"", "]"}]],

  (* ----- Section 1 ----- *)

  section["1.  Setup"],

  text["The attention package and the cached pretrained GPT-2 net."],
  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"src/Attention.wl\"]"],

  text[{
    "GPT-2 small is downloaded via WL's ",
    StyleBox["NetModel", FontFamily -> "Courier"],
    ".  First-time access pulls about 500 MB of weights and caches them; ",
    "subsequent loads are local-file fast.  We save a copy to the part's ",
    StyleBox["data/", FontFamily -> "Courier"],
    " directory so the rest of the notebook does not depend on a network ",
    "connection."
  }],

  input["gpt = NetModel[\"GPT-2 Transformer Trained on WebText Data\",
                \"Task\" -> \"LanguageModeling\"];"],

  (* ----- Section 2: why attention ----- *)

  section["2.  Attention is a learnable weighted sum"],

  text[{
    "Strip everything fancy away. Attention is the operation \
\[OpenCurlyDoubleQuote]each output token is a weighted sum of the input \
tokens, where the weights are learned from the input itself.\[CloseCurlyDoubleQuote] ",
    "Suppose we have a 4-token sequence ",
    math[RowBox[{SubscriptBox["x", "1"], ",", SubscriptBox["x", "2"], ",",
                 SubscriptBox["x", "3"], ",", SubscriptBox["x", "4"]}]],
    " each in ",
    math[SuperscriptBox["\[DoubleStruckCapitalR]", "d"]],
    ".  For token 3, a weighted sum is just ",
    math[RowBox[{SubscriptBox["y", "3"], "=",
                 SubscriptBox["a", "31"], SubscriptBox["x", "1"], "+",
                 SubscriptBox["a", "32"], SubscriptBox["x", "2"], "+",
                 SubscriptBox["a", "33"], SubscriptBox["x", "3"]}]],
    " with the ",
    math[SubscriptBox["a", RowBox[{"3", "j"}]]],
    " summing to 1 (causal masking forbids attending to ",
    math[SubscriptBox["x", "4"]], ").  All attention does is decide those ",
    "weights from the content of the tokens themselves rather than hand-coding them."
  }],

  text[{
    "The three-line learnable version: turn each ",
    math[SubscriptBox["x", "i"]],
    " into a query ",
    math[SubscriptBox["q", "i"]],
    ", a key ",
    math[SubscriptBox["k", "i"]],
    ", and a value ",
    math[SubscriptBox["v", "i"]],
    " via three learned linear maps. The compatibility of token i with ",
    "token j is the dot product ",
    math[RowBox[{SubscriptBox["s", RowBox[{"i", "j"}]], "=",
                 SubscriptBox["q", "i"], "\[CenterDot]", SubscriptBox["k", "j"]}]],
    ". A softmax across j gives the attention weights; their weighted sum ",
    "of the values is the output of token i."
  }],

  text[{
    "An intuition that differs from the usual exposition: the ",
    StyleBox["query", FontSlant -> Italic],
    " is the question a token asks of the others; the ",
    StyleBox["key", FontSlant -> Italic],
    " is the answer it provides to such queries; the ",
    StyleBox["value", FontSlant -> Italic],
    " is what it actually passes along once selected. They are three ",
    "separate learned views of the same underlying vector."
  }],

  (* ----- Section 3: the scaling factor ----- *)

  section["3.  Why divide by Sqrt[d_k]?"],

  text[{
    "Most treatments state the scaled dot-product formula ",
    mathSoftmaxAtt,
    " and leave the ",
    mathSqrtDk,
    " factor unexplained.  The argument is short and is one of the ",
    "places where a small piece of probability theory makes a design ",
    "choice obvious."
  }],

  text[{
    "Suppose ",
    math[RowBox[{"q", ",", " ", "k", " ", "\[Element]", " ",
                 SuperscriptBox["\[DoubleStruckCapitalR]", SubscriptBox["d", "k"]]}]],
    " with each component independently drawn from a zero-mean ",
    "unit-variance distribution.  The dot product ",
    math[RowBox[{"q", "\[CenterDot]", "k", "=",
                 UnderoverscriptBox["\[Sum]",
                   RowBox[{"i", "=", "1"}],
                   SubscriptBox["d", "k"]],
                 RowBox[{SubscriptBox["q", "i"], SubscriptBox["k", "i"]}]}]],
    " is a sum of ", mathDk, " random terms ",
    math[RowBox[{SubscriptBox["q", "i"], SubscriptBox["k", "i"]}]],
    ".  Each term has mean ",
    math[RowBox[{"E", "[", SubscriptBox["q", "i"], SubscriptBox["k", "i"], "]",
                 "=", "E", "[", SubscriptBox["q", "i"], "]", "\[CenterDot]",
                 "E", "[", SubscriptBox["k", "i"], "]", "=", "0"}]],
    " and variance ",
    math[RowBox[{"E", "[", SuperscriptBox[SubscriptBox["q", "i"], "2"], "]",
                 "\[CenterDot]",
                 "E", "[", SuperscriptBox[SubscriptBox["k", "i"], "2"], "]",
                 "=", "1"}]],
    " (using independence of ",
    math[SubscriptBox["q", "i"]], " and ", math[SubscriptBox["k", "i"]],
    " and that each has unit variance).  The ", mathDk,
    " terms are themselves independent across i, so the variance of ",
    "the sum is the sum of the variances: ",
    math[RowBox[{"Var", "(", "q", "\[CenterDot]", "k", ")", "=", SubscriptBox["d", "k"]}]],
    ", and the standard deviation grows as ",
    math[SqrtBox[SubscriptBox["d", "k"]]], "."
  }],

  text[{
    "We can verify this in two lines of Wolfram Language. Sample 2000 ",
    "Gaussian (q, k) pairs at three values of ",
    mathDk,
    " and look at the empirical distribution of their dot product:"
  }],

  input["SeedRandom[1];
nSamples = 2000;
dks = {16, 64, 256};
dots = Association @ Table[
  d -> Table[
    With[{q = RandomReal[NormalDistribution[], d],
          k = RandomReal[NormalDistribution[], d]},
      q . k],
    {nSamples}],
  {d, dks}];
StandardDeviation /@ dots"],

  text[{
    "The empirical standard deviations come out to roughly 4, 8, 16 \
\[LongDash] matching the predicted ",
    math[SqrtBox[SubscriptBox["d", "k"]]],
    " of 4, 8, 16 to within statistical error.  Now look at what this ",
    "does to a softmax. Softmax of a vector whose entries have very ",
    "large spread is degenerate: the largest entry dominates ",
    "exponentially and all others are crushed to zero, so the gradient ",
    "through the softmax is essentially zero. That is the gradient ",
    "pathology that prevents a transformer from learning at any but the ",
    "smallest scale."
  }],

  input["Import[\"figures/qkdotproduct_variance.png\"]"],

  text[{
    "Dividing by ",
    math[SqrtBox[SubscriptBox["d", "k"]]],
    " returns the pre-softmax logits to unit variance, which keeps the ",
    "softmax in its high-gradient regime.  The right-hand panel above ",
    "shows the three distributions collapsing onto each other once we ",
    "apply the scaling.  A concrete softmax demo on synthetic 8-token ",
    "scores at ",
    mathDk, " = 256:"
  }],

  input["Import[\"figures/softmax_saturation.png\"]"],

  subsection["3.1  Interactive: turn the d_k dial"],

  text[{
    "Slide ", mathDk, " from 4 up to 1024 and watch what happens to ",
    "(left) the empirical distribution of ", mathQKT,
    " entries and (right) the softmax of a single row of those scores. ",
    "The unscaled histogram blows up; the unscaled softmax collapses to ",
    "near-one-hot. The right-hand panel switches scaling on or off with ",
    "the second slider so you can see the difference instantly."
  }],

  input["Manipulate[
  SeedRandom[1];
  Module[{nSamples = 600, nRow = 16, dots, row, scale, sm},
    (* Empirical distribution of q.k for many random (q,k) pairs. *)
    dots = Table[
      With[{q = RandomReal[NormalDistribution[], dk],
            k = RandomReal[NormalDistribution[], dk]},
        q . k],
      {nSamples}];
    scale = If[useScaling, Sqrt[N[dk]], 1.0];
    (* One row of Q.K^T: a single query against nRow keys, all in R^{dk}. *)
    With[{q = RandomReal[NormalDistribution[], dk],
          K = RandomReal[NormalDistribution[], {nRow, dk}]},
      row = (K . q) / scale];
    sm = Exp[row - Max[row]]; sm = sm / Total[sm];
    GraphicsRow[{
      Histogram[dots/scale, 30, \"PDF\",
        PlotRange -> {{-6, 6}, {0, 0.8}},
        Frame -> True,
        FrameLabel -> {If[useScaling, \"q\[CenterDot]k / \[Sqrt]d_k\", \"q\[CenterDot]k\"],
                       \"PDF\"},
        PlotLabel -> Row[{\"std = \",
          NumberForm[StandardDeviation[dots/scale], 3]}],
        ImageSize -> 300],
      BarChart[sm,
        Frame -> True,
        PlotLabel -> Row[{\"softmax row (\", If[useScaling, \"scaled\", \"unscaled\"], \")\"}],
        FrameLabel -> {\"key position\", \"attention weight\"},
        PlotRange -> {Automatic, {0, 1}},
        ImageSize -> 300]
    }, ImageSize -> 640]
  ],
  {{dk, 64, \"d_k\"}, 4, 1024, 4, Appearance -> \"Labeled\"},
  {{useScaling, True, \"divide by \[Sqrt]d_k\"}, {True, False}},
  ControlPlacement -> Top]"],

  text[{
    "Why this matters: pull ", mathDk, " up past about 128 with scaling ",
    "off and you can see the softmax collapse to a single non-zero entry. ",
    "That is what a transformer would face on every step of training if ",
    "the ", mathSqrtDk, " divisor were not there \[LongDash] gradient ",
    "everywhere zero, optimisation stalled. With the divisor on, the ",
    "softmax stays usable regardless of ", mathDk, "."
  }],

  text[{
    "Left panel: without scaling, the softmax is essentially picking ",
    "one entry per row, so the gradient of the loss with respect to the ",
    "non-winning logits is essentially zero.  Right panel: with the ",
    mathSqrtDk, " scaling, the softmax distributes mass meaningfully, ",
    "and gradients flow."
  }],

  (* ----- Section 4: causal mask ----- *)

  section["4.  Causal masking"],

  text[{
    "A language model is only allowed to attend to past tokens, never ",
    "to future ones.  Implementation: add a strict-upper-triangular ",
    "matrix of ",
    math["-\[Infinity]"],
    " to the pre-softmax score matrix; after row-wise softmax those ",
    "entries become exactly zero.  In Wolfram Language, with a large ",
    "negative number standing in for ",
    math["-\[Infinity]"],
    " (the latter participates in NaN-producing arithmetic upstream):"
  }],

  input["LLMAtoZ`Attention`causalMask[6] // MatrixForm"],

  text[{
    "The triangular indexing here uses WL's 1-based convention. ",
    "UpperTriangularize with second argument ",
    StyleBox["1", FontFamily -> "Courier"],
    " means \[OpenCurlyDoubleQuote]above the main diagonal\
\[CloseCurlyDoubleQuote] (so the diagonal itself stays zero \[LongDash] ",
    "a token may attend to itself)."
  }],

  input["Import[\"figures/causal_mask_visualisation.png\"]"],

  text[{
    "Left to right: the mask, the masked scores (random pre-softmax ",
    "scores plus the mask), and the row-wise softmax weights. The upper ",
    "triangle is exactly zero after softmax; the lower triangle plus ",
    "diagonal carries the actual attention weights."
  }],

  (* ----- Section 5: single-head NetGraph ----- *)

  section["5.  Single-head causal self-attention as a NetGraph"],

  text[{
    "Six nodes: three ",
    StyleBox["LinearLayer", FontFamily -> "Courier"],
    " projections for Q, K, V; a ",
    StyleBox["FunctionLayer", FontFamily -> "Courier"],
    " that computes ",
    math[FractionBox[RowBox[{"Q", "\[CenterDot]",
        SuperscriptBox["K", "\[Transpose]"]}],
        SqrtBox[SubscriptBox["d", "head"]]]],
    " and adds the causal mask; a ",
    StyleBox["SoftmaxLayer", FontFamily -> "Courier"],
    " for the row-wise normalisation; and a final ",
    StyleBox["FunctionLayer", FontFamily -> "Courier"],
    " for the value gather. The whole graph is six lines."
  }],

  input["head = LLMAtoZ`Attention`singleHeadCausalAttention[64, 32, 16]"],

  text[{
    "Each ",
    StyleBox["LinearLayer", FontFamily -> "Courier"],
    " is wrapped in ",
    StyleBox["NetMapOperator", FontFamily -> "Courier"],
    " so it applies row-wise to a (seqLen, dModel) input \[LongDash] ",
    "this is the WL idiom for \[OpenCurlyDoubleQuote]apply this layer to every position in the \
sequence\[CloseCurlyDoubleQuote]. The score and output ",
    StyleBox["FunctionLayer", FontFamily -> "Courier"],
    "s are transparent: anyone reading the source sees the actual ",
    "matrix multiplication."
  }],

  input["SeedRandom[42];
x = RandomReal[NormalDistribution[0, 0.1], {16, 64}];
out = head[x];
Dimensions[out]"],

  text[{
    "(seqLen, dHead) in, (seqLen, dHead) out. Sixteen positions, ",
    "each rebuilt as a weighted average of its allowed predecessors. ",
    "Note that the causal mask is baked into the NetGraph at the maxSeqLen ",
    "passed to the constructor; calling this single-head block on a ",
    "shorter input requires either padding to maxSeqLen or constructing ",
    "a smaller block. In a production training loop we use one block at ",
    "the training maxSeqLen throughout. Production attention often also ",
    "includes a dropout on the post-softmax weights for regularisation; ",
    "we omit it here for pedagogical clarity and re-introduce it in Part 4."
  }],

  (* ----- Section 6: multi-head ----- *)

  section["6.  Multi-head attention"],

  text[{
    "A single head can only attend in one way at a time.  Multi-head ",
    "attention runs ",
    math[SubscriptBox["n", "heads"]],
    " parallel single-head computations: each head applies its own ",
    "learned projections of the full input down to dimension ",
    math[SubscriptBox["d", "head"]],
    ", computes attention independently, and produces its own ",
    math[SubscriptBox["d", "head"]],
    "-dimensional output. The heads are then concatenated and a final ",
    "output projection mixes them back into a ",
    mathDmodel, "-vector. The architectural trick is to split ",
    mathDmodel, " into ",
    math[SubscriptBox["n", "heads"]],
    " equal pieces so the multi-head version has the same parameter ",
    "count as a single-head version at ",
    mathDmodel, "."
  }],

  text[{
    "For pedagogical clarity our package implements the parallel-heads ",
    "form directly: ",
    math[SubscriptBox["n", "heads"]],
    " independent ",
    StyleBox["singleHeadCausalAttention", FontFamily -> "Courier"],
    " subgraphs feeding a ",
    StyleBox["CatenateLayer", FontFamily -> "Courier"],
    " and an output ",
    StyleBox["LinearLayer", FontFamily -> "Courier"],
    ".  A production version would project Q, K, V once at the full ",
    mathDmodel, " width and then reshape into heads, saving the ",
    "duplicate projection overhead; the I/O is identical."
  }],

  input["mha = LLMAtoZ`Attention`multiHeadAttentionBlock[64, 4, 16];
mha[x] // Dimensions"],

  (* ----- Section 7: GPT-2 heatmaps ----- *)

  section["7.  What real attention looks like: GPT-2 small"],

  text[{
    "We can do something the typical from-scratch tutorial cannot: ",
    "load a fully pretrained language model from ",
    StyleBox["NetModel", FontFamily -> "Courier"],
    " and look at its actual attention patterns.  GPT-2 small is a ",
    "12-layer transformer with 12 heads per layer at ",
    mathDmodel, " = 768, ",
    mathDhead, " = 64.  Each block's attention sublayer holds four ",
    "learned matrices: ",
    math[SubscriptBox["W", "Q"]], ", ",
    math[SubscriptBox["W", "K"]], ", ",
    math[SubscriptBox["W", "V"]],
    " (768\[Times]768 each) plus a 768\[Times]768 output projection.  ",
    "We have the weights; we can recompute the attention weight matrix ",
    "for any input prompt by hand using the same scaled-dot-product ",
    "formula we just built."
  }],

  input["ids = LLMAtoZ`Attention`gpt2EncodePrompt[gpt,
  \"The quick brown fox jumps over the lazy dog\"];
weights5 = LLMAtoZ`Attention`gpt2AttentionWeights[gpt, ids, 5];
Dimensions[weights5]"],

  text[{
    "Twelve heads, nine tokens, nine tokens per head.  Each row of each ",
    "head sums to one (we use the same ",
    StyleBox["softmaxRows", FontFamily -> "Courier"],
    " utility the from-scratch attention uses).  Plotting all twelve ",
    "heads of layer 5:"
  }],

  input["Import[\"figures/gpt2_heads_gallery.png\"]"],

  text[{
    "Two features jump out.  First, every head has a heavily-shaded ",
    "leftmost column \[LongDash] a phenomenon called the ",
    StyleBox["attention sink", FontSlant -> Italic],
    " (Xiao et al., 2024): attention heads dump probability mass onto ",
    "the first token as a \[OpenCurlyDoubleQuote]no-op\
\[CloseCurlyDoubleQuote] vote when they have nothing more useful to do.  ",
    "Second, the heads are not all doing the same thing.  Hand-picking ",
    "three to illustrate the kinds of specialisation that emerge from ",
    "training:"
  }],

  input["Import[\"figures/gpt2_head_specialisation.png\"]"],

  text[{
    "Left: a head that allocates noticeable weight to the previous ",
    "token (the row-i, col-(i\[Minus]1) diagonal).  Middle: a head whose ",
    "non-sink mass sits firmly on the diagonal \[LongDash] each token ",
    "re-attends to itself, a re-write rather than a look-back. ",
    "Right: a head that spreads roughly uniformly across all previous ",
    "tokens, contributing a smoothed average of the prefix rather than ",
    "any specific position. ",
    "None of these specialisations is enforced architecturally; they ",
    "are what twelve heads turn into when trained against billions of ",
    "tokens of text."
  }],

  subsection["7.1  Interactive: browse all 12 layers \[Times] 12 heads"],

  text[{
    "Pre-extracted attention tensors for every layer of GPT-2 small on ",
    "the same prompt are cached in ",
    StyleBox["data/gpt2_attention_all_layers.wxf", FontFamily -> "Courier"],
    ". The widget below lets you scrub through layer and head and watch ",
    "the structure change. Lower layers look fairly similar across heads; ",
    "by the middle of the stack each head has carved out its own pattern."
  }],

  input["attCache = Import[\"data/gpt2_attention_all_layers.wxf\"];
Manipulate[
  With[{w = attCache[\"weights\"][layer][[head]],
        n = Length[attCache[\"ids\"]]},
    MatrixPlot[w,
      ColorFunction -> \"ThermometerColors\",
      ColorFunctionScaling -> True,
      FrameTicks -> {{Range[n], None}, {Range[n], None}},
      PlotLabel -> Row[{\"GPT-2 small  \[Bullet]  layer \", layer,
                        \"  \[Bullet]  head \", head,
                        \"  \[Bullet]  prompt: \\\"\",
                        attCache[\"prompt\"], \"\\\"\"}],
      PlotLegends -> Automatic,
      ImageSize -> 460]
  ],
  {{layer, 5, \"layer\"}, 1, 12, 1, Appearance -> \"Labeled\"},
  {{head, 1, \"head\"}, 1, 12, 1, Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  text[{
    "Things worth scrubbing for:\n",
    "\[FilledCircle] Layer 1, all heads: mostly diffuse \[LongDash] there ",
    "is not yet much per-head differentiation.\n",
    "\[FilledCircle] Layers 4\[Dash]7: the cleanest specialisation \[LongDash] ",
    "diagonal heads, previous-token heads, uniform heads.\n",
    "\[FilledCircle] Layers 8\[Dash]11: messier looking but doing more ",
    "abstract work \[LongDash] heads here often attend to specific ",
    "syntactic relationships that are hard to spot without an ",
    "interpretability tool.\n",
    "\[FilledCircle] Layer 12: returns to relatively simple patterns ",
    "because it has to produce the final output distribution."
  }],

  (* ----- Section 8: summary ----- *)

  section["8.  Summary and what comes next"],

  text[{
    "We finish Part 2 with six concrete deliverables, each reusable in ",
    "the parts that follow:"
  }],
  text[{
    "\[FilledSquare] ",
    StyleBox["A first-principles derivation of the ", FontWeight -> Bold],
    mathSqrtDk, " ",
    StyleBox["factor.", FontWeight -> Bold],
    " The dot product of two zero-mean unit-variance vectors of length ",
    mathDk, " has variance ", mathDk,
    " by independence; dividing by ",
    math[SqrtBox[SubscriptBox["d", "k"]]],
    " restores unit variance so the softmax does not saturate. The figure ",
    StyleBox["qkdotproduct_variance.png", FontFamily -> "Courier"],
    " confirms the prediction empirically at ",
    mathDk, " \[Element] {16, 64, 256}.\n",
    "\[FilledSquare] ",
    StyleBox["A live d_k dial.", FontWeight -> Bold],
    " Section 3.1's ",
    StyleBox["Manipulate", FontFamily -> "Courier"],
    " widget lets the reader slide ", mathDk,
    " from 4 to 1024 and toggle the ",
    mathSqrtDk,
    " scaling on or off, watching the histogram and a sample softmax row ",
    "update in real time \[LongDash] the variance argument as an active ",
    "document, not a static figure.\n",
    "\[FilledSquare] ",
    StyleBox["A causal mask.", FontWeight -> Bold],
    " ",
    StyleBox["causalMask[n]", FontFamily -> "Courier"],
    " plus a three-panel figure showing the mask, the masked scores, and ",
    "the row-wise softmax weights.\n",
    "\[FilledSquare] ",
    StyleBox["A single-head causal-attention NetGraph and a multi-head ",
            FontWeight -> Bold],
    StyleBox["block constructor.", FontWeight -> Bold],
    " Six and 4+nHeads nodes respectively; everything is a ",
    StyleBox["LinearLayer", FontFamily -> "Courier"], ", a ",
    StyleBox["FunctionLayer", FontFamily -> "Courier"], ", or a ",
    StyleBox["SoftmaxLayer", FontFamily -> "Courier"],
    " so every step is inspectable via ",
    StyleBox["NetExtract", FontFamily -> "Courier"], ".\n",
    "\[FilledSquare] ",
    StyleBox["A GPT-2 attention extractor.", FontWeight -> Bold],
    " ",
    StyleBox["gpt2AttentionWeights[gpt, ids, layer]", FontFamily -> "Courier"],
    " recovers the (12, seqLen, seqLen) tensor for any layer of pretrained ",
    "GPT-2 small by manually replaying the scaled-dot-product computation ",
    "on the extracted Q, K weights. Used to produce a 12-head gallery and ",
    "a 3-head specialisation figure that names three of the recurring ",
    "head archetypes (previous-token, self-attention diagonal, ",
    "approximately-uniform).\n",
    "\[FilledSquare] ",
    StyleBox["An interactive layer/head scrubber.", FontWeight -> Bold],
    " Section 7.1's ",
    StyleBox["Manipulate", FontFamily -> "Courier"],
    " reads the cached all-layer attention tensor (94 KB of ",
    StyleBox[".wxf", FontFamily -> "Courier"],
    ") and renders the heatmap for any (layer, head) pair on the fly. ",
    "Scrub up through the layers and watch the per-head differentiation ",
    "appear by layers 4\[Dash]7."
  }],

  text[{
    "In ", StyleBox["Part 3", FontWeight -> Bold],
    " we wrap this attention block in the canonical transformer ",
    "envelope: pre-norm, residual, feed-forward, residual, stacked ",
    math[SubscriptBox["n", "layers"]],
    " times.  We will derive the parameter count as a closed-form function ",
    "of (",
    mathDmodel, ", ",
    math[SubscriptBox["d", "ff"]], ", ",
    math[SubscriptBox["n", "heads"]], ", ",
    math[SubscriptBox["n", "layers"]], ", |V|), ",
    "build three model sizes (1M, 5M, 25M parameters) for use in Part 4, ",
    "and verify the formula against the actual ",
    StyleBox["Information", FontFamily -> "Courier"],
    " counts WL returns for each net."
  }],

  section["References"],

  text[{
    "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.\n",
    "Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners.\n",
    "Xiao, G. et al. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR.\n",
    "Raschka, S. (2024). Build a Large Language Model from Scratch. Manning."
  }]
};

nb = Notebook[
  mainCells,
  WindowSize -> {900, 750},
  StyleDefinitions -> "Default.nb",
  Saveable -> True
];

outPath = FileNameJoin[{nbDir, "Part02.nb"}];
Export[outPath, nb, "NB"];
Print["Wrote ", outPath, "  (", FileByteCount[outPath], " bytes)"];
