#!/usr/bin/env wolframscript
(* Assemble the Wolfram Community notebook for Part 1.

   Narrative mirrors POST.md, but every code block is a live,
   evaluatable Input cell, and every inline mathematical phrase
   uses TextData/BoxData so it typesets properly in the front end. *)

SetDirectory[DirectoryName[$InputFileName]];

partDir = ParentDirectory[];
dataDir = FileNameJoin[{partDir, "data"}];
figDir  = FileNameJoin[{partDir, "figures"}];
nbDir   = FileNameJoin[{partDir, "notebook"}];

If[!DirectoryQ[nbDir], CreateDirectory[nbDir]];

(* ---------- Cell-construction helpers ---------- *)

title[s_]      := Cell[s, "Title"];
subtitle[s_]   := Cell[s, "Subtitle"];
section[s_]    := Cell[s, "Section"];
subsection[s_] := Cell[s, "Subsection"];
text[s_String] := Cell[s, "Text"];
text[parts_List] := Cell[TextData[parts], "Text"];

input[s_String] := Cell[s, "Input", FormatType -> InputForm];
inputBox[box_]  := Cell[BoxData[box], "Input"];

(* Inline math fragment producing a Cell[BoxData[..]] for use in TextData. *)
math[box_] := Cell[BoxData[box], "Text", FontSize -> 14];

(* Common reusable math fragments. *)
mathDmodel := math[FractionBox["1", SqrtBox[SubscriptBox["d", "k"]]]];
mathRnd    := math[RowBox[{SuperscriptBox["\[DoubleStruckCapitalR]", RowBox[{"n", "\[Times]", "d"}]],
                          " \[Rule] ",
                          SuperscriptBox["\[DoubleStruckCapitalR]",
                            RowBox[{"n", "\[Times]", "|V|"}]]}]];
mathDmodelSimple := math[SubscriptBox["d", "model"]];

(* ---------- Cells ---------- *)

mainCells = Flatten @ {

  (* ----- title / front matter ----- *)

  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["Part 1: Tokeniser and embeddings"],

  text[{
    "This is the opening post of a eight-part series that builds a small ",
    "transformer language model entirely in Wolfram Language. Each part is ",
    "self-contained and runs on a Mac mini with no GPU. The series is a ",
    "companion-in-spirit to Sebastian Raschka's ",
    "\[OpenCurlyDoubleQuote]Build a Large Language Model from \
Scratch\[CloseCurlyDoubleQuote] (Manning, 2024), but every line of code, every ",
    "derivation, and the whole exposition is original. The intersection with ",
    "the book is the canonical mathematics of the transformer, which has been ",
    "the same since 2017, and the use of Tiny Shakespeare as the corpus, ",
    "which has been a community convention since Karpathy's ",
    StyleBox["char-rnn", FontSlant -> Italic], " (2015)."
  }],

  text[{
    "A language model is, in the end, a function ",
    mathRnd, ": a sequence of n continuous vectors of size d goes in, and ",
    "out comes a sequence of probability distributions over a vocabulary of ",
    "size |V|. Everything attention papers and GPT pre-prints argue about ",
    "\[LongDash] attention, normalisation, residuals \[LongDash] sits inside ",
    "that function. But there are two pieces outside it that decide what the ",
    "function is allowed to see, and they are the subject of this post: the ",
    "tokeniser, which decides what \[OpenCurlyDoubleQuote]an input\
\[CloseCurlyDoubleQuote] is in the first place, and the embedding layer, ",
    "which lifts the discrete tokens it produces into the continuous space ",
    "the model operates on."
  }],

  text[{
    "The choice of tokeniser fixes two numbers that play against each other ",
    "for the rest of the series: the vocabulary size |V|, which sets the ",
    "size of the output softmax and a non-trivial fraction of the model's ",
    "parameters, and the sequence length n, which controls how much the ",
    "attention block has to look at at once. Doubling |V| does not exactly ",
    "halve n, but it shrinks it; the empirical curve of how this trade-off ",
    "plays out is one of the deliverables of this post."
  }],

  text[{
    "By the end we will have a ",
    Cell[BoxData["bpeTokenise"], "Text", FontFamily -> "Courier",
         FontSize -> 14],
    " function and a ",
    Cell[BoxData["NetGraph"], "Text", FontFamily -> "Courier",
         FontSize -> 14],
    " embedder, both reusable by Parts 2 through 5, and we will have looked ",
    "at the geometry of token space before any training has happened so we ",
    "have a baseline for the \[OpenCurlyDoubleQuote]after\
\[CloseCurlyDoubleQuote] picture in Part 4."
  }],

  (* Hero illustration. *)
  inputBox[RowBox[{"Import", "[",
    "\"figures/illustration_pipeline.png\"", "]"}]],

  (* ----- Section 1: Setup ----- *)

  section["1.  Setup"],

  text[{
    "Load the tokeniser package and the Tiny Shakespeare corpus. Both live ",
    "alongside the notebook in the repository at ",
    StyleBox["part-01-tokeniser-and-embeddings/", FontFamily -> "Courier"],
    ". Running the notebook from a fresh kernel \[LongDash] there is no ",
    "external state to set up, no Python environment to provision, no GPU ",
    "driver to wrestle with. ",
    "The corpus is ~1.1 MB of public-domain Shakespeare play text, ",
    "downloaded the first time the build script runs and cached locally ",
    "from then on. Tiny Shakespeare has been a community convention since ",
    "Karpathy's char-rnn; we use it throughout the series so that every ",
    "result is reproducible from a 1 MB file rather than a multi-gigabyte ",
    "scrape."
  }],

  input["SetDirectory[NotebookDirectory[]]"],
  input["Get[\"src/Tokeniser.wl\"]"],
  input["text = Import[\"data/tinyshakespeare.txt\", \"Text\"];
StringLength[text]"],

  (* ----- Section 2: three slicing strategies ----- *)

  section["2.  Three ways to slice a sentence"],

  text[{
    "Before we get to byte-pair encoding, the two trivial baselines.  These ",
    "are useful both because they bracket the design space at its extremes ",
    "and because each is occasionally the right choice for a specific task."
  }],

  text[{
    StyleBox["Character-level. ", FontWeight -> Bold],
    "The vocabulary is just the set of characters that appear in the corpus. ",
    "On Tiny Shakespeare that is 65 characters (uppercase and lowercase ",
    "letters, digits, punctuation, whitespace, plus a newline). The sequence ",
    "length is the length of the text in characters, ~1,115,000 for the full ",
    "corpus. Models in this regime are easy to implement (no tokeniser, just ",
    "the text), but the sequence is so long that an attention block with ",
    "n\[ThinSpace]\[Times]\[ThinSpace]n cost has to do quadratically too much ",
    "work, and long-range dependencies are diluted across a million ",
    "positions. Character-level was the dominant choice in 2015\[Dash]2017 ",
    "(the era of char-rnn and ByteNet) and is rarely the right answer today, ",
    "except as a baseline."
  }],

  text[{
    StyleBox["Word-level. ", FontWeight -> Bold],
    "The vocabulary is the set of distinct whitespace-separated words; ",
    "on Tiny Shakespeare that is ~14,500 of them. The sequence length is ",
    "the number of word occurrences, ~252,000. The numbers are kinder to the ",
    "attention block, but every rare inflection or proper noun is its own ",
    "token, so the model sees most of them once or twice during training and ",
    "cannot generalise. Worse, anything outside the training vocabulary is ",
    "an out-of-vocabulary token and must be replaced with a fixed ",
    StyleBox["<UNK>", FontFamily -> "Courier"],
    " marker, losing all information about what the unseen word was."
  }],

  input["LLMAtoZ`Tokeniser`charBaselineVocabulary[text] // Length"],
  input["LLMAtoZ`Tokeniser`wordBaselineVocabulary[text] // Length"],

  text[{
    "Even the word-level definition needs a little care: do you count ",
    StyleBox["\"world!\"", FontFamily -> "Courier"], " as one token or two? ",
    "Splitting only on whitespace bundles punctuation into the previous ",
    "word, ballooning the vocabulary with near-duplicates ",
    "(\"world\", \"world!\", \"world,\", \"world?\"). The ",
    StyleBox["preTokenise", FontFamily -> "Courier"],
    " function in the package splits on whitespace and on the boundary ",
    "between alphabetic and non-alphabetic characters, so that punctuation ",
    "becomes its own pre-token. This pre-tokenisation step is what BPE ",
    "actually operates on; the algorithm never merges across a pre-token ",
    "boundary, which keeps it from inventing tokens like ",
    StyleBox["\"world!\"", FontFamily -> "Courier"],
    " out of distinct word and punctuation pieces."
  }],

  input["LLMAtoZ`Tokeniser`preTokenise[StringTake[text, 80]]"],

  (* ----- Section 3: BPE ----- *)

  section["3.  Byte-pair encoding, from first principles"],

  text[{
    "Byte-pair encoding (BPE) was originally a compression algorithm proposed ",
    "by Philip Gage in The C Users Journal (1994).  Its transplant to ",
    "neural-network text tokenisation is by Sennrich, Haddow and Birch (ACL ",
    "2016): the same iterative-merge procedure, but the symbols being merged ",
    "are characters or sub-word pieces rather than bytes of an arbitrary ",
    "file. GPT-2 uses a byte-level variant of the same idea (Radford et al., ",
    "2019), tiktoken is the production-grade modern reimplementation, and ",
    "essentially every transformer language model since 2018 ships some ",
    "flavour of BPE as its front door."
  }],

  text["The algorithm is four lines:"],

  text[{
    StyleBox["1.", FontWeight -> Bold], " Start with the alphabet of base symbols (characters).\n",
    StyleBox["2.", FontWeight -> Bold], " Across the pre-tokenised corpus, count every adjacent pair of symbols.\n",
    StyleBox["3.", FontWeight -> Bold], " Treat the most frequent pair as a new symbol from now on; record the merge.\n",
    StyleBox["4.", FontWeight -> Bold], " Repeat until the merge budget is spent or no pair occurs more than once."
  }],

  text[{
    "The vocabulary you end up with contains common short subword pieces ",
    "\[LongDash] ",
    StyleBox["th", FontFamily -> "Courier"], ", ",
    StyleBox["the", FontFamily -> "Courier"], ", ",
    StyleBox["ing", FontFamily -> "Courier"], ", ",
    StyleBox["ed", FontFamily -> "Courier"], ", ",
    StyleBox["Mr", FontFamily -> "Courier"], ", ",
    StyleBox["ou", FontFamily -> "Courier"],
    " \[LongDash] alongside the original characters.  A pre-token like ",
    StyleBox["\"speaking\"", FontFamily -> "Courier"], " typically tokenises to ",
    StyleBox["{\"speak\", \"ing</w>\"}", FontFamily -> "Courier"],
    ": the end-of-word marker ",
    StyleBox["</w>", FontFamily -> "Courier"],
    " always merges left into the final piece, because it is just another ",
    "symbol from the algorithm's point of view. A rare proper noun like ",
    StyleBox["\"Coriolanus\"", FontFamily -> "Courier"],
    " falls all the way back to characters in the worst case \[LongDash] ",
    StyleBox["{\"C\", \"or\", \"io\", \"lan\", \"us</w>\"}",
      FontFamily -> "Courier"],
    ". The property that no input is ever out-of-vocabulary is the deep ",
    "reason BPE has stuck around in 2026 long after the language model on ",
    "top of it has been redesigned half a dozen times."
  }],

  (* Illustration: BPE merge into subwords *)
  inputBox[RowBox[{"Import", "[",
    "\"figures/illustration_bpe_merge.png\"", "]"}]],

  subsection["3.1  The merge loop in Wolfram Language"],

  text[{
    "We represent each pre-token's frequency once and never duplicate work.  ",
    "The state of the algorithm is a list of ",
    StyleBox["{symbolList, frequency}", FontFamily -> "Courier"],
    " pairs, one entry per unique pre-token. Whitespace is dropped at ",
    "pre-tokenisation time; the end-of-word marker carries the word ",
    "boundary forward through the merges."
  }],

  text[{
    "Three Wolfram Language idioms make this concise, ",
    "and worth pausing on because they replace what is several pages of ",
    "bookkeeping in Python:"
  }],

  text[{
    "\[FilledCircle] ",
    StyleBox["Counts[preTokenise[text]]", FontFamily -> "Courier"],
    " gives the pre-token frequency map as a one-liner. There is no ",
    StyleBox["defaultdict(int)", FontFamily -> "Courier"], " loop.\n",
    "\[FilledCircle] ",
    StyleBox["Merge[..., Total]", FontFamily -> "Courier"],
    " is the right hammer for \[OpenCurlyDoubleQuote]many key\[Rule]value pairs with duplicate keys; ",
    "sum the values\[CloseCurlyDoubleQuote]. The keys can be arbitrary expressions including a ",
    "two-element list like ",
    StyleBox["{\"t\", \"h\"}", FontFamily -> "Courier"],
    ", so the pair-frequency map needs no encoding tricks.\n",
    "\[FilledCircle] ",
    StyleBox["SequenceReplace[syms, {a, b} -> a <> b]", FontFamily -> "Courier"],
    " does the in-place pair merge inside a symbol list. No two-pointer ",
    "loop, no off-by-one. ",
    "The result is a tokeniser fewer than 200 lines long, including ",
    "package boilerplate and usage messages, that trains 1000 merges on ",
    "Tiny Shakespeare in under 8 seconds on a 2023 Mac mini."
  }],

  input["trained = LLMAtoZ`Tokeniser`trainBPE[text, 1000]; // AbsoluteTiming"],
  input["Take[trained[\"MergeRules\"], 20]"],

  text[{
    "The very first merge is ",
    StyleBox["{\"e\", \"</w>\"} \[Rule] \"e</w>\"", FontFamily -> "Courier"],
    " \[LongDash] words ending in -e are the most common compressible ",
    "pattern in English. The second is ",
    StyleBox["{\"t\", \"h\"} \[Rule] \"th\"", FontFamily -> "Courier"],
    ", which then combines with ",
    StyleBox["e</w>", FontFamily -> "Courier"],
    " a few merges later to form ",
    StyleBox["the</w>", FontFamily -> "Courier"],
    " \[LongDash] the most frequent word in the corpus, learned in three ",
    "merges. This kind of order-of-discovery is what you would hope a ",
    "compression algorithm trained on English would produce, and is a useful ",
    "sanity check that the implementation is doing what it claims."
  }],

  text[{
    "An implementation detail worth flagging: the production-grade form ",
    "(implemented in the accompanying ",
    StyleBox["Tokeniser.wl", FontFamily -> "Courier"],
    ") maintains the pair-count table incrementally rather than rebuilding ",
    "it after each merge \[LongDash] only the pre-tokens that actually ",
    "contained the merged pair get revisited. The naive ",
    "rebuild-from-scratch version is conceptually identical and is what the ",
    "snippet above describes; the incremental version is what makes 4000 ",
    "merges on 1 MB of text finish in ~12 seconds rather than ~30 minutes."
  }],

  subsection["3.2  Encoding new text"],

  text[{
    "Once the merge rules are learned, tokenising new text is just applying ",
    "them in order to each pre-token. A subtlety: the merge order matters. ",
    "If we have learned ",
    StyleBox["t+h\[Rule]th", FontFamily -> "Courier"], ", ",
    StyleBox["th+e</w>\[Rule]the</w>", FontFamily -> "Courier"], ", and ",
    StyleBox["e+r\[Rule]er", FontFamily -> "Courier"],
    ", then encoding ",
    StyleBox["\"there\"", FontFamily -> "Courier"], " applies the merges in ",
    "the order they were learned: first ",
    StyleBox["{t,h,e,r,e,</w>}\[Rule]{th,e,r,e,</w>}", FontFamily -> "Courier"],
    ", and only then ",
    StyleBox["{th,e,r,e,</w>}\[Rule]{th,er,e,</w>}", FontFamily -> "Courier"],
    ". This is one of the places where BPE differs from a flat lookup ",
    "table \[LongDash] the encoding is path-dependent."
  }],

  input["mergeRules = trained[\"MergeRules\"];
LLMAtoZ`Tokeniser`bpeEncode[\"To be or not to be, that is the question.\", mergeRules]"],

  text[{
    "Common short words (",
    StyleBox["To", FontFamily -> "Courier"], ", ",
    StyleBox["be", FontFamily -> "Courier"], ", ",
    StyleBox["or", FontFamily -> "Courier"], ", ",
    StyleBox["the", FontFamily -> "Courier"],
    ") compress to a single token each; the less-common ",
    StyleBox["\"question\"", FontFamily -> "Courier"],
    " decomposes into three sub-word pieces ",
    StyleBox["{qu, es, tion</w>}", FontFamily -> "Courier"],
    ". A rarer proper noun like ",
    StyleBox["\"Coriolanus\"", FontFamily -> "Courier"],
    " would decompose further. Crucially, every input \[LongDash] including ",
    "out-of-corpus or non-English characters \[LongDash] is representable, ",
    "because the base alphabet of single characters always sits at the ",
    "bottom of the merge tree. No ",
    StyleBox["<UNK>", FontFamily -> "Courier"], ", ever."
  }],

  subsection["3.3  See the tokens"],

  text[{
    "The clearest way to convey what BPE has done is to look at the actual ",
    "tokens. Each coloured box below is one BPE token; the small black ",
    "square at the end of a box marks the end-of-word boundary ",
    "(the ",
    StyleBox["</w>", FontFamily -> "Courier"],
    " marker). Identical tokens get identical colours, so repeated short ",
    "words are easy to spot."
  }],

  input["LLMAtoZ`Tokeniser`colouredTokens[
  \"To be or not to be, that is the question.\", mergeRules]"],

  text[{
    "Drag the slider below to change the merge count and see the same ",
    "passage retokenised on the fly. At 0 merges, every box is a single ",
    "character; at 50, common bigrams collapse; at the maximum, several ",
    "whole words fit in a single box."
  }],

  input["Manipulate[
  LLMAtoZ`Tokeniser`colouredTokens[
    \"To be or not to be, that is the question.\",
    Take[mergeRules, n]],
  {{n, 500, \"merge rules used\"}, 0, Length[mergeRules], 50, Appearance -> \"Labeled\"},
  ControlPlacement -> Top, FrameMargins -> 0]"],

  text[{
    "And the same idea on a single word, animated so you can watch BPE ",
    "build the long-token form out of characters one merge at a time:"
  }],

  input["With[{prog = LLMAtoZ`Tokeniser`mergeProgression[
    \"speaking\", Take[mergeRules, 500]]},
  Animate[
    Framed[
      Style[Row[Riffle[prog[[step]], \"  \"]], 18, FontFamily -> \"Courier\"],
      FrameMargins -> 12],
    {{step, 1, \"merges applied\"}, 1, Length[prog], 1, Appearance -> \"Labeled\"},
    AnimationRate -> 1.5,
    ControlPlacement -> Top]
]"],

  text[{
    "What you should see (depending on how many merges have been learned): ",
    StyleBox["{s, p, e, a, k, i, n, g, </w>}", FontFamily -> "Courier"],
    " gradually collapsing into something like ",
    StyleBox["{speak, ing</w>}", FontFamily -> "Courier"],
    " after enough merges. The end-of-word marker survives all the way ",
    "through; the model needs that signal to know where one word ends and ",
    "the next begins."
  }],

  subsection["3.4  The Zipf shape of token frequency"],

  text[{
    "How often does each BPE token appear in Tiny Shakespeare? The answer ",
    "is the empirical Zipf curve \[LongDash] on a log-log plot, an ",
    "almost-straight line. The most frequent token (typically ",
    StyleBox["the</w>", FontFamily -> "Courier"],
    " or ",
    StyleBox["</w>", FontFamily -> "Courier"],
    " of some closed-class word) is dramatically more common than the ",
    "10th, which is dramatically more common than the 100th, and so on. ",
    "This is the structure BPE is exploiting: a tiny number of tokens ",
    "account for most of the corpus."
  }],

  input["allTokens = LLMAtoZ`Tokeniser`bpeEncode[text, mergeRules];
zipf = LLMAtoZ`Tokeniser`tokenZipf[allTokens];
ListLogLogPlot[zipf,
  Frame -> True,
  FrameLabel -> {\"rank\", \"frequency\"},
  PlotLabel -> \"Token-frequency Zipf curve on Tiny Shakespeare\",
  PlotStyle -> Directive[PointSize[0.003], ColorData[97][1]],
  ImageSize -> 500,
  GridLines -> Automatic
]"],

  text[{
    "If we fit a line to the log-log curve, the slope is the negative ",
    "Zipf exponent. Values close to 1 are the canonical \"natural ",
    "language\" regime. We will use that empirical fact later when we ",
    "talk about why language models work at all: small models cannot ",
    "afford to remember every token in their training data, but they can ",
    "afford to remember the head of the distribution well \[LongDash] and ",
    "the head is most of the probability mass."
  }],

  text[{
    "How big to make the vocabulary is the only real hyperparameter of the ",
    "tokeniser. GPT-2 uses 50,257 tokens; GPT-4 uses 100,256. For a Tiny ",
    "Shakespeare-scale model, 1,000 to 5,000 is plenty \[LongDash] the ",
    "trade-off curve in the next section is how we choose."
  }],

  (* ----- Section 4: vocab vs seq length curve ----- *)

  section["4.  The vocab\[Hyphen]sequence trade-off"],

  text[{
    "Train BPE with progressively more merges and watch what happens. At ",
    "zero merges, the BPE-tokenised corpus is one symbol per pre-token ",
    "character plus a single ",
    StyleBox["</w>", FontFamily -> "Courier"],
    " marker per pre-token \[LongDash] ",
    StyleBox["not", FontSlant -> Italic],
    " the raw corpus character count, because pre-tokenisation drops ",
    "whitespace before BPE ever sees it. The arithmetic is chars-in-pretokens ",
    "+ n-pretokens = 905,502 + 251,610 = 1,157,112 tokens at 0 merges, with ",
    "a vocabulary of 64 (the alphabet that actually appears inside ",
    "pre-tokens). Every additional merge grows the vocabulary by one and ",
    "shrinks the sequence by as many positions as it eliminates pair ",
    "occurrences. The shape of the resulting curve is the empirical ",
    "justification for preferring sub-word tokenisation to either extreme."
  }],

  input["Import[\"figures/vocab_vs_seqlen.png\"]"],

  text[{
    "The numerical picture: 50 merges already cuts the sequence to 769 k ",
    "tokens \[LongDash] a third shorter than the character baseline for the ",
    "price of 50 extra vocabulary entries. 500 merges gets us to 465 k, ",
    "below half. 4000 merges gets us to 299 k, still six times shorter than ",
    "character-level. The curve flattens because later merges chase ",
    "increasingly rare bigrams \[LongDash] the marginal vocabulary entry ",
    "gives less and less sequence-length reduction. The two extreme ",
    "baselines sit at the corners: character-level is the ",
    "cheap-vocabulary, expensive-sequence corner; word-level is the ",
    "cheap-sequence, expensive-vocabulary corner. A practical model picks ",
    "an interior point; Part 4 uses 1000 merges and a vocabulary of 1064."
  }],

  text[{
    "The information-theoretic shape of this curve is not accidental. As ",
    "long as the corpus has Zipfian word-frequency structure (which English ",
    "does, and which Shakespeare does in particular), the n-th most frequent ",
    "bigram has roughly 1/n the count of the most frequent one, so the ",
    "marginal compression from the n-th merge falls as 1/n. ",
    "Integrating, total compression grows as log(n) \[LongDash] which on a ",
    "log-log plot like the one above appears as the curve's gentle drift ",
    "downward toward the word-level corner."
  }],

  (* ----- Section 5: tokens to tensors ----- *)

  section["5.  From tokens to tensors"],

  text[{
    "A transformer does not see token IDs; it sees vectors. Once we have a ",
    "list of integer token IDs, we feed them through two embedding tables ",
    "in parallel:"
  }],

  text[{
    "\[FilledSmallCircle] ",
    StyleBox["Token embedding ", FontWeight -> Bold],
    math[SubscriptBox["E", "tok"]],
    " \[Element] ",
    math[SuperscriptBox["\[DoubleStruckCapitalR]",
      RowBox[{"|V|", "\[Times]", "d"}]]],
    ": row i is the vector for token-ID i. Learnable.\n",
    "\[FilledSmallCircle] ",
    StyleBox["Positional embedding ", FontWeight -> Bold],
    math[SubscriptBox["E", "pos"]],
    " \[Element] ",
    math[SuperscriptBox["\[DoubleStruckCapitalR]",
      RowBox[{SubscriptBox["n", "max"], "\[Times]", "d"}]]],
    ": row j is the vector for the j-th position. Also learnable."
  }],

  text[{
    "The model input is the elementwise sum ",
    math[RowBox[{
      SubscriptBox["E", "tok"], "[", "ids", "]",
      " + ",
      SubscriptBox["E", "pos"], "[", "positions", "]"}]],
    ". That sum is what flows into the first transformer block. ",
    "Why a sum rather than a concatenation? Two reasons. ",
    "First, with the dimensions kept aligned the transformer needs no extra ",
    "parameters to combine them; concatenation would double the input ",
    "dimension and require the next layer to be wider. ",
    "Second, the sum is what falls out naturally if you ask ",
    "\[OpenCurlyDoubleQuote]what is the right basis-free way to combine two pieces of \
information that live in the same vector space\[CloseCurlyDoubleQuote] \[LongDash] ",
    "either is just a vector, and their sum is a vector. The model is free ",
    "to learn token- and position-information directions that are nearly ",
    "orthogonal, which is what trained transformers tend to do in practice."
  }],

  text[{
    "Notice that we picked learned positional embeddings here, not the ",
    "sinusoidal ones from Vaswani et al. 2017 and not rotary positional ",
    "embeddings (RoPE, Su et al. 2021). Learned absolute is the simplest ",
    "choice and is what GPT-2 used. We will revisit this in Part 3 or 5: ",
    "learned positions cannot generalise beyond ",
    math[SubscriptBox["n", "max"]],
    ", whereas RoPE can; for Tiny Shakespeare at 256-token sequences, the ",
    "limitation does not bite."
  }],

  text["In Wolfram Language this is a three-node NetGraph:"],

  input["vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[trained];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[vocab];
embedder = LLMAtoZ`Tokeniser`tokenPositionEmbedder[Length[vocab], 64, 256,
  RandomSeeding -> 42];"],
  input["embedder"],

  text[{
    StyleBox["NetInitialize", FontFamily -> "Courier"],
    " populates the two embedding matrices with small Gaussian-distributed ",
    "random values. Until we train (Part 4), this is what the model sees: a ",
    "lookup of random vectors. ",
    "Pass a sequence of token IDs and a parallel sequence of position IDs, ",
    "and the network returns the (seqLen, dModel) input tensor:"
  }],

  input["ids = LLMAtoZ`Tokeniser`encodeWithIds[
  \"To be or not to be, that is the question.\", mergeRules, tokId];
ids"],
  input["LLMAtoZ`Tokeniser`embedTokens[embedder, ids] // Dimensions"],

  text[{
    "Twelve tokens in, a ",
    math[RowBox[{"12", "\[Times]", "64"}]],
    " tensor out. That tensor is exactly what Part 2 will take as input to ",
    "the attention block."
  }],

  subsection["5.1  Positional embeddings, as a picture"],

  inputBox[RowBox[{"Import", "[",
    "\"figures/illustration_pos_encoding.png\"", "]"}]],

  text[{
    "We can look at the positional-embedding matrix directly. Each row is ",
    "the vector that the model adds to any token sitting at that position. ",
    "Right now those vectors are random, but the structure of the matrix ",
    "(256 rows of length 64) is exactly the shape of the learned ",
    "positional embedder at the end of Part 4. The heatmap below shows ",
    "the first 32 positions; after training, similar positions will have ",
    "similar rows (because positions 5 and 6 do similar work in a sentence)."
  }],

  input["ArrayPlot[
  Normal[NetExtract[embedder, {\"positionEmbed\", \"Weights\"}]][[;; 32]],
  ColorFunction -> \"RedBlueTones\",
  FrameTicks -> {Range[32], None},
  ImageSize -> 600,
  PlotLabel -> \"Positional-embedding matrix, first 32 positions \[Times] 64 dims\"]"],

  (* ----- Section 6: geometry before training ----- *)

  section["6.  Geometry before training"],

  text[{
    "We have built a ",
    StyleBox["NetGraph", FontFamily -> "Courier"],
    " whose token-embedding row for ",
    StyleBox["\"the\"", FontFamily -> "Courier"],
    " is currently a vector of 64 Gaussian samples. Whatever similarity ",
    "structure the trained model will eventually have \[LongDash] ",
    "King near Queen, thee near thou, punctuation off in its own corner ",
    "\[LongDash] is not in there yet. It is useful to see this with our own ",
    "eyes so that we have a \[OpenCurlyDoubleQuote]before\[CloseCurlyDoubleQuote] ",
    "picture to compare against in Part 4. The figure below is the ",
    "conceptual prediction; the next two are the empirical version of it ",
    "on this very embedder."
  }],

  (* Illustration: random vs trained embedding space *)
  inputBox[RowBox[{"Import", "[",
    "\"figures/illustration_embedding_compare.png\"", "]"}]],

  text[{
    "On the left, at random initialisation, the labelled tokens are spread ",
    "across the cloud at random; there is no semantic structure because ",
    "the embedding matrix is just Gaussian noise. On the right, after the ",
    "training of Part 4, words of similar grammatical role and meaning ",
    "cluster: articles cluster with articles, pronouns with pronouns, ",
    "punctuation off by itself."
  }],

  text["Now the empirical version on the actual embedder we just built:"],

  input["Import[\"figures/embedding_geometry_random.png\"]"],

  text[{
    "The diagonal is dark (cosine similarity = 1; each token is identical ",
    "to itself), and the off-diagonal entries are uniform noise centred ",
    "near zero. There is a clean prediction here we can check: the ",
    "standard deviation of the cosine similarity between two independent ",
    "random unit vectors in d dimensions is ",
    math[FractionBox["1", SqrtBox["d"]]],
    ", which for d = 64 gives 0.125. Eyeballing the colour bar, the bulk ",
    "of the off-diagonal entries sit in [\[Minus]0.2, 0.2] \[LongDash] ",
    "consistent with that prediction. No two tokens are systematically ",
    "closer than any other pair, which is exactly the structurelessness ",
    "we expect."
  }],

  input["Import[\"figures/embedding_pca_random.png\"]"],

  text[{
    "A 2D PCA projection makes the same point visually: a Gaussian blob ",
    "with the eight highlighted high-frequency tokens scattered all over ",
    "it. This is the picture we will revisit in Part 4 after training: ",
    "the same plot will then show clear clusters, with grammatical and ",
    "semantic structure visible at a glance. Until then, the model knows ",
    "nothing."
  }],

  subsection["6.1  Interactive: pick any two tokens"],

  text[{
    "Pick two tokens from the vocabulary and look at the cosine similarity ",
    "of their (random-init) embedding vectors. Move the sliders. Most ",
    "pairs will sit near zero; the diagonal (a token compared with itself) ",
    "is exactly one. After training in Part 4 you can run the same widget ",
    "and the off-diagonal entries will no longer be random."
  }],

  input["With[{
    tw = Normal[NetExtract[embedder, {\"tokenEmbed\", \"Weights\"}]],
    candidateTokens =
      Take[ReverseSort[Counts[LLMAtoZ`Tokeniser`bpeEncode[
        StringTake[text, 5000], mergeRules]]], UpTo[40]] // Keys,
    cosSim = Function[{a, b}, (a.b) / (Norm[a] Norm[b])]},
  Manipulate[
    With[{a = Lookup[tokId, t1], b = Lookup[tokId, t2]},
      Column[{
        Style[\"cosine similarity:\", 12, Gray],
        Style[NumberForm[cosSim[tw[[a]], tw[[b]]], 4], 22, Bold,
              If[Abs[cosSim[tw[[a]], tw[[b]]]] > 0.3, Red, Black]],
        Spacer[12],
        ListPlot[{tw[[a]], tw[[b]]},
          PlotStyle -> {ColorData[97][1], ColorData[97][2]},
          PlotLegends -> {t1, t2}, Joined -> True,
          ImageSize -> 460, PlotLabel -> \"the two embedding vectors\"]
      }]],
    {{t1, First[candidateTokens], \"token 1\"}, candidateTokens},
    {{t2, candidateTokens[[2]], \"token 2\"}, candidateTokens},
    ControlPlacement -> Top]
  ]"],

  (* ----- Section 7: summary and onward ----- *)

  section["7.  Summary and what comes next"],

  text[{
    "We finish Part 1 with the following deliverables, each reusable in ",
    "the parts that follow:"
  }],

  text[{
    "\[FilledSquare] ",
    StyleBox["A from-scratch BPE tokeniser.", FontWeight -> Bold],
    " ",
    StyleBox["trainBPE", FontFamily -> "Courier"],
    " on the full 1.1 MB Tiny Shakespeare corpus finishes in ~12 s for ",
    "4000 merges thanks to incremental pair-count maintenance, and ",
    "produces a vocabulary of size ",
    math[RowBox[{"64 + ", "n", SubscriptBox["", "merges"]}]],
    ". ",
    StyleBox["bpeEncode", FontFamily -> "Courier"], ", ",
    StyleBox["bpeDecode", FontFamily -> "Courier"],
    ", and the integer-ID encode/decode pair are all exported from the ",
    "package.\n",
    "\[FilledSquare] ",
    StyleBox["An empirical vocabulary\[Hyphen]sequence trade-off curve.",
            FontWeight -> Bold],
    " The log-log plot in section 4 justifies the BPE choice with eight ",
    "concrete (vocab, seqLen) points spanning the character and word-level ",
    "extremes, and an information-theoretic ",
    StyleBox["log(n)", FontFamily -> "Courier"],
    " explanation of its shape on Zipfian text.\n",
    "\[FilledSquare] ",
    StyleBox["A live tokenisation widget.", FontWeight -> Bold],
    " Section 3.3's ",
    StyleBox["Manipulate", FontFamily -> "Courier"],
    " slider retokenises a fixed passage at every merge count from 0 to ",
    "4000, with identical tokens rendered in identical colours so the ",
    "reader can watch shorter pieces fuse into single tokens as the ",
    "vocabulary grows. The accompanying ",
    StyleBox["Animate", FontFamily -> "Courier"],
    " of ",
    StyleBox["mergeProgression[\"speaking\", ...]", FontFamily -> "Courier"],
    " plays the same idea on a single word, one merge at a time.\n",
    "\[FilledSquare] ",
    StyleBox["A Zipfian token-frequency curve.", FontWeight -> Bold],
    " The empirical rank/frequency log-log plot shows the head-heavy ",
    "distribution that justifies why a 1000-token vocabulary captures most ",
    "of the corpus mass.\n",
    "\[FilledSquare] ",
    StyleBox["A three-node embedder NetGraph.", FontWeight -> Bold],
    " ",
    StyleBox["tokenPositionEmbedder[vocabSize, dModel, maxSeqLen]",
            FontFamily -> "Courier"],
    " combines a learnable token-embedding lookup with a learnable ",
    "positional embedding by elementwise sum, producing the ",
    math[RowBox[{"(seqLen", "\[Times]", SubscriptBox["d", "model"], ")"}]],
    " tensor a transformer block expects.\n",
    "\[FilledSquare] ",
    StyleBox["A baseline picture of token-space geometry at random ",
            FontWeight -> Bold],
    StyleBox["initialisation.", FontWeight -> Bold],
    " Cosine-similarity heatmap + PCA(2) scatter + a ",
    StyleBox["Manipulate", FontFamily -> "Courier"],
    " widget letting the reader pick any two of the 40 most-frequent ",
    "tokens and inspect their similarity. Right now the geometry is ",
    "structureless noise \[LongDash] the standard deviation of pairwise ",
    "cosine similarity matches the theoretical ",
    math[FractionBox["1", SqrtBox["d"]]],
    " prediction. Part 4 will revisit this plot after training."
  }],

  text[{
    "In ",
    StyleBox["Part 2", FontWeight -> Bold],
    " we feed that tensor into the heart of the transformer: scaled ",
    "dot-product attention, derived from first principles \[LongDash] ",
    "including the ",
    math[FractionBox["1", SqrtBox[SubscriptBox["d", "k"]]]],
    " factor that is not arbitrary, but the unique scaling that keeps the ",
    "pre-softmax logits at unit variance. We then visualise attention ",
    "heatmaps on real text using a pretrained GPT-2 small (accessible in ",
    "WL via ",
    StyleBox["NetModel[\"GPT-2 Transformer Trained on WebText Data\"]",
      FontFamily -> "Courier"],
    ") so the reader sees real, learned attention structure before ",
    "training their own from scratch in Part 4."
  }],

  text[{
    "The remaining parts build the transformer block (Part 3), pre-train ",
    "with the option to scale up via ",
    StyleBox["RemoteBatchSubmit", FontFamily -> "Courier"],
    " (Part 4), implement five sampling strategies plus speculative ",
    "decoding (Part 5), instruction-tune the model on a small SFT mixture ",
    "(Part 6), apply GRPO-style RL on a verifiable task (Part 7), and ",
    "finish with an interactive chat widget powered by the final ",
    "post-trained model (Part 8). The whole series runs on a Mac mini by ",
    "default; the cloud path is opt-in."
  }],

  (* ----- References ----- *)

  section["References"],

  text[{
    "Gage, P. (1994). A New Algorithm for Data Compression. The C Users Journal, 12(2).\n",
    "Sennrich, R., Haddow, B., Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.\n",
    "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.\n",
    "Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2 technical report.)\n",
    "Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.\n",
    "Raschka, S. (2024). Build a Large Language Model from Scratch. Manning.\n",
    "Karpathy, A. (2015). char-rnn. github.com/karpathy/char-rnn. Tiny Shakespeare lives at data/tinyshakespeare/input.txt in that repo."
  }]
};

nb = Notebook[
  mainCells,
  WindowSize -> {900, 750},
  StyleDefinitions -> "Default.nb",
  Saveable -> True
];

outPath = FileNameJoin[{nbDir, "Part01.nb"}];
Export[outPath, nb, "NB"];
Print["Wrote ", outPath, "  (", FileByteCount[outPath], " bytes)"];
