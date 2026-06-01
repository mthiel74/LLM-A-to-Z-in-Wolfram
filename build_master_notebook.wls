#!/usr/bin/env wolframscript
(* ============================================================
   build_master_notebook.wls

   Assembles LLM_A_to_Z.nb at the repo root.  The notebook is the
   single canonical deliverable; the eight per-part notebooks are
   retired in favour of this consolidated walkthrough.

   The notebook is meant to be opened in Wolfram Desktop, with the
   repo root as the working directory (NotebookDirectory[] takes
   care of this).  All imports reference part-XX/... paths directly;
   no symlinks needed.
   ============================================================ *)

SetDirectory[DirectoryName[$InputFileName]];

(* Load LLMAtoZ.wl so evalInput[] can call its functions at build time. *)
Quiet @ Get["LLMAtoZ.wl"];

(* ---------- Cell-construction helpers ---------- *)

title[s_]      := Cell[s, "Title"];
subtitle[s_]   := Cell[s, "Subtitle"];
chapter[s_]    := Cell[s, "Chapter"];
section[s_]    := Cell[s, "Section"];
subsection[s_] := Cell[s, "Subsection"];
subsubsection[s_] := Cell[s, "Subsubsection"];
para[s_String] := Cell[s, "Text"];
para[parts_List] := Cell[TextData[parts], "Text"];

(* When the script is invoked with "--evaluated", we redefine input[] to
   evaluate the code and embed an exported PNG of the result via the
   Front End.  Default mode (no flag) writes plain Input cells, leaving
   LLM_A_to_Z.nb unchanged. *)
$evalMode = MemberQ[$ScriptCommandLine, "--evaluated"];
$evalFigDir = FileNameJoin[{Directory[], "discussion-model",
                             "figures", "built_eval"}];
$evalCellIndex = 0;
If[$evalMode && !DirectoryQ[$evalFigDir], CreateDirectory[$evalFigDir]];

(* Cells with code matching any of these patterns stay as plain Input
   (no evaluation) even in --evaluated mode.  They're too heavy to
   evaluate at build time. *)
$skipRenderPatterns = {
  "NetTrain[", "NetModel[",
  "loadGpt2Small", "loadGpt2Medium", "loadPrebuiltGpt2",
  "buildGpt2Architecture", "RemoteBatchSubmit", "RemoteBatchJobObject",
  "RemoteBatchSubmissionEnvironment",
  "trainGptModel", "trainOvernight", "trainSft", "runGrpoEpoch",
  "discussion_medium_20k.wlnet", "discussion_large_10k.wlnet",
  "gpt2_sft_chat.wlnet",
  "tiny_gpt_trained.wlnet", "smallcpu_gpt_trained.wlnet",
  "charlevel_shakespeare_trained.wlnet", "submitRemotePretraining",
  "CloudConnect", "URLDownload",
  "buildTrainableTrunk",
  "chatWidget",
  "submit_gpu", "CloudPut", "CloudObject",
  "GPT-2 Transformer", "FileExistsQ[sftCkpt"};

(* Manipulate / Animate can't be rasterized as live interactive widgets
   in a headless render kernel -- they show "red box" errors because
   slider variables aren't bound.  Strategy: at build time, replace
   Manipulate[body, {{n, default, ...}, ...}, ...] with `body /. n ->
   default`, i.e. evaluate the body at the *default slider position*
   and rasterize THAT static snapshot.  The reader still gets the
   original interactive Manipulate cell to evaluate.                 *)
$manipulateRules = {
  HoldPattern[Manipulate[body_, sliderSpecs___]] :>
    With[
      {rules = Cases[Hold[{sliderSpecs}],
         ({{var_Symbol, default_, ___}, ___} | {var_Symbol, default_, ___}) :>
           (var -> default),
         {0, Infinity}]},
      (* Hold[body] keeps the body un-evaluated (body is a pattern var
         bound to the held Manipulate body); ReplaceAll substitutes
         the slider vars inside the Hold structurally; ReleaseHold
         then evaluates the substituted body. *)
      ReleaseHold[Hold[body] /. rules]],
  HoldPattern[Animate[body_, sliderSpec_, ___]] :>
    With[
      {rule = Cases[Hold[sliderSpec],
         ({{var_Symbol, default_, ___}, ___} | {var_Symbol, default_, ___}) :>
           (var -> default),
         {0, Infinity}]},
      ReleaseHold[Hold[body] /. rule]]};

shouldRenderQ[code_String] := !AnyTrue[$skipRenderPatterns,
  StringContainsQ[code, #] &];

plainInput[s_String] := Cell[s, "Input", FormatType -> InputForm];

(* renderCell: evaluate the code in this kernel, then render the output
   via UsingFrontEnd[Rasterize[...]] so we get pixel data instead of
   wolframscript's TextForm "-Graphics-" placeholder.  Save the result
   to a PNG and embed via figureCell.                                    *)
renderCell[code_String] := Module[
  {expr, evald, raster, dims, path, relPath, str},
  If[!shouldRenderQ[code], Return[{plainInput[code]}]];
  expr = ToExpression["CompoundExpression[\n" <> code <> "\n]",
                       InputForm, Hold];
  (* Substitute Manipulate / Animate with the body at default slider
     value -- live Manipulates fail when opened because they reference
     build-kernel state the reader hasn't loaded yet.  The reader still
     gets the original interactive Input cell to evaluate manually.

     The Block[] temporarily unbinds the common slider variable names
     so they don't evaluate to globally-bound values when Cases captures
     them.  Section 3.3 defines `head` globally (as a NetGraph), which
     would collide with the slider variable in Section 3.5.1's
     Manipulate without this. *)
  evald = Quiet @ Check[
    Block[{layer, head, T, step, dk, useScaling},
      expr = expr /. $manipulateRules;
      ReleaseHold[expr]],
    $Failed];
  Which[
    evald === Null,
      {plainInput[code]},
    evald === $Failed,
      {plainInput[code], Cell["(* evaluation failed at build time *)",
        "Output", FontSlant -> Italic, FontColor -> Gray]},
    MatchQ[Head[evald],
       Integer | Real | Rational | Complex | String | Symbol |
       True | False],
      (* Plain numbers / short strings: small compact text. *)
      {plainInput[code],
       Cell[ToString[evald, OutputForm], "Output",
            CellLabel -> "Out= ",
            FontFamily -> "Courier", FontSize -> 12]},
    MatchQ[evald, _List | _Association] && ByteCount[evald] < 600,
      (* Small lists / associations: compact 12pt Courier. *)
      {plainInput[code],
       Cell[ToString[evald, OutputForm, PageWidth -> 90], "Output",
            CellLabel -> "Out= ",
            FontFamily -> "Courier", FontSize -> 12]},
    True,
      $evalCellIndex += 1;
      raster = Quiet @ Check[
        UsingFrontEnd[Rasterize[evald, ImageResolution -> 100,
                                 Background -> White]],
        $Failed];
      If[!ImageQ[raster] || raster === $Failed,
        str = Quiet @ ToString[evald, OutputForm, PageWidth -> 90];
        Return[
          If[StringQ[str] && StringLength[str] < 500,
            {plainInput[code],
             Cell[str, "Output", CellLabel -> "Out= ",
                  FontFamily -> "Courier", FontSize -> 11]},
            {plainInput[code]}]]];
      dims = ImageDimensions[raster];
      (* If Rasterize produced a tiny "text-form" image (the raster of
         a literal -Graphics- placeholder), prefer text. *)
      If[Times @@ dims < 2000,
        str = Quiet @ ToString[evald, OutputForm, PageWidth -> 90];
        If[StringQ[str] && StringLength[str] < 500,
          Return[{plainInput[code],
                  Cell[str, "Output", CellLabel -> "Out= ",
                       FontFamily -> "Courier", FontSize -> 11]}]]];
      path = FileNameJoin[{$evalFigDir,
                            "out_" <> ToString[$evalCellIndex] <> ".png"}];
      Export[path, raster];
      relPath = "discussion-model/figures/built_eval/out_" <>
                 ToString[$evalCellIndex] <> ".png";
      Flatten @ {plainInput[code], figureCell[relPath]}]];

input[s_String] := If[$evalMode, renderCell[s], plainInput[s]];
inputBox[box_]  := Cell[BoxData[box], "Input"];

(* evalInput[code]: evaluate code in the build kernel's global context
   and emit BOTH the Input cell AND a baked-in Output cell, so the
   reader sees the result without having to evaluate.

   Use this for *light* cells (parameter counts, dataset previews,
   small graphics).  For heavy cells (NetTrain calls, NetModel pulls
   over the network, training loops) keep plain `input[...]` so the
   reader can choose when to run them.

   If the evaluation fails, the output cell renders the error message
   instead of $Failed so the build doesn't silently hide a bug.        *)
(* Compact pretty-printed string for text-like outputs (numbers, lists,
   short associations, Dimensions tuples).  Anything bigger than a few
   hundred characters of InputForm goes through evalImage instead.       *)
$evalCounter = 0;
$buildFigDir = FileNameJoin[{Directory[], "discussion-model", "figures", "built"}];
If[!DirectoryQ[$buildFigDir], CreateDirectory[$buildFigDir]];

evalInput[code_String] := Module[
  {expr, evald, str},
  (* Wrap the code in CompoundExpression so multi-statement cells parse
     as a single Hold[..] argument instead of Hold[a, b, c] (which would
     ReleaseHold to a Sequence and break Head[evald]). *)
  expr = ToExpression["CompoundExpression[\n" <> code <> "\n]",
                       InputForm, Hold];
  evald = Quiet @ Check[ReleaseHold[expr], $Failed];
  Which[
    evald === Null,
      {input[code]},
    evald === $Failed,
      {input[code], Cell["(* evaluation failed at build time *)",
        "Output", FontSlant -> Italic, FontColor -> Gray]},
    True,
      (* Try text output first, but only commit if it's short and clean
         (no nested NetArray / large NumericArray blobs).  Anything
         else rasterizes via PNG file -> figureCell. *)
      (* Fast path: small-ByteCount values that ToString to a short
         string get embedded as text.  Everything else (NetGraphs with
         millions of weight bytes, Graphics objects, Images,
         Manipulates, etc.) goes through evalImage which rasterizes
         and JPEG-compresses via figureCell. *)
      If[ByteCount[evald] > 8000,
        evalImage[code, evald],
        str = Quiet @ ToString[evald, OutputForm, PageWidth -> 90];
        If[StringQ[str] && StringLength[str] < 600,
          {input[code],
           Cell[str, "Output", CellLabel -> "Out= ",
                FontFamily -> "Courier", FontSize -> 11,
                CellMargins -> {{Inherited, Inherited}, {5, 5}}]},
          evalImage[code, evald]]]]];

(* Rasterize the evaluation result to a PNG on disk and embed via
   figureCell so the JPEG-compression path kicks in.  ~80 KB per cell. *)
evalImage[code_String, evald_] := Module[{raster, path},
  $evalCounter += 1;
  raster = Quiet @ Check[
    UsingFrontEnd[Rasterize[evald, ImageResolution -> 100]],
    $Failed];
  If[raster === $Failed,
    Return[{input[code], Cell["(* could not rasterize output *)",
      "Output", FontSlant -> Italic, FontColor -> Gray]}]];
  path = FileNameJoin[{$buildFigDir, "eval_" <> ToString[$evalCounter] <> ".png"}];
  Export[path, raster];
  Flatten @ {
    input[code],
    figureCell["discussion-model/figures/built/eval_" <>
               ToString[$evalCounter] <> ".png"]}];
caption[s_]     := Cell[s, "Caption", FontSlant -> Italic, FontSize -> 11];
math[box_] := Cell[BoxData[box], "Text", FontSize -> 14];
displayMath[box_] := Cell[BoxData[box], "DisplayFormula",
  TextAlignment -> Center, FontSize -> 15];

(* Image cell that *embeds* the figure directly into the notebook so it
   shows without the reader having to evaluate anything.  Reads the file
   at build time, downsamples to a 500-px-wide image (to keep the .nb
   file size reasonable), and wraps as a static Output cell.  Falls back
   to a placeholder if the file isn't there.                              *)
figureCell[path_String, caption_:""] := Module[{img, jpegBytes, smallImg, boxes},
  If[FileExistsQ[path],
    img = Import[path];
    (* Resize to a generous notebook-friendly width (1000 px) and keep
       the aspect ratio.  Then re-encode through JPEG quality 75 to cut
       the embedded byte budget by roughly 8x compared to the default
       PNG-style raster ToBoxes produces.  Visually indistinguishable
       on a normal display; lets the whole notebook stay under 20 MB
       even with ~30 figures inline. *)
    img = ImageResize[img, {1000, Automatic}];
    jpegBytes = ExportByteArray[img, "JPEG", "Quality" -> 78];
    smallImg = ImportByteArray[jpegBytes, "JPEG"];
    boxes = ToBoxes[smallImg];
    Flatten @ {
      Cell[BoxData[boxes], "Output",
           TextAlignment -> Center,
           CellMargins -> {{Inherited, Inherited}, {5, 10}},
           CellLabel -> ""],
      If[caption =!= "",
         para[{StyleBox[caption, FontSlant -> Italic, FontSize -> 11]}],
         Nothing]},
    Flatten @ {
      Cell["[image not found at build time: " <> path <> "]", "Text",
           FontColor -> RGBColor[0.7, 0.2, 0.2]],
      If[caption =!= "",
         para[{StyleBox[caption, FontSlant -> Italic, FontSize -> 11]}],
         Nothing]}]];

(* Reusable math fragments *)
mathDmodel := math[SubscriptBox["d", "model"]];
mathDhead  := math[SubscriptBox["d", "head"]];
mathDk     := math[SubscriptBox["d", "k"]];
mathDFF    := math[SubscriptBox["d", "ff"]];
mathNHeads := math[SubscriptBox["n", "heads"]];
mathNLayers := math[SubscriptBox["n", "layers"]];
mathV      := math[RowBox[{"|", "V", "|"}]];
mathSqrtDk := math[FractionBox["1", SqrtBox[SubscriptBox["d", "k"]]]];

(* Load progress-log for the Section 9 gallery *)
progressLogPath = "part-04-pretraining/data/charlevel_progress_log.wxf";
progressLog = If[FileExistsQ[progressLogPath],
  Import[progressLogPath], <||>];
progressRounds = If[AssociationQ[progressLog],
  SortBy[Values[progressLog], #["round"] &], {}];
Print["Progress log has ", Length[progressRounds], " rounds."];

progressEntry[entry_Association] := With[
  {round = entry["round"],
   loss  = N @ entry["loss"],
   secs  = N @ entry["wallSeconds"],
   sROMEO  = entry["samples"]["ROMEO:"]["T=0.8"],
   sHAMLET = entry["samples"]["HAMLET:\n"]["T=0.8"]},
  Flatten @ {
    Cell[TextData[{
      StyleBox["Round " <> ToString[round], FontWeight -> Bold,
               FontSize -> 13, FontColor -> RGBColor[0.4, 0.2, 0.5]],
      "  \[Bullet]  loss = ",
      ToString[NumberForm[loss, {Infinity, 3}]],
      "  \[Bullet]  perplexity = ",
      ToString[NumberForm[Exp[loss], {Infinity, 2}]],
      "  \[Bullet]  wall = ",
      ToString[NumberForm[secs/60., {Infinity, 1}]], " min"
    }], "Subsubsection"],
    Cell[sROMEO,  "Program", FontFamily -> "Courier", FontSize -> 11],
    Cell[sHAMLET, "Program", FontFamily -> "Courier", FontSize -> 11]
  }
];

(* ============================================================
   CELLS
   ============================================================ *)

cells = Flatten @ {

  (* ============ Front matter ============ *)

  title["Build a Large Language Model from Scratch \[LongDash] in Wolfram Language"],
  subtitle["A single-notebook walkthrough of the full pipeline, from byte-pair encoding to chat"],

  para[{
    "This notebook is a complete, self-contained tour of how to build a ",
    "small but functional transformer-based language model entirely in ",
    "Wolfram Language.  It walks every piece of the modern LLM stack \
\[LongDash] tokenisation, embeddings, attention, the transformer ",
    "block, parameter accounting, pre-training, sampling, supervised ",
    "fine-tuning, LoRA, classification, reinforcement learning, and an ",
    "interactive chat widget \[LongDash] in one linear evaluation."
  }],

  para[{
    "The final deliverable is a chat-style model you can actually talk ",
    "to inside Wolfram Desktop.  The whole pipeline runs on a Mac mini ",
    "in under a day of CPU time; a cloud path (",
    StyleBox["RemoteBatchSubmit", FontFamily -> "Courier"],
    ") is sketched for scaling up."
  }],

  para[{
    "The notebook is companion-in-spirit to Sebastian Raschka's ",
    StyleBox["Build a Large Language Model from Scratch",
             FontSlant -> Italic],
    " (Manning, 2024), and goes further into post-training territory ",
    "covered by Andrej Karpathy's ",
    StyleBox["nanochat", FontSlant -> Italic],
    " (2025).  Every line of code, every derivation, and the whole ",
    "exposition is original; the only overlap with the book is the ",
    "canonical mathematics of the transformer, which has been the same ",
    "since Vaswani et al. (2017)."
  }],

  section["What you will build"],

  para[{
    "Concretely, after evaluating this notebook end-to-end you will have ",
    "on disk:"
  }],
  para[{
    "\[FilledSquare] A from-scratch byte-pair-encoding tokeniser learned ",
    "on Tiny Shakespeare (~4000 token vocabulary).\n",
    "\[FilledSquare] A character-level tokeniser (~65 symbol vocabulary) ",
    "for the small-model regime.\n",
    "\[FilledSquare] A configurable ",
    StyleBox["gptModel[d, h, L, dFF, V, n]", FontFamily -> "Courier"],
    " constructor that builds the canonical pre-norm transformer at any ",
    "size from 1M parameters (toy) up to GPT-2 small at 124M parameters ",
    "(weight-tying enabled).\n",
    "\[FilledSquare] Two trained checkpoints: a 1.8M-parameter char-",
    "level model that produces Shakespeare-flavoured prose, and a ",
    "124M-parameter fine-tuned GPT-2 that speaks fluent conversational ",
    "English.\n",
    "\[FilledSquare] Five sampling strategies (greedy, temperature, ",
    "top-k, top-p, beam) and speculative decoding, all from scratch.\n",
    "\[FilledSquare] Three fine-tuning regimes (full SFT, LoRA, ",
    "classification head).\n",
    "\[FilledSquare] A GRPO-style RL post-training loop.\n",
    "\[FilledSquare] An interactive chat widget powered by the trained ",
    "model.\n",
    "\[FilledSquare] A working understanding of why every part of the ",
    "pipeline is the way it is."
  }],

  section["The 30,000-foot view"],

  figureCell["part-01-tokeniser-and-embeddings/figures/illustration_pipeline.png",
    "Figure 0.1.  The pipeline this notebook implements end-to-end: text becomes BPE tokens; tokens become integer IDs; two embedding layers turn IDs into vectors; a stack of transformer blocks processes the vectors; the head projects back to a probability distribution over the vocabulary."],

  para[{
    "Every modern language model \[LongDash] GPT-2, Llama, Mistral, the ",
    "model you talk to in the wild \[LongDash] is fundamentally the same ",
    "function: a sequence of ",
    math["n"], " integer token IDs goes in, a sequence of probability ",
    "distributions over a vocabulary of size ", mathV,
    " comes out.  What changes between systems is the size of the ",
    "matrices and the training regime, not the shape of the function."
  }],
  para[{
    "The notebook proceeds linearly through the pipeline above.  Each ",
    "section introduces one piece, derives whatever needs deriving, ",
    "implements it as a Wolfram Language ",
    StyleBox["NetGraph", FontFamily -> "Courier"], ", and uses it.  By ",
    "the end of Chapter ", StyleBox["4", FontWeight -> Bold],
    " you have a model architecture; by Chapter ",
    StyleBox["5", FontWeight -> Bold],
    " you have a trained model that produces text; by Chapter ",
    StyleBox["7", FontWeight -> Bold],
    " you have a model that follows instructions and can be chatted ",
    "with from inside the notebook."
  }],

  section["Setup"],

  para[{
    "The notebook expects to be opened from its repository directory; ",
    StyleBox["NotebookDirectory[]", FontFamily -> "Courier"],
    " is used to anchor every file path.  No global PATH or environment ",
    "variable is required."
  }],

  input["SetDirectory[NotebookDirectory[]]"],

  para[{
    "All of the auxiliary functions used across the notebook live in a ",
    "single companion file, ",
    StyleBox["LLMAtoZ.wl", FontFamily -> "Courier"],
    ", at the repo root.  Each chapter's helpers live in their own ",
    "context (",
    StyleBox["LLMAtoZ`Tokeniser`", FontFamily -> "Courier"], ", ",
    StyleBox["LLMAtoZ`Attention`", FontFamily -> "Courier"],
    ", \[Ellipsis], ",
    StyleBox["LLMAtoZ`Discussion`", FontFamily -> "Courier"],
    ") so you can tell which chapter a symbol comes from.  ",
    "If you'd rather see each chapter's source on its own, the file is ",
    "auto-generated from the per-part packages in ",
    StyleBox["part-NN/src/", FontFamily -> "Courier"], "; run ",
    StyleBox["wolframscript -file build_LLMAtoZ_wl.wls", FontFamily -> "Courier"],
    " to regenerate."
  }],

  input["Get[\"LLMAtoZ.wl\"]"],

  para[{
    "Load the corpus.  Tiny Shakespeare is a 1.1-megabyte concatenation ",
    "of public-domain Shakespeare play text, originally assembled by ",
    "Andrej Karpathy for ",
    StyleBox["char-rnn", FontSlant -> Italic],
    " (2015) and used as a benchmark across the literature ever since.  ",
    "It is small enough to train on, large enough that BPE produces a ",
    "non-trivial vocabulary, and out of copyright."
  }],

  input["text = Import[\"part-01-tokeniser-and-embeddings/data/tinyshakespeare.txt\", \"Text\"];
{StringLength[text], \"characters\"}"],

  para[{
    "About 1.1 million characters.  About 250,000 whitespace-separated ",
    "words.  65 distinct characters (upper- and lower-case letters, ",
    "digits, punctuation, whitespace, newline).  Three numbers that ",
    "will reappear throughout the notebook."
  }],

  (* ============ Part 1: Tokenisation ============ *)

  chapter["1.  Tokenisation: turning text into integers"],

  para[{
    "A language model does not see characters or words.  It sees ",
    "integers.  The tokeniser is the function that converts between the ",
    "two: a deterministic mapping ",
    math[RowBox[{"text", " \[RightArrow] ", SuperscriptBox["\[DoubleStruckCapitalN]", "n"]}]],
    " sending text to a length-",
    math["n"],
    " sequence of integer IDs in ",
    math[RowBox[{"{", "1", ",", " ", "\[Ellipsis]", ",", " ", "|V|", "}"}]],
    ", and back.  The choice of tokeniser is the choice of two numbers ",
    "that play against each other for the rest of the pipeline: the ",
    "vocabulary size ", mathV, " and the sequence length ", math["n"],
    "."
  }],

  section["1.1  Three slicing strategies"],

  para[{
    "Three obvious tokenisation schemes bracket the design space."
  }],
  para[{
    StyleBox["Character-level. ", FontWeight -> Bold],
    "Each character is its own token.  Tiny Shakespeare gives ",
    mathV, " = 65, ", math["n"],
    " = 1.1 million.  Tiny vocabulary, gigantic sequence; the attention ",
    "block, which is ",
    math[RowBox[{"O", "(", SuperscriptBox["n", "2"], ")"}]],
    " expensive, spends almost all its compute on positional bookkeeping."
  }],
  para[{
    StyleBox["Word-level. ", FontWeight -> Bold],
    "Each whitespace-separated word is its own token.  Tiny Shakespeare ",
    "gives ", mathV, " \[TildeEqual] 14,500, ", math["n"], " \[TildeEqual] 252,000.  ",
    "Large vocabulary, manageable sequence; but every rare inflection ",
    "and proper noun gets a fresh ID the model sees once or twice in ",
    "training, and anything outside the vocabulary is an unrecoverable ",
    StyleBox["<UNK>", FontFamily -> "Courier"], "."
  }],
  para[{
    StyleBox["Subword (BPE). ", FontWeight -> Bold],
    "An interior point: learn a vocabulary by repeatedly merging the ",
    "most frequent adjacent pair of symbols.  Common short words ",
    "collapse to a single token; rare words decompose into a handful of ",
    "sub-word pieces; every input \[LongDash] including out-of-corpus ",
    "characters \[LongDash] is representable, because the base alphabet ",
    "of single characters always sits at the bottom of the merge tree.  ",
    "No ", StyleBox["<UNK>", FontFamily -> "Courier"], ", ever.  ",
    "This is the algorithm GPT-2, GPT-3, Llama, and tiktoken all use."
  }],

  section["1.2  Byte-pair encoding, from first principles"],

  para[{
    "BPE was originally a compression algorithm proposed by Philip ",
    "Gage in 1994.  Its transplant to neural-network text tokenisation ",
    "is by Sennrich, Haddow and Birch (ACL 2016).  The four-line ",
    "algorithm:"
  }],
  para[{
    StyleBox["1.", FontWeight -> Bold], " Start with the alphabet of base symbols (characters).\n",
    StyleBox["2.", FontWeight -> Bold], " Across the pre-tokenised corpus, count every adjacent pair of symbols.\n",
    StyleBox["3.", FontWeight -> Bold], " Treat the most frequent pair as a new symbol from now on; record the merge.\n",
    StyleBox["4.", FontWeight -> Bold], " Repeat until the merge budget is spent."
  }],
  figureCell["part-01-tokeniser-and-embeddings/figures/illustration_bpe_merge.png",
    "Figure 1.1.  A pre-token like \"speaking\" decomposes into a small number of sub-word pieces plus the end-of-word marker."],
  figureCell["part-01-tokeniser-and-embeddings/figures/illustration_merge_tree.png",
    "Figure 1.2.  The BPE merge tree for the token 'the': three pairwise merges combine four characters (t, h, e, </w>) into one merged token, then the same three rules are reapplied to every occurrence of 'the' in the corpus."],

  para[{
    "Three Wolfram Language idioms make the implementation concise:"
  }],
  para[{
    "\[FilledCircle] ",
    StyleBox["Counts[preTokenise[text]]", FontFamily -> "Courier"],
    " builds the pre-token frequency map in one expression.\n",
    "\[FilledCircle] ",
    StyleBox["Merge[..., Total]", FontFamily -> "Courier"],
    " sums values over duplicate keys; the keys can be arbitrary ",
    "expressions including a two-element list ",
    StyleBox["{\"t\", \"h\"}", FontFamily -> "Courier"],
    ".\n",
    "\[FilledCircle] ",
    StyleBox["SequenceReplace[syms, {a, b} -> a <> b]",
             FontFamily -> "Courier"],
    " merges a pair inside a symbol list.  No two-pointer loop, no off-by-one."
  }],

  para[{
    "Train BPE on the corpus.  4000 merges takes ~12 seconds on a Mac ",
    "mini thanks to incremental pair-count maintenance."
  }],
  input["bpe = Import[\"part-01-tokeniser-and-embeddings/data/bpe_trained.mx\"];
mergeRules = bpe[\"MergeRules\"];
vocab = LLMAtoZ`Tokeniser`buildBPEVocabulary[bpe];
tokId = LLMAtoZ`Tokeniser`tokenToIdAssociation[vocab];
{Length[mergeRules], Length[vocab]}"],
  para[{
    "4000 merge rules, 4064 final vocabulary entries (64 base ",
    "characters + 4000 merged tokens)."
  }],

  para[{
    "The first few merge rules are the highest-frequency bigrams of ",
    "Shakespeare:"
  }],
  input["Take[mergeRules, 8]"],
  para[{
    "Words ending in -e, the bigram th, ' ' followed by certain words, ",
    "etc.  In any English corpus the first hundred BPE merges all look ",
    "like this."
  }],

  section["1.3  See the tokens"],

  para[{
    "The clearest way to convey what BPE has done is to look at the ",
    "actual output.  Each coloured box is one BPE token; the small ",
    "black square at the end of a box marks the end-of-word boundary ",
    "(the ",
    StyleBox["</w>", FontFamily -> "Courier"],
    " marker).  Identical tokens get identical colours."
  }],
  input["LLMAtoZ`Tokeniser`colouredTokens[
  \"To be or not to be, that is the question.\", mergeRules]"],

  para[{
    "Drag the slider to retokenise the same passage at different merge ",
    "counts.  At 0 merges every box is a single character; at the ",
    "maximum, common short words collapse to a single token each."
  }],
  input["Manipulate[
  LLMAtoZ`Tokeniser`colouredTokens[
    \"To be or not to be, that is the question.\",
    Take[mergeRules, n]],
  {{n, 500, \"merge rules used\"}, 0, Length[mergeRules], 50,
   Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  para[{
    "Watch BPE assemble \"speaking\" out of characters one merge at a ",
    "time:"
  }],
  input["With[{prog = LLMAtoZ`Tokeniser`mergeProgression[
    \"speaking\", Take[mergeRules, 500]]},
  Animate[
    Framed[
      Style[Row[Riffle[prog[[step]], \"  \"]], 18, FontFamily -> \"Courier\"],
      FrameMargins -> 12],
    {{step, 1, \"merges applied\"}, 1, Length[prog], 1,
     Appearance -> \"Labeled\"},
    AnimationRate -> 1.5, ControlPlacement -> Top]
]"],

  section["1.4  The vocab-sequence trade-off"],

  para[{
    "Train BPE with progressively more merges and the sequence length ",
    "drops as the vocabulary grows.  The shape of the curve is the ",
    "empirical justification for sub-word tokenisation.  The data ",
    "below was computed once and cached in ",
    StyleBox["part-01-tokeniser-and-embeddings/data/vocab_seqlen_sweep.json",
             FontFamily -> "Courier"],
    "; we add the character and word-level extremes from the loaded ",
    "corpus and plot them on a log\[Dash]log scale."
  }],
  input["sweep = Import[
  \"part-01-tokeniser-and-embeddings/data/vocab_seqlen_sweep.json\",
  \"RawJSON\"];
bpePoints  = {#[\"vocab\"], #[\"seqLen\"]} & /@ sweep;
charPoint  = {65, StringLength[text]};
wordPoint  = {Length[Union[StringSplit[text]]], Length[StringSplit[text]]};
ListLogLogPlot[
  {Prepend[bpePoints, charPoint], {wordPoint}},
  Frame -> True, GridLines -> Automatic,
  FrameLabel -> {\"vocabulary size |V|\",
                  \"sequence length n (tokens)\"},
  PlotLabel -> \"Tiny Shakespeare: vocab\\[Dash]seqlen trade-off\",
  PlotStyle -> {
    Directive[ColorData[97][1], PointSize[0.018]],
    Directive[ColorData[97][2], PointSize[0.025]]},
  Joined -> {True, False},
  PlotLegends -> Placed[
    {\"char  +  BPE sweep (0\\[Dash]4000 merges)\", \"word-level\"},
    {Right, Bottom}],
  ImageSize -> 850]"],
  para[{
    StyleBox["Figure 1.3.", FontWeight -> Bold],
    "  Character (top-left corner: ", mathV, " = 65, ",
    math["n"], " \[TildeEqual] 1.16 M), word (bottom-right corner: ",
    mathV, " \[TildeEqual] 14.5 k, ", math["n"], " \[TildeEqual] 252 k), ",
    "and BPE (the curve between, one point per merge budget).  ",
    "The interior of the curve is where every modern tokeniser lives."
  }],
  para[{
    "Numerical picture: at 50 merges the sequence is already 769k ",
    "tokens (a third shorter than character) for the price of 50 extra ",
    "vocabulary entries.  At 4000 merges, 299k tokens, six times shorter ",
    "than character.  The curve flattens because later merges chase ",
    "increasingly rare bigrams.  Practical models pick an interior ",
    "point: GPT-2 uses 50,257 merges, GPT-4 uses 100,256."
  }],

  para[{
    "The information-theoretic shape of this curve is not accidental.  ",
    "Tiny Shakespeare is approximately Zipfian (the ", math["n"],
    "-th most frequent word has about 1/", math["n"],
    " the frequency of the most frequent), so the marginal compression ",
    "from the ", math["n"], "-th merge falls as 1/", math["n"],
    ".  Total compression grows as ",
    math[RowBox[{"log", "(", "n", ")"}]],
    " \[LongDash] the gentle downward drift toward the word-level ",
    "corner."
  }],

  section["1.5  Character-level as the small-model alternative"],

  para[{
    "BPE is the right choice for any model with a parameter budget in ",
    "the tens or hundreds of millions.  For a small model (say 1\[Dash]5 M ",
    "parameters) trained on a Shakespeare-sized corpus, the embedding ",
    "tables \[LongDash] one row per vocabulary entry \[LongDash] consume ",
    "most of the parameter budget, leaving very little for the ",
    "transformer to actually compute with."
  }],
  para[{
    "A character-level vocabulary shrinks the embedding tables by a ",
    "factor of ~60.  The trade-off is much longer sequences, but at ",
    "this scale the per-parameter modelling capacity is what matters.  ",
    "This is the choice Karpathy made in ",
    StyleBox["char-rnn", FontSlant -> Italic],
    " (2015) at LSTM ~1M parameters on the same corpus, and the choice ",
    "we will make for our from-scratch trained model later in this ",
    "notebook."
  }],
  input["charVocab = LLMAtoZ`CharTokeniser`charVocabulary[text];
charTokId = LLMAtoZ`CharTokeniser`charTokenToIdAssociation[charVocab];
charIdTok = LLMAtoZ`CharTokeniser`charIdToTokenAssociation[charVocab];
{Length[charVocab], First[charVocab[[1]]] // Quiet}"],

  (* ============ Part 2: Embeddings ============ *)

  chapter["2.  Embeddings: integers to vectors"],

  para[{
    "A transformer does not see token IDs; it sees vectors.  Once we ",
    "have a list of integer token IDs, we feed them through two ",
    "embedding tables in parallel: a token embedding (one row per ",
    "vocabulary entry, learnable) and a positional embedding (one row ",
    "per position in the sequence, learnable).  The model input is the ",
    "elementwise sum."
  }],
  figureCell["part-01-tokeniser-and-embeddings/figures/illustration_pos_encoding.png",
    "Figure 2.1.  Learned positional embeddings: each position id (1, 2, ..., n) is looked up in a learned table; the resulting vector is added to the token embedding at that position."],

  section["2.1  Why a sum, not a concatenation?"],

  para[{
    "Two reasons.  First, with the dimensions kept aligned the ",
    "transformer needs no extra parameters to combine them; ",
    "concatenation would double the input dimension and require the ",
    "next layer to be wider.  Second, the sum is what falls out ",
    "naturally if you ask \[OpenCurlyDoubleQuote]what is the right ",
    "basis-free way to combine two pieces of information that live in ",
    "the same vector space\[CloseCurlyDoubleQuote] \[LongDash] either ",
    "is just a vector, and their sum is a vector.  The model is free ",
    "to learn token- and position-information directions that are ",
    "nearly orthogonal, which is what trained transformers tend to do ",
    "in practice."
  }],

  section["2.2  The embedder NetGraph"],

  input["embedder = LLMAtoZ`Tokeniser`tokenPositionEmbedder[
  Length[vocab], 64, 256, RandomSeeding -> 42];
embedder"],

  para[{
    "Three nodes: token embedding, position embedding, and an ",
    "elementwise add.  Inputs are a token-ID sequence and a position-ID ",
    "sequence; output is a ",
    math[RowBox[{"(", "seqLen", ",", " ", SubscriptBox["d", "model"], ")"}]],
    " tensor, the standard input shape of the transformer block we ",
    "build in Chapter 4."
  }],

  input["ids = LLMAtoZ`Tokeniser`encodeWithIds[
  \"To be or not to be, that is the question.\",
  mergeRules, tokId];
LLMAtoZ`Tokeniser`embedTokens[embedder, ids] // Dimensions"],
  para[{
    "Twelve tokens in, a 12\[Times]64 tensor out.  That tensor is ",
    "exactly what the first transformer block will consume."
  }],

  section["2.3  Geometry before training"],

  para[{
    "The token-embedding row for ",
    StyleBox["\"the\"", FontFamily -> "Courier"],
    " right now is just a vector of 64 Gaussian noise samples.  ",
    "Whatever similarity structure the trained model will eventually ",
    "have \[LongDash] words near their inflections, punctuation off in ",
    "its own corner \[LongDash] is not in there yet.  It is useful to ",
    "see the \[OpenCurlyDoubleQuote]before\[CloseCurlyDoubleQuote] ",
    "picture so we have a baseline to revisit later."
  }],
  figureCell["part-01-tokeniser-and-embeddings/figures/illustration_embedding_compare.png",
    "Figure 2.2.  Conceptual: random initialisation versus after-training.  At random init, labelled tokens are scattered through a Gaussian cloud with no structure.  After training, words of similar role cluster: articles with articles, pronouns with pronouns, punctuation off by itself."],
  figureCell["part-01-tokeniser-and-embeddings/figures/embedding_geometry_random.png",
    "Figure 2.3.  Empirical: cosine similarity heatmap of eight high-frequency tokens at random init.  Diagonal = 1 (each token is identical to itself); off-diagonal entries are uniform noise centred near zero with standard deviation ~1/\[Sqrt]64 = 0.125, exactly the theoretical prediction for unit Gaussian vectors in 64 dimensions."],

  (* ============ Part 3: Attention ============ *)

  chapter["3.  Attention: the operation the architecture is named for"],

  figureCell["part-02-self-attention/figures/illustration_attention_hero.png",
    "Figure 3.1.  The QKV picture.  Each input embedding is projected by three matrices into a query, a key, and a value.  The dot product of every query with every key gives the attention-score matrix.  The upper triangle is masked (causal).  Row-wise softmax produces the attention weights.  The weighted sum of the values is the output."],

  para[{
    "Strip the fancy parts away.  Attention is the operation ",
    "\[OpenCurlyDoubleQuote]each output token is a weighted sum of the ",
    "input tokens, where the weights are learned from the input itself.",
    "\[CloseCurlyDoubleQuote]  Three lines:"
  }],
  para[{
    "\[FilledSmallCircle] Project each input vector three times by ",
    "three learned linear maps: a ",
    StyleBox["query ", FontSlant -> Italic],
    math["q"], ", a ",
    StyleBox["key ", FontSlant -> Italic],
    math["k"], ", and a ",
    StyleBox["value ", FontSlant -> Italic],
    math["v"], ".\n",
    "\[FilledSmallCircle] For each position ", math["i"],
    ", compute the dot product of ", math[SubscriptBox["q", "i"]],
    " with every ", math[SubscriptBox["k", "j"]],
    " in the sequence.  Softmax these scores along ", math["j"],
    " to get attention weights summing to 1.\n",
    "\[FilledSmallCircle] The output at position ", math["i"],
    " is the weighted sum of ", math[SubscriptBox["v", "j"]],
    " using those weights."
  }],
  displayMath[
    RowBox[{"Attention", "(", "Q", ",", " ", "K", ",", " ", "V", ")",
      " = ", "softmax", "(",
      FractionBox[RowBox[{"Q", "\[CenterDot]", SuperscriptBox["K", "\[Transpose]"]}],
                  SqrtBox[SubscriptBox["d", "k"]]],
      ")", "\[CenterDot]", "V"}]
  ],

  section["3.1  Why divide by Sqrt[d_k]?"],

  para[{
    "Most expositions assert the ", mathSqrtDk,
    " scaling factor.  We will derive it.  This derivation, along with ",
    "a few similar first-principles arguments later in this notebook, ",
    "is what separates a teaching post from a code dump."
  }],
  para[{
    "Suppose ",
    math[RowBox[{"q", ",", " ", "k", " ", "\[Element]", " ",
                 SuperscriptBox["\[DoubleStruckCapitalR]",
                                SubscriptBox["d", "k"]]}]],
    " with each component independently drawn from a zero-mean ",
    "unit-variance distribution.  The dot product is a sum of ",
    mathDk, " random terms ",
    math[RowBox[{SubscriptBox["q", "i"], SubscriptBox["k", "i"]}]],
    ".  Each term has mean 0 (since ",
    math[RowBox[{"E", "[", SubscriptBox["q", "i"], SubscriptBox["k", "i"], "]"}]],
    " = ", math[RowBox[{"E", "[", SubscriptBox["q", "i"], "]", "\[CenterDot]",
                       "E", "[", SubscriptBox["k", "i"], "]"}]],
    " = 0) and variance ",
    math[RowBox[{"E", "[", SuperscriptBox[SubscriptBox["q", "i"], "2"], "]",
        "\[CenterDot]",
        "E", "[", SuperscriptBox[SubscriptBox["k", "i"], "2"], "]"}]],
    " = 1.  By independence across ", math["i"],
    ", the variance of the sum is the sum of the variances: ",
    math[RowBox[{"Var", "(", "q", "\[CenterDot]", "k", ")", " = ",
                 SubscriptBox["d", "k"]}]],
    ".  Standard deviation grows as ",
    math[SqrtBox[SubscriptBox["d", "k"]]], "."
  }],
  para[{
    "Verify empirically.  Sample 2000 Gaussian (q, k) pairs at three ",
    "values of ", mathDk, " and look at the dot-product distribution:"
  }],
  input["SeedRandom[1]; nSamples = 2000;
dks = {16, 64, 256};
dots = Association @ Table[
  d -> Table[
    With[{q = RandomReal[NormalDistribution[], d],
          k = RandomReal[NormalDistribution[], d]},
      q . k], {nSamples}], {d, dks}];
StandardDeviation /@ dots"],
  para[{
    "The empirical standard deviations come out to ~4, 8, 16 \
\[LongDash] exactly the predicted ",
    math[SqrtBox[SubscriptBox["d", "k"]]],
    " of 4, 8, 16."
  }],
  figureCell["part-02-self-attention/figures/qkdotproduct_variance.png",
    "Figure 3.2.  Empirical distribution of q\[CenterDot]k for three values of d_k.  Without scaling (left), the spread grows like \[Sqrt](d_k).  After dividing by \[Sqrt](d_k) (right), all three distributions collapse onto a unit-variance shape."],
  para[{
    "The cost of skipping the divisor: softmax of a high-variance ",
    "score row saturates onto a single entry, the gradient through ",
    "softmax vanishes, training stalls.  Demo:"
  }],
  figureCell["part-02-self-attention/figures/softmax_saturation.png",
    "Figure 3.3.  Left: without scaling, softmax(Q\[CenterDot]K\[Transpose]) at d_k=256 is essentially a row of one-hots.  Right: with the 1/\[Sqrt](d_k) scaling, the softmax distributes mass meaningfully."],

  subsection["3.1.1  Interactive: the d_k dial"],
  input["Manipulate[
  SeedRandom[1];
  Module[{nSamples = 600, nRow = 16, dots, row, scale, sm},
    dots = Table[
      With[{q = RandomReal[NormalDistribution[], dk],
            k = RandomReal[NormalDistribution[], dk]},
        q . k], {nSamples}];
    scale = If[useScaling, Sqrt[N[dk]], 1.0];
    With[{q = RandomReal[NormalDistribution[], dk],
          K = RandomReal[NormalDistribution[], {nRow, dk}]},
      row = (K . q) / scale];
    sm = Exp[row - Max[row]]; sm = sm/Total[sm];
    GraphicsRow[{
      Histogram[dots/scale, 30, \"PDF\",
        PlotRange -> {{-6, 6}, {0, 0.8}}, Frame -> True,
        FrameLabel -> {If[useScaling, \"q.k / \[Sqrt]d_k\", \"q.k\"], \"PDF\"},
        PlotLabel -> Row[{\"std = \",
          NumberForm[StandardDeviation[dots/scale], 3]}],
        ImageSize -> 440],
      BarChart[sm, Frame -> True,
        PlotLabel -> Row[{\"softmax row (\",
          If[useScaling, \"scaled\", \"unscaled\"], \")\"}],
        FrameLabel -> {\"key position\", \"attention weight\"},
        PlotRange -> {Automatic, {0, 1}}, ImageSize -> 440]
    }, ImageSize -> 900]
  ],
  {{dk, 64, \"d_k\"}, 4, 1024, 4, Appearance -> \"Labeled\"},
  {{useScaling, True, \"divide by \[Sqrt]d_k\"}, {True, False}},
  ControlPlacement -> Top]"],

  section["3.2  Causal masking"],

  para[{
    "A language model is allowed to attend to past tokens, never to ",
    "future ones (otherwise it cheats during training).  Implementation: ",
    "add a strict-upper-triangular matrix of ",
    math["-\[Infinity]"],
    " to the pre-softmax score matrix; the row-wise softmax sends those ",
    "entries to exactly zero.  In WL, with a large negative number as ",
    "a finite stand-in (the literal infinity propagates NaN through ",
    "the dot product earlier):"
  }],
  input["LLMAtoZ`Attention`causalMask[6] // MatrixForm"],
  figureCell["part-02-self-attention/figures/causal_mask_visualisation.png",
    "Figure 3.4.  The mask, the masked-scores tensor, and the softmax weights.  The upper triangle is exactly zero after softmax; the lower triangle + diagonal carries the actual attention weights."],

  section["3.3  Single-head attention as a NetGraph"],

  para[{
    "Six nodes: three ",
    StyleBox["LinearLayer", FontFamily -> "Courier"],
    " projections for Q, K, V; a ",
    StyleBox["FunctionLayer", FontFamily -> "Courier"],
    " that computes ", math[FractionBox[RowBox[{"Q", "\[CenterDot]",
        SuperscriptBox["K", "\[Transpose]"]}],
        SqrtBox[SubscriptBox["d", "head"]]]],
    " and adds the causal mask; a ",
    StyleBox["SoftmaxLayer", FontFamily -> "Courier"],
    "; a final ",
    StyleBox["FunctionLayer", FontFamily -> "Courier"],
    " for the value gather.  Six lines."
  }],
  input["head = LLMAtoZ`Attention`singleHeadCausalAttention[64, 32, 16];
head"],
  para[{
    "Drop the embedder's output into it, get a same-shape output:"
  }],
  input["SeedRandom[42];
x = RandomReal[NormalDistribution[0, 0.1], {16, 64}];
out = head[x];
Dimensions[out]"],

  section["3.4  Multi-head attention"],

  para[{
    "A single head can only attend in one way at a time.  Multi-head ",
    "attention runs ", mathNHeads, " parallel single-head computations: ",
    "each head applies its own learned projections of the full input ",
    "down to dimension ", mathDhead,
    ", computes attention independently, and produces a ", mathDhead,
    "-dimensional output.  The heads are concatenated and a final ",
    "linear projection mixes them back into a ", mathDmodel, "-vector."
  }],
  para[{
    "The architectural trick is to split ", mathDmodel, " into ",
    mathNHeads,
    " equal pieces so the multi-head version has the same parameter ",
    "count as a single-head version at ", mathDmodel, "."
  }],
  input["mha = LLMAtoZ`Attention`multiHeadAttentionBlock[64, 4, 16];
mha[x] // Dimensions"],

  section["3.5  What real attention looks like: GPT-2 small"],

  para[{
    "We can do something the typical from-scratch tutorial cannot: load ",
    "a fully pretrained language model from ",
    StyleBox["NetModel", FontFamily -> "Courier"],
    " and look at its actual attention patterns.  GPT-2 small is a ",
    "12-layer transformer with 12 heads per layer at ", mathDmodel,
    " = 768.  Each block's attention sub-layer holds four learned ",
    "matrices: ", math[SubscriptBox["W", "Q"]], ", ",
    math[SubscriptBox["W", "K"]], ", ",
    math[SubscriptBox["W", "V"]],
    " (768\[Times]768 each) plus an output projection.  We have the ",
    "weights; we can recompute the attention weight matrix by hand."
  }],
  figureCell["part-02-self-attention/figures/gpt2_heads_gallery.png",
    "Figure 3.5.  All 12 heads of layer 5 of GPT-2 small on the prompt \"The quick brown fox jumps over the lazy dog\".  Each row of each heatmap sums to one."],
  para[{
    "Two features jump out.  First, every head has a heavily-shaded ",
    "leftmost column \[LongDash] the ",
    StyleBox["attention sink", FontSlant -> Italic],
    " (Xiao et al. 2024): heads dump probability mass onto the first ",
    "token as a \[OpenCurlyDoubleQuote]no-op\
\[CloseCurlyDoubleQuote] vote when they have nothing useful to do.  ",
    "Second, the heads are not all doing the same thing:"
  }],
  figureCell["part-02-self-attention/figures/gpt2_head_specialisation.png",
    "Figure 3.6.  Three hand-picked heads of layer 5.  Left: previous-token tracker.  Middle: self-attention diagonal.  Right: approximately uniform.  None of these specialisations is enforced architecturally; they are what twelve heads turn into when trained against billions of tokens of text."],

  subsection["3.5.1  Interactive: layer/head scrubber"],
  para[{
    "Pre-extracted attention tensors for every layer of GPT-2 small on ",
    "the same prompt are cached in ",
    StyleBox["part-02-self-attention/data/gpt2_attention_all_layers.wxf",
            FontFamily -> "Courier"], "."
  }],
  input["attCache = Import[\"part-02-self-attention/data/gpt2_attention_all_layers.wxf\"];
Manipulate[
  With[{w = attCache[\"weights\"][layer][[head]],
        n = Length[attCache[\"ids\"]]},
    MatrixPlot[w, ColorFunction -> \"ThermometerColors\",
      ColorFunctionScaling -> True,
      FrameTicks -> {{Range[n], None}, {Range[n], None}},
      PlotLabel -> Row[{\"GPT-2 small  \[Bullet]  layer \", layer,
        \"  \[Bullet]  head \", head}],
      PlotLegends -> Automatic, ImageSize -> 640]],
  {{layer, 5, \"layer\"}, 1, 12, 1, Appearance -> \"Labeled\"},
  {{head, 1, \"head\"}, 1, 12, 1, Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  (* ============ Part 4: Transformer block ============ *)

  chapter["4.  The transformer block + parameter accounting"],

  figureCell["part-03-transformer-block/figures/illustration_block_diagram.png",
    "Figure 4.1.  One pre-norm transformer block: LayerNorm \[RightArrow] attention \[RightArrow] residual \[RightArrow] LayerNorm \[RightArrow] feed-forward \[RightArrow] residual.  Two sublayers, each in its own additive residual.  The residual stream is never normalised; only the input to each sublayer is."],

  section["4.1  Position-wise feed-forward"],
  para[{
    "Attention lets every position look at every other position, but ",
    "within each position all it does is take a weighted average of ",
    "value vectors.  There is no per-position transformation that ",
    "mixes the features of a single token nonlinearly \[LongDash] no ",
    "place for the model to actually ",
    StyleBox["compute", FontSlant -> Italic],
    " on what it has gathered.  The feed-forward block is the model's ",
    "place to compute: a two-layer position-wise MLP with a non-",
    "linearity in the middle."
  }],
  para[{
    "Conventionally ", mathDFF, " = 4 \[Times] ", mathDmodel,
    " and the non-linearity is GELU.  This block is where most of the ",
    "transformer's parameter budget lives."
  }],
  input["ffn = LLMAtoZ`Transformer`feedForwardBlock[64, 256, 16];
SeedRandom[1]; xx = RandomReal[NormalDistribution[0, 0.1], {16, 64}];
Dimensions[ffn[xx]]"],

  section["4.2  Pre-norm vs post-norm"],
  para[{
    "Two sub-layers, each in its own residual connection:"
  }],
  displayMath[RowBox[{
    "x'", " = ", "x", " + ", "Attention(LayerNorm(x))"
  }]],
  displayMath[RowBox[{
    "y", " = ", "x'", " + ", "FFN(LayerNorm(x'))"
  }]],
  para[{
    "Two design choices visible here.  First, the LayerNorm sits ",
    StyleBox["before", FontSlant -> Italic],
    " each sub-layer, not after.  This is the ",
    StyleBox["pre-norm ", FontWeight -> Bold],
    "ordering, which differs from the post-norm ordering of the ",
    "original Vaswani 2017 paper.  Second, each sub-layer is wrapped ",
    "in an additive residual."
  }],
  para[{
    "Why pre-norm?  In a deep stack what matters is whether the ",
    "residual stream is ever destroyed.  Pre-norm normalises the ",
    StyleBox["input", FontSlant -> Italic],
    " to each sub-layer, then adds the sub-layer's output to the ",
    "unnormalised residual.  The residual stream is never normalised, ",
    "so a gradient flowing backward through ", mathNLayers,
    " layers is the product of (1 + small_correction) factors \
\[LongDash] a well-behaved O(1) number.  Post-norm normalises ",
    StyleBox["after", FontSlant -> Italic],
    " the residual addition, so each layer rescales the residual ",
    "stream by 1/std, and the backward gradient is a product of ",
    "repeatedly-rescaled numbers, which can compress to zero quickly."
  }],
  figureCell["part-03-transformer-block/figures/pre_vs_post_norm_gradients.png",
    "Figure 4.2.  Forward-pass activation-norm growth on a randomly-initialised stack.  Pre-norm grows linearly with depth; post-norm flattens (the LayerNorm at the end of every block pins the magnitude to 1).  The reverse pattern in backprop is what makes post-norm hard to train at depth without warmup tricks.  GPT-2/3, LLaMA, Mistral, nanochat all use pre-norm."],
  input["block = LLMAtoZ`Transformer`transformerBlock[64, 4, 256, 16];
{block[xx] // Dimensions, Information[block, \"ArraysTotalElementCount\"]}"],

  section["4.3  Stacking + the full GPT model"],
  para[{
    "Stack ", mathNLayers,
    " of these blocks, prepend an embedder, append a final LayerNorm ",
    "and an output projection of shape (", mathDmodel, ", ", mathV,
    "), and you have a complete GPT-style language model."
  }],
  input["model = LLMAtoZ`Transformer`gptModel[256, 8, 6, 1024, 1064, 256];
demoIds = RandomInteger[{1, 1064}, 256];
Dimensions[model[demoIds]]"],
  para[{
    "(seqLen, |V|) out, as advertised.  Every row of the output is the ",
    "pre-softmax logit vector at one position.  During training we shift ",
    "by one and ask the model to predict the next token.  During ",
    "inference we apply softmax to the last row and sample."
  }],

  section["4.4  Counting parameters"],
  para[{
    "A pre-norm transformer block of size (", mathDmodel, ", ",
    mathDFF, ", ", mathNHeads, ") has, in closed form:"
  }],
  para[{
    "\[FilledSmallCircle] Q, K, V projections summed over heads: ",
    math[RowBox[{"3", SuperscriptBox[SubscriptBox["d", "model"], "2"]}]], ".\n",
    "\[FilledSmallCircle] Output projection: ",
    math[SuperscriptBox[SubscriptBox["d", "model"], "2"]], ".\n",
    "\[FilledSmallCircle] Two LayerNorms (scale + bias each): ",
    math[RowBox[{"4", SubscriptBox["d", "model"]}]], ".\n",
    "\[FilledSmallCircle] FFN linears (with biases): ",
    math[RowBox[{"2", SubscriptBox["d", "model"], SubscriptBox["d", "ff"],
                 " + ", SubscriptBox["d", "ff"], " + ", SubscriptBox["d", "model"]}]], "."
  }],
  displayMath[RowBox[{
    SubscriptBox["params", "block"], " = ",
    "4", SuperscriptBox[SubscriptBox["d", "model"], "2"], " + ",
    "2", SubscriptBox["d", "model"], SubscriptBox["d", "ff"], " + ",
    SubscriptBox["d", "ff"], " + ", "5", SubscriptBox["d", "model"]
  }]],
  para[{
    "On top of the blocks: embeddings (",
    math[RowBox[{SubscriptBox["d", "model"], "(",
                  "|V|", " + ", SubscriptBox["n", "max"], ")"}]],
    "), a final LayerNorm (", math[RowBox[{"2", SubscriptBox["d", "model"]}]],
    "), and an output projection (", math[RowBox[{SubscriptBox["d", "model"], "|V|"}]],
    "; can be tied to the input embedding to save these parameters).  ",
    StyleBox["parameterCount[spec]", FontFamily -> "Courier"],
    " computes the closed form; ",
    StyleBox["parameterCountFull[spec]", FontFamily -> "Courier"],
    " adds the frozen causal-mask arrays we bake into each attention ",
    "head (one ", mathNHeads,
    " \[Times] (maxSeqLen)\[Times](maxSeqLen) tensor per layer)."
  }],
  figureCell["part-03-transformer-block/figures/params_breakdown.png",
    "Figure 4.3.  Where the parameters go in our reference 5M-parameter model.  Most of the trainable parameters (37%) live in the FFN; the frozen causal masks contribute another 37% of the array count even though they are not trainable.  Attention QKV projections are 14%.  Embedding tables and LayerNorms together are < 5%."],

  section["4.5  Reconstructing GPT-2"],
  para[{
    "A natural test of our ", StyleBox["gptModel", FontFamily -> "Courier"],
    " constructor: can it build GPT-2 small?  Plug in the architectural ",
    "spec and the parameter formula should give ~124M with weight ",
    "tying enabled (sharing the input embedding with the output ",
    "projection)."
  }],
  input["gpt2Spec = LLMAtoZ`Transformer`gpt2SmallSpec[];
{LLMAtoZ`Transformer`parameterCount[gpt2Spec],
 LLMAtoZ`Transformer`parameterCount[gpt2Spec, \"WeightTying\" -> True]}"],
  para[{
    "Without weight tying: 163M.  With weight tying: 124,453,201 \
\[LongDash] matching real GPT-2 small (124,439,808) to within 13,000 ",
    "parameters, a < 0.01% discrepancy from minor LayerNorm-bias ",
    "conventions.  The architecture itself is the same."
  }],
  para[{
    "Two equivalent ways forward exist for the rest of the notebook:"
  }],
  para[{
    "\[FilledCircle] ",
    StyleBox["Build from scratch:", FontWeight -> Bold], " ",
    StyleBox["buildGpt2Architecture[\"Small\"]", FontFamily -> "Courier"],
    " constructs an empty 124M net at GPT-2's spec; we train it from ",
    "random init.  Infeasible on Mac mini CPU.\n",
    "\[FilledCircle] ",
    StyleBox["Load the pretrained NetModel:", FontWeight -> Bold], " ",
    StyleBox["loadPrebuiltGpt2[]", FontFamily -> "Courier"],
    " returns OpenAI's trained checkpoint (498 MB on first download).  ",
    "We skip pretraining and fine-tune from real weights.  This is what ",
    "Path A in Chapter 7 below uses."
  }],

  (* ============ Part 5: Training ============ *)

  chapter["5.  Pre-training: actually training a model"],

  para[{
    "Up to here, every model we have built has been randomly ",
    "initialised.  Time to train one.  The training task is next-token ",
    "prediction: each (input, target) pair is a context window and the ",
    "same window shifted one step forward.  Cross-entropy loss against ",
    "the shifted targets is what ",
    StyleBox["NetTrain", FontFamily -> "Courier"], " optimises."
  }],

  section["5.1  The training loop, in one call"],
  para[{
    "After three chapters of careful construction, the training loop is ",
    "wrapped in one function:"
  }],
  input["(* SKIP: heavy training; the trained checkpoints already exist on disk *)
(* tinyResult = LLMAtoZ`Training`trainGptModel[
     <|\"label\" -> \"tiny\", \"dModel\" -> 96, \"nHeads\" -> 4,
       \"nLayers\" -> 3, \"dFF\" -> 256,
       \"vocabSize\" -> Length[vocab], \"maxSeqLen\" -> 64|>,
     ids, <|\"BaseLR\" -> 0.002, \"MaxRounds\" -> 3|>]; *)
\"checkpoint at part-04-pretraining/data/tiny_gpt_trained.wlnet\""],
  para[{
    "Internally ", StyleBox["trainGptModel", FontFamily -> "Courier"],
    " does what every transformer training loop does:"
  }],
  para[{
    "1. Slide a length-", math["L"],
    " window through the token stream to produce (input, target) pairs.\n",
    "2. Build the model at the given spec via ",
    StyleBox["gptModel", FontFamily -> "Courier"], ".\n",
    "3. Wrap with a final softmax (see Section 5.3 for why).\n",
    "4. Hand it all to ", StyleBox["NetTrain", FontFamily -> "Courier"],
    " with ", StyleBox["CrossEntropyLossLayer", FontFamily -> "Courier"],
    ", ADAM, a batch size, a learning-rate schedule, and a ",
    StyleBox["MaxTrainingRounds", FontFamily -> "Courier"], " cap.\n",
    "5. Return the trained net plus its per-round loss history."
  }],

  section["5.2  The cosine LR schedule"],
  para[{
    "The first ~10% of training linearly ramps the learning rate up to ",
    "its peak; the remaining 90% cosine-decays it to 10% of peak.  ",
    "Warmup avoids first-batch instability; cosine decay avoids late-",
    "training noise when the loss landscape flattens."
  }],
  input["Plot[LLMAtoZ`Training`cosineSchedule[r, 100, 0.001, 0.1],
  {r, 0, 100}, Frame -> True,
  FrameLabel -> {\"round (% of total)\", \"learning rate\"},
  PlotLabel -> \"Linear warmup (10%) + cosine decay\",
  PlotStyle -> Directive[ColorData[97][2], Thick], ImageSize -> 640]"],

  section["5.3  A bug worth sharing: the missing softmax"],

  figureCell["figures/illustration_xent_curves.png",
    "Figure 5.1.  The signature of the bug, side-by-side.  Left: cross-entropy without a softmax wrap descends past zero into unbounded-negative territory \[LongDash] not training, just driving one logit to +infinity to game an unbounded loss.  Right: with the softmax wrap, proper non-negative cross-entropy descends from log(|V|) to a positive asymptote."],

  para[{
    "This section earns its place because it captures a class of silent ",
    "error that the series has the right tooling to surface, and ",
    "because it tripped us up during development.  The naive way to ",
    "define the loss is"
  }],
  input["(* DON'T do this with raw logits *)
(*  NetTrain[..., LossFunction -> CrossEntropyLossLayer[\"Index\"]]  *)"],
  para[{
    StyleBox["CrossEntropyLossLayer[\"Index\"]", FontFamily -> "Courier"],
    " expects ",
    StyleBox["probabilities", FontSlant -> Italic],
    " as input and computes ",
    math[RowBox[{"-log(p[target])"}]],
    " \[LongDash] non-negative by construction.  Our ",
    StyleBox["gptModel", FontFamily -> "Courier"],
    " emits raw logits.  Feed logits straight into ",
    StyleBox["CrossEntropyLossLayer[\"Index\"]", FontFamily -> "Courier"],
    " and the \[OpenCurlyDoubleQuote]loss\
\[CloseCurlyDoubleQuote] becomes ",
    math[RowBox[{"-(logit[target])", "+", "constant"}]],
    ", which is unbounded below."
  }],
  para[{
    "For BPE vocab ~4000 this happened to be okay: the optimiser ",
    "couldn't drive a single logit unbounded without similarly moving ",
    "all the others through the shared linear projection, and the ",
    "training landed in regimes that produced sensible token ",
    "distributions.  For a 65-character vocabulary, no such constraint ",
    "applied: the first overnight char-level run, after 6.5 hours, ",
    "produced a model whose maximum logit was ",
    StyleBox["+9,415", FontWeight -> Bold],
    " and whose generations were the single character ",
    StyleBox["'s'", FontFamily -> "Courier"], " repeated to fill the buffer."
  }],
  figureCell["part-04-pretraining/figures/charlevel_loss_curve_BROKEN.png",
    "Figure 5.2.  The broken loss curve, descending smoothly past zero into -10.  Preserved as a teaching artefact."],
  para[{
    "A cross-entropy loss curve that descends ",
    StyleBox["below zero", FontSlant -> Italic],
    " is the signature.  Cross-entropy is bounded below by zero.  A ",
    "loss going negative is not \[OpenCurlyDoubleQuote]training is ",
    "going great\[CloseCurlyDoubleQuote]; it is \[OpenCurlyDoubleQuote]",
    "you are not optimising what you think you are.\[CloseCurlyDoubleQuote]"
  }],
  para[{
    "The fix is one line: wrap the model in ",
    StyleBox["NetMapOperator[SoftmaxLayer[]]", FontFamily -> "Courier"],
    " before training, strip after.  ",
    StyleBox["trainGptModel", FontFamily -> "Courier"],
    " in the package now does this automatically."
  }],

  section["5.4  Watching the model dream up Shakespeare"],
  para[{
    "We ran the corrected training for 5.5 hours on a Mac mini.  A ",
    "background watchdog snapshot generations every time a new round ",
    "completed.  Below: every round of training, the loss, the ",
    "perplexity, and 200-character generations from two prompts at ",
    "temperature 0.8.  Watch the output evolve from noise to ",
    "Shakespeare-flavoured prose."
  }],

  Sequence @@ Flatten[progressEntry /@ progressRounds, 1],

  section["5.5  Loss curve and overfitting"],
  figureCell["part-04-pretraining/figures/charlevel_loss_curve.png",
    "Figure 5.3.  Full training loss curve for the 1.8M-parameter char-level model on Tiny Shakespeare, 23 rounds, 5.5 hours on Mac mini CPU.  Fast initial drop (the model learns the character-frequency distribution), a slower elbow as it discovers word boundaries, and a long shallow tail."],
  figureCell["figures/illustration_overfit.png",
    "Figure 5.4.  Conceptually: the train/eval gap.  Train loss continues to descend; eval loss plateaus.  The growing gap is memorisation, not learning."],
  para[{
    "After 23 rounds, train loss was 0.74.  Held-out eval loss was 2.59, ",
    "for a train/eval ratio of 3.49 \[LongDash] a clear overfit ",
    "signature.  With 1.8M parameters memorising a 1.08M-character ",
    "corpus, the model has more capacity than the data needs and ",
    "gradient descent preferentially memorises.  Further training would ",
    "push train loss down without making the model actually generalise ",
    "better.  The checkpoint we ship is the one at this point."
  }],

  (* ============ Part 6: Sampling ============ *)

  chapter["6.  Sampling: turning logits into text"],

  para[{
    "The trained model is a function from a length-", math["L"],
    " context window to a length-", math["|V|"],
    " logit vector at every position.  Sampling is the policy that ",
    "turns those logits into a chosen next token.  Five strategies plus ",
    "speculative decoding, all from scratch."
  }],

  section["6.1  Temperature"],
  para[{
    "The simplest dial: softmax-with-temperature.  ",
    math[RowBox[{"p", " ", "\[Proportional]", " ",
                 "exp", "(", "logits", "/", "T", ")"}]], ".  ",
    math["T \[Rule] 0"], " is argmax (greedy); ",
    math["T = 1"], " is the model's calibrated distribution; ",
    math["T > 1"], " softens; ",
    math["T \[Rule] \[Infinity]"], " is uniform.  The same 20-element ",
    "logit vector, five different temperatures \[LongDash] watch how ",
    "the probability mass spreads out:"
  }],
  input["SeedRandom[3]; demoLogits = Sort[
  RandomReal[NormalDistribution[0, 1.5], 20], Greater];
softmaxT[logits_, T_] :=
  LLMAtoZ`Sampling`softmaxTemperature[logits, T];
entropyOf[p_] := -Total[p * Log[p + 1.*^-12]];

GraphicsGrid[
  Partition[
    Table[
      BarChart[softmaxT[demoLogits, T],
        Frame -> True,
        FrameLabel -> {\"token id (sorted by logit)\", \"p\"},
        PlotLabel -> Row[{\"T = \", NumberForm[T, 3],
          \"   entropy = \",
          NumberForm[entropyOf[softmaxT[demoLogits, T]], 3]}],
        PlotRange -> {Automatic, {0, 1}},
        ChartStyle -> ColorData[97][1],
        ImageSize -> 420],
      {T, {0.2, 0.5, 1.0, 2.0, 3.0}}],
    2, 2, {1, 1}, {}],
  ImageSize -> 900, Spacings -> 8]"],
  para[{
    StyleBox["Figure 6.1.", FontWeight -> Bold],
    "  Same logit vector at ", math["T = 0.2, 0.5, 1.0, 2.0, 3.0"], ".  ",
    "Entropy goes from ~0.6 nats (peaky) to ~3 nats (near-uniform); the ",
    "argmax stays the same, but how often we'd sample it changes ",
    "completely."
  }],

  subsection["6.1.1  Interactive: the temperature dial"],
  para[{
    "Drag the slider to retemperature the same distribution and watch ",
    "the entropy update live.  This cell only renders the interactive ",
    "control when the notebook is evaluated in Wolfram Desktop; in the ",
    "PDF/static view above we already showed five fixed snapshots."
  }],
  input["Manipulate[
  BarChart[softmaxT[demoLogits, T],
    Frame -> True, FrameLabel -> {\"token id\", \"probability\"},
    PlotLabel -> Row[{\"T = \", NumberForm[T, 3], \"   entropy = \",
      NumberForm[entropyOf[softmaxT[demoLogits, T]], 3]}],
    PlotRange -> {Automatic, {0, 1}},
    ChartStyle -> ColorData[97][1], ImageSize -> 700],
  {{T, 1.0, \"temperature\"}, 0.05, 3.0, 0.05, Appearance -> \"Labeled\"},
  ControlPlacement -> Top]"],

  section["6.2  Top-k and top-p (nucleus)"],
  para[{
    "Truncation strategies.  ",
    StyleBox["Top-k", FontWeight -> Bold],
    " keeps the ", math["k"],
    " highest-probability tokens before re-normalising.  ",
    StyleBox["Top-p", FontWeight -> Bold],
    " (nucleus, Holtzman et al. 2020) keeps the smallest set whose ",
    "cumulative probability exceeds ", math["p"],
    ".  Top-p adapts the cutoff to the local sharpness of the ",
    "distribution."
  }],
  input["SeedRandom[7]; demoP =
  N @ Normalize[Sort[Exp[RandomReal[NormalDistribution[0, 1.2], 20]],
                      Greater], Total];

drawTopK[k_] := BarChart[
  ReplacePart[demoP,
    Thread[Range[k + 1, Length[demoP]] -> 0.]],
  Frame -> True,
  FrameLabel -> {\"token id (rank order)\", \"p\"},
  PlotLabel -> Row[{\"top-k, k = \", k}],
  PlotRange -> {Automatic, {0, Max[demoP] * 1.05}},
  ChartStyle -> ColorData[97][1], ImageSize -> 420];

drawTopP[p_] := Module[
  {sorted = demoP, cum, cut, kept},
  cum = Accumulate[sorted];
  cut = LengthWhile[cum, # <= p &] + 1;
  kept = ReplacePart[sorted,
    Thread[Range[cut + 1, Length[sorted]] -> 0.]];
  BarChart[kept, Frame -> True,
    FrameLabel -> {\"token id (rank order)\", \"p\"},
    PlotLabel -> Row[{\"top-p, p = \", p,
      \"   (kept \", cut, \" tokens)\"}],
    PlotRange -> {Automatic, {0, Max[demoP] * 1.05}},
    ChartStyle -> ColorData[97][2], ImageSize -> 420]];

GraphicsGrid[{
  {drawTopK[3], drawTopP[0.5]},
  {drawTopK[5], drawTopP[0.8]},
  {drawTopK[10], drawTopP[0.95]}},
  ImageSize -> 900, Spacings -> 8]"],
  para[{
    StyleBox["Figure 6.2.", FontWeight -> Bold],
    "  Same 20-token probability row, three top-k cutoffs vs three ",
    "top-p cutoffs.  Notice that top-p with ", math["p"], " = 0.95 ",
    "automatically expands the kept set when the head is flat and ",
    "contracts it when the head is sharp \[LongDash] something a fixed ",
    math["k"], " can't do."
  }],

  section["6.3  Beam search"],
  para[{
    "For translation-style tasks where one answer is correct, beam ",
    "search is the deterministic alternative.  Track ",
    StyleBox["beamWidth", FontFamily -> "Courier"],
    " most likely partial sequences at every step; at the end return ",
    "the highest-scoring complete one.  Rare in generative LLMs but ",
    "still common in machine translation."
  }],

  section["6.4  Speculative decoding"],
  para[{
    "A small draft model proposes ", math["k"],
    " tokens autoregressively; the main model verifies them in a single ",
    "forward pass; accepted tokens commit, the first rejected token is ",
    "resampled.  Useful when the draft is much cheaper than the main."
  }],
  figureCell["part-05-sampling/figures/speculative_throughput.png",
    "Figure 6.3.  Throughput model: expected speedup vs draft length k for four accept rates, with the draft being 20x cheaper than the main."],

  (* ============ Part 7: SFT ============ *)

  chapter["7.  Supervised fine-tuning"],

  para[{
    "The model after Section 5 can continue Shakespeare but has no ",
    "concept of \[OpenCurlyDoubleQuote]a user asks a question, an ",
    "assistant answers.\[CloseCurlyDoubleQuote] Supervised fine-tuning ",
    "teaches that concept by training the same next-token loss on a ",
    "dataset formatted in a chat template."
  }],

  figureCell["figures/illustration_sft_pipeline.png",
    "Figure 7.1.  Supervised fine-tuning a pretrained model on a small chat-formatted dataset.  Same loss, same optimiser; different data."],

  section["7.1  Path A vs Path B"],
  para[{
    "Two routes meet here.  ",
    StyleBox["Path A: ", FontWeight -> Bold],
    "the canonical move.  Take OpenAI's pretrained GPT-2 small (124M ",
    "parameters), prepend it with our chat template, train for a few ",
    "rounds on a small Q&A dataset.  The result is a model that ",
    "actually speaks English.  ",
    StyleBox["Path B: ", FontWeight -> Bold],
    "the from-scratch demonstration.  Take the 1.8M model we trained ",
    "in Section 5, extend its vocabulary with three special tokens, ",
    "train.  The result is a model that, given enough scale, could ",
    "speak English; at this scale it produces locally plausible ",
    "Shakespeare-flavoured exchanges only."
  }],
  para[{
    "We will execute Path A here.  The Path B mechanics are identical; ",
    "only the starting checkpoint changes."
  }],

  section["7.2  Path A: SFT-ing pretrained GPT-2 in 60 seconds"],
  para[{
    "The toy instruction dataset ships in ",
    StyleBox["SFT.wl", FontFamily -> "Courier"],
    ": 20 hand-written {user, assistant} pairs of Shakespeare-themed ",
    "Q&A.  Just enough to demonstrate SFT end-to-end.  The same code ",
    "consumes SmolTalk-sized datasets when scaled up."
  }],
  input["LLMAtoZ`SFT`toyInstructionDataset[][[1 ;; 4]]"],
  para[{
    "Load the trained checkpoint and exercise it.  (The 498 MB ",
    "GPT-2 file is gitignored; if you have not downloaded it via ",
    "NetModel yet, evaluate ",
    StyleBox["NetModel[\"GPT-2 Transformer Trained on WebText Data\", \"Task\" -> \"LanguageModeling\"]",
            FontFamily -> "Courier"], " once.)"
  }],
  input["(* uncomment if needed:
gpt = NetModel[\"GPT-2 Transformer Trained on WebText Data\",
  \"Task\" -> \"LanguageModeling\"];
*)
(* For now we point at a previously SFT'd checkpoint on disk *)
sftCkpt = \"part-06-supervised-fine-tuning/data/gpt2_sft_chat.wlnet\";
If[FileExistsQ[sftCkpt],
  sftNet = Import[sftCkpt];
  Print[\"loaded SFT'd net, layers: \", Length[sftNet]],
  Print[\"SFT checkpoint not present; run src/run_gpt2_sft.wls\"]];"],
  para[{
    "Five rounds of SFT on 20 examples took ", math["~60"], " seconds ",
    "on a Mac mini.  Loss dropped from 0.93 to 0.31.  Examples:"
  }],

  para[{
    "User: What is a sonnet?\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " A sonnet is a poem in which a poem is followed by a chorus, but a ",
    "chorus is not a poem."
  }],
  para[{
    "User: Tell me a Shakespeare quote.\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " A melancholy verse is a great tragedy."
  }],
  para[{
    "User: How are you?\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " I am well."
  }],
  para[{
    "User: Who wrote Hamlet?\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " Hamlet is a playwright who wrote the play in 1763."
  }],
  para[{
    "Real conversational English.  Some answers are wrong (1763 is not ",
    "1601), but the model now follows the chat template; the underlying ",
    "language ability is intact from pretraining."
  }],

  section["7.3  Parameter-efficient SFT: LoRA"],
  figureCell["figures/illustration_lora.png",
    "Figure 7.2.  LoRA decomposition.  The frozen pretrained weight W is left alone; two small matrices A (8x768) and B (768x8) are added.  Effective weight: W + B.A.  Trainable parameters: 12K vs 589K, a 48-fold reduction."],
  para[{
    "Full SFT updates every weight in the pretrained model.  ",
    StyleBox["LoRA", FontWeight -> Bold],
    " (Hu et al. 2021) updates a tiny fraction.  For each linear ",
    "layer with weight ", math["W"], " of shape ",
    math[RowBox[{SubscriptBox["d", "in"], "\[Times]",
                 SubscriptBox["d", "out"]}]], ", freeze ", math["W"],
    " and learn two small matrices ", math["A"], " and ", math["B"],
    " of rank ", math["r"], " so the effective weight at inference is ",
    math[RowBox[{"W", "+", "BA"}]], "."
  }],
  input["LLMAtoZ`Lora`loraParameterCount[768, 768, #] & /@ {4, 8, 16, 32, 64}"],
  para[{
    "12K LoRA parameters per 768\[Times]768 layer at rank 8 versus 590K ",
    "for full fine-tuning.  48\[Times] reduction per layer; with 12 ",
    "layers and four LoRA-wrapped projections per block (Q, K, V, O), ",
    "the total trainable parameter count drops to ~1.2M vs 28M for ",
    "full SFT."
  }],

  section["7.4  Classification fine-tuning (Raschka chapter 6)"],
  para[{
    "Same pretrained model, different task.  Strip the LM head, pool ",
    "the last-position hidden state with a ",
    StyleBox["SequenceLastLayer", FontFamily -> "Courier"],
    ", project to ", math[SubscriptBox["n", "classes"]], " logits.  ",
    "The rest of the pretrained backbone is reused as-is."
  }],
  input["(* Demo: classify Shakespeare excerpts as tragedy vs comedy *)
LLMAtoZ`Classification`shakespeareGenreLabels[][[1 ;; 4]]"],

  (* ============ Part 8: RL ============ *)

  chapter["8.  Reinforcement learning post-training"],

  para[{
    "SFT teaches the model to imitate.  RL teaches it to optimise.  ",
    "After SFT, we take the model, sample completions on a task with a ",
    "verifiable reward (or a reward model), and update the policy to ",
    "make high-reward completions more likely.  Following nanochat we ",
    "use ", StyleBox["GRPO", FontWeight -> Bold],
    " in its simplest form: no reference model, no KL term, just ",
    "centred advantages over a group of rollouts."
  }],

  figureCell["figures/illustration_grpo.png",
    "Figure 8.1.  One GRPO step.  Sample K rollouts from the policy at temperature T, score with a reward function, centre rewards to get advantages, take a policy-gradient step.  No value head, no KL term, no reference model.  The variance across the K rollouts IS the advantage signal."],

  section["8.1  The recipe in five lines"],
  para[{
    "1. ", StyleBox["Sample.", FontWeight -> Bold],
    " ", math["K"],
    " rollouts from the current policy at temperature ", math["T"], ".\n",
    "2. ", StyleBox["Score.", FontWeight -> Bold],
    " Each rollout gets a reward (binary for verifiable tasks).\n",
    "3. ", StyleBox["Centre.", FontWeight -> Bold],
    " ",
    math[RowBox[{"advantages", " = ", "rewards", " - ", "mean(rewards)"}]],
    ".  No z-score normalisation (per DAPO).\n",
    "4. ", StyleBox["Compute.", FontWeight -> Bold],
    " Policy-gradient loss ",
    math[RowBox[{"L", " = ", "-", "\[Sum]"}]],
    "  advantage_i * log p_\[Theta](rollout_i).\n",
    "5. ", StyleBox["Step.", FontWeight -> Bold],
    " ADAM on L."
  }],
  para[{
    "What is deliberately absent: no reference model, no KL ",
    "divergence, no value head, no importance weighting.  The reward ",
    StyleBox["is", FontSlant -> Italic],
    " the ground truth; the variance across the ", math["K"],
    " rollouts ", StyleBox["is", FontSlant -> Italic],
    " the advantage signal.  It is shocking how simple this is once you ",
    "accept that you don't need PPO's machinery for tasks with ",
    "verifiable correctness."
  }],

  section["8.2  The toy verifiable task"],
  para[{
    "For the educational demo we use the trivial task ",
    "\[OpenCurlyDoubleQuote]end your continuation with the ",
    StyleBox["<|eot|>", FontFamily -> "Courier"],
    " token within k tokens.\[CloseCurlyDoubleQuote]  Binary reward: ",
    "1.0 if the rollout contains ",
    StyleBox["<|eot|>", FontFamily -> "Courier"],
    ", else 0.0.  Small models post-trained from random init learn to ",
    "do this in a few minutes of CPU time, while preserving all the ",
    "GRPO bookkeeping that scales to real verifiable tasks (GSM8K, ",
    "code, format-following)."
  }],

  (* ============ Part 9: Chat widget ============ *)

  chapter["9.  An interactive chat widget"],

  para[{
    "We close with the artefact every reader actually wants: a chat ",
    "interface, inside the notebook, talking to the model we just ",
    "built.  An input field, a transcript pane, a streaming generator.  ",
    "Drop the widget cell into a notebook, evaluate, type, get a reply."
  }],

  para[{
    "The widget is a ",
    StyleBox["DynamicModule", FontFamily -> "Courier"],
    " holding a running transcript, an input field, and Send/Reset ",
    "buttons.  Each Send triggers ",
    StyleBox["chatTurn", FontFamily -> "Courier"],
    ", which formats the existing transcript and the new user message ",
    "in the chat template, runs the model autoregressively until ",
    StyleBox["<|eot|>", FontFamily -> "Courier"],
    " or the max-tokens limit, and appends the assistant turn.  A ",
    "screenshot of what the widget looks like in action:"
  }],

  figureCell["discussion-model/figures/chat_widget_mockup.png",
    "Figure 9.1.  Static snapshot of the in-notebook chat widget.  The transcript at top accumulates turns; the input field and Send/Reset buttons drive new requests.  Evaluate the cell below in Wolfram Desktop to interact with the real version (requires a loaded SFT checkpoint)."],

  input["(* The chat widget.  Open this notebook in Wolfram Desktop and
   evaluate to interact. *)
If[ValueQ[sftNet],
  LLMAtoZ`Chat`chatWidget[sftNet,
    bpe[\"MergeRules\"], tokId,
    AssociationThread[Range[Length[vocab]] -> vocab],
    \"MaxNew\" -> 60, \"Temperature\" -> 0.8, \"TopK\" -> 40],
  Style[\"SFT checkpoint not loaded; run Path A first.\", Red]]"],

  para[{
    "The model is small and undertrained for the scale of natural ",
    "language; it will not pass a Turing test.  But the entire pipeline ",
    "from BPE tokenisation through GRPO post-training is something you ",
    "wrote (and could change).  That is the point of this notebook."
  }],

  (* ============ Part 10: Scaling up ============ *)

  chapter["10.  Scaling up: from 20 toy pairs to 20,000 Alpaca instructions"],

  figureCell["discussion-model/figures/scaling_up.png",
    "Figure 10.0.  The plot of this chapter: same pipeline, but the toy SFT corpus of 20 hand-written Q&A pairs (Chapter 7) is replaced by 20,000 instruction\[Dash]response pairs from Alpaca-cleaned, and the 124M small base is replaced by the 355M medium.  Two knobs, simultaneously."],

  para[{
    "The chat widget above works \[LongDash] but the answers are obviously ",
    "wrong (\"Hamlet is a playwright who wrote the play in 1763\").  The ",
    "model from Chapter 7 was fine-tuned on twenty hand-written Q&A pairs.  ",
    "It learned the ",
    StyleBox["template", FontSlant -> Italic],
    " (User: \[Ellipsis] Assistant: \[Ellipsis]) but no real content.  ",
    "If we want a model that ",
    StyleBox["actually", FontSlant -> Italic],
    " gives sensible answers, we need to pull two levers at once: a ",
    "bigger base model, and a much bigger SFT dataset."
  }],

  para[{
    "This chapter documents the actual run we did to find out what each ",
    "lever buys us.  Same pipeline; we just feed it more.  Code is in ",
    StyleBox["discussion-model/src/", FontFamily -> "Courier"],
    " of the repo, separate from the per-part notebooks so you can run it ",
    "without disturbing the rest of the series."
  }],

  section["10.1  The plan, and what we ended up doing"],

  para[{
    "Three knobs.  ",
    StyleBox["Base model", FontWeight -> Bold],
    " (GPT-2 small, medium, large \[LongDash] 124M / 355M / 774M ",
    "params).  ",
    StyleBox["Dataset", FontWeight -> Bold],
    " (toy 20 pairs \[Rule] Alpaca-cleaned, 51,760 instructions, Apache ",
    "2.0).  ",
    StyleBox["Compute", FontWeight -> Bold],
    " (Mac mini CPU \[Rule] Wolfram Cloud GPU via ",
    StyleBox["RemoteBatchSubmit", FontFamily -> "Courier"], ")."
  }],

  para[{
    "What worked, what didn't:"
  }],
  para[{
    "\[Bullet] ",
    StyleBox["GPT-2 small (124M) SFT on 10k Alpaca, Mac mini CPU, 3 ",
             FontWeight -> Bold],
    StyleBox["epochs", FontWeight -> Bold], ".  Died of an unrelated Wolfram ",
    "15.0 prerelease license check at 2h26m, mid-epoch 2.  Epoch-1 ",
    "checkpoint survives.\n",
    "\[Bullet] ",
    StyleBox["GPT-2 medium (355M) SFT on full Alpaca, cloud GPU1xL40S, 1 ",
             FontWeight -> Bold],
    StyleBox["epoch", FontWeight -> Bold], ".  Out-of-memory at 37k of 51,760 ",
    "examples.  Cause: our own ",
    StyleBox["TrainingProgressFunction", FontFamily -> "Courier"],
    " periodically saving a ",
    StyleBox["NetTake", FontFamily -> "Courier"],
    " copy to a ",
    StyleBox["CloudObject", FontFamily -> "Courier"],
    " leaked 1.4 GB per checkpoint.  \[Sterling]5.48 burnt.\n",
    "\[Bullet] ",
    StyleBox["GPT-2 medium SFT on 20k Alpaca, cloud GPU4xL4, 1 epoch ",
             FontWeight -> Bold],
    "\[Dash] same code with the periodic checkpoint removed.  ",
    StyleBox["Succeeded.", FontWeight -> Bold],
    "  53.6 min wall, final loss 1.367.  \[Sterling]12.75.  This is our ",
    "production discussion model.\n",
    "\[Bullet] ",
    StyleBox["GPT-2 large (774M) SFT on 10k Alpaca, two attempts on ",
             FontWeight -> Bold],
    StyleBox["GPU4xL4", FontWeight -> Bold], ".  Both got within ",
    "one batch (>99.9%) of finishing the epoch, then OOM\[CloseCurlyQuote]d ",
    "on the post-training ",
    StyleBox["CloudPut", FontFamily -> "Courier"],
    " cascade: 3 GB net + ",
    StyleBox["NetTake", FontFamily -> "Courier"],
    " copy + serialization buffer > 24 GiB single-GPU memory.  ",
    "\[Sterling]25.46 burnt.  Lesson banked, large shelved."
  }],

  para[{
    "Total spend: about \[Sterling]45 (\[TildeTilde] $56) across the cloud ",
    "runs.  Time: a working day from \"can we even submit a GPU job\" to ",
    "\"the model says 'The capital of France is Paris.'\"  The educational ",
    "point isn't the dollar figure; it's that the bugs you hit at this ",
    "scale are not the bugs you hit on Mac-mini-sized toy runs."
  }],

  section["10.2  The dataset: Alpaca-cleaned"],

  para[{
    "Alpaca-cleaned is the Stanford ",
    StyleBox["Alpaca", FontSlant -> Italic],
    " dataset (Taori et al. 2023) after a community cleanup by ",
    StyleBox["yahma", FontFamily -> "Courier"],
    " on HuggingFace.  51,760 instruction\[Dash]response pairs in three ",
    "fields: ",
    StyleBox["instruction", FontFamily -> "Courier"], ", ",
    StyleBox["input", FontFamily -> "Courier"], " (optional context), ",
    StyleBox["output", FontFamily -> "Courier"], ".  Apache 2.0.  44 MB on ",
    "disk."
  }],

  input["(* Fetch + look at one example. *)
example = First @ LLMAtoZ`Discussion`loadAlpacaExamples[1];
example"],

  para[{
    "Format each row as plain text with markers GPT-2's BPE already ",
    "tokenises naturally:"
  }],
  input["LLMAtoZ`Discussion`formatPair[example]"],
  para[{
    "Why plain text and not the ",
    StyleBox["<|user|>...<|assistant|>...<|eot|>", FontFamily -> "Courier"],
    " template from Chapter 7?  Because extending GPT-2's tied input/output ",
    "embedding to add three new tokens is fiddly (we tried; it broke ",
    "NetTrain's autograd in a way we didn't want to spend more credits ",
    "debugging).  Plain text keeps the 50,257-entry vocabulary intact and ",
    "the weight tying alive.  The price: no learned end-of-turn signal, so ",
    "the trained model sometimes babbles past the answer (see 10.5)."
  }],

  section["10.3  Surgery on the pretrained NetModel"],

  para[{
    "The unmodified ",
    StyleBox["NetModel", FontFamily -> "Courier"],
    " for GPT-2 small / medium / large is wired for ",
    StyleBox["inference", FontSlant -> Italic],
    ": a long prompt in, one next-token distribution out.  Its tail is ",
    StyleBox["SequenceLastLayer", FontFamily -> "Courier"],
    " \[Rule] ",
    StyleBox["LinearLayer[vocab]", FontFamily -> "Courier"],
    " \[Rule] ",
    StyleBox["SoftmaxLayer", FontFamily -> "Courier"],
    ".  For training we need per-position logits at every step.  Three ",
    "changes, in three lines:"
  }],
  input["(* Pull the pretrained net, drop the inference tail, wrap the
   per-position classifier with NetMapOperator so it applies row-wise. *)
gpt = NetModel[{
  \"GPT-2 Transformer Trained on WebText Data\",
  \"Size\" -> \"345M\", \"Task\" -> \"LanguageModeling\"}];
trunk = NetChain[{
  NetExtract[gpt, \"embedding\"],
  NetExtract[gpt, \"decoder\"],
  NetMapOperator[NetExtract[gpt, \"classifier\"]]}];
Information[trunk, \"ArraysTotalElementCount\"]"],
  para[{
    "Same 354,823,168 parameters as before.  The classifier's ",
    StyleBox["SharedArrays", FontFamily -> "Courier"],
    " linkage to the token embedding stays intact; we don't have to ",
    "re-tie it manually.  This trunk is what we train."
  }],

  section["10.4  RemoteBatchSubmit: code that runs on someone else's GPU"],

  figureCell["discussion-model/figures/cloud_training.png",
    "Figure 10.1.  RemoteBatchSubmit conceptually: we send the training expression up to the Wolfram Cloud, it lands on a GPU machine somewhere (an L4 or L40S), runs there, and we get the trained net back on disk.  The Mac mini at the bottom stays at idle."],

  para[{
    "Mac mini CPU can SFT GPT-2 small overnight.  Anything bigger needs ",
    "a GPU.  Wolfram Cloud's batch system exposes one through ",
    StyleBox["RemoteBatchSubmit", FontFamily -> "Courier"],
    ".  The pattern:"
  }],
  input["(* Build an environment that targets a specific machine class. *)
env = RemoteBatchSubmissionEnvironment[\"WolframBatch\",
  <|\"RemoteMachineClass\" -> \"GPU1xL40S\"|>];
(* Available classes on this account include:
     Basic1x8       -- default CPU box, very cheap
     Basic4x16      -- bigger CPU
     GPU1xL40S      -- 1x NVIDIA L40S, 44 GiB VRAM, ~1120 credits/hr
     GPU4xL4        -- 4x NVIDIA L4, 89 GiB total, ~2765 credits/hr
   ~200 credits per pound sterling. *)"],

  para[{
    "And then submit a piece of WL code that runs out there.  ",
    StyleBox["Critical gotcha", FontWeight -> Bold],
    ": ",
    StyleBox["RemoteBatchSubmit", FontFamily -> "Courier"],
    " holds its second argument internally.  An explicit ",
    StyleBox["Hold[\[Ellipsis]]", FontFamily -> "Courier"],
    " wrapper around the Module ",
    StyleBox["survives", FontSlant -> Italic],
    " to the remote, where evaluating ",
    StyleBox["Hold[Module[\[Ellipsis]]]", FontFamily -> "Courier"],
    " just returns the Hold itself \[LongDash] the Module never runs.  ",
    "Pass the Module bare:"
  }],
  input["(* CORRECT *)
(* job = RemoteBatchSubmit[env, Module[{out = <||>, ...}, ...; out]]; *)
(* WRONG -- the Module never evaluates remotely *)
(* job = RemoteBatchSubmit[env, Hold[Module[...]]]; *)"],

  para[{
    "We learnt this by burning \[Sterling]0.27 on a no-op job that ",
    "looked successful but had no Result.  Documented at length in ",
    StyleBox["discussion-model/src/submit_gpu_prod.wls", FontFamily -> "Courier"],
    "."
  }],

  section["10.5  What the trained medium model actually says"],

  para[{
    "After the successful 20k-Alpaca, 1-epoch run on GPU4xL4, the saved ",
    "stripped net lives at ",
    StyleBox["discussion-model/data/discussion_medium_20k.wlnet", FontFamily -> "Courier"],
    " (1.35 GB, gitignored; the CloudObject URL is in the run summary).  ",
    "Compare its answers to the Chapter 7 toy SFT model:"
  }],

  para[{
    "Sample generations from the Alpaca-trained medium (",
    StyleBox["T = 0.7, top-k = 40", FontFamily -> "Courier"],
    "):"
  }],

  para[{
    StyleBox["User:", FontWeight -> Bold],
    " What is the capital of France?\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " The capital of France is Paris."
  }],
  para[{
    StyleBox["User:", FontWeight -> Bold],
    " Give three tips for staying healthy.\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " 1. Eat a balanced and nutritious diet: A healthy diet can help to ",
    "maintain good health by providing the necessary vitamins, minerals, ",
    "and fiber to support good heart health and mental wellbeing\[Ellipsis] ",
    "2. Get enough sleep\[Ellipsis]"
  }],
  para[{
    StyleBox["User:", FontWeight -> Bold],
    " Recommend a book to read.\n",
    StyleBox["Assistant:", FontWeight -> Bold],
    " One great book to read is \"To Kill a Mockingbird\" by Harper Lee. ",
    "This novel is a classic that teaches the importance of empathy, the ",
    "power of a good deed, and the true worth of one's choices."
  }],

  para[{
    "Compare to Chapter 7's toy SFT for the same prompts:"
  }],
  para[{
    StyleBox["User:", FontWeight -> Bold], " Who wrote Hamlet?\n",
    StyleBox["Assistant", FontWeight -> Bold],
    StyleBox[" (toy SFT, Chapter 7)", FontSlant -> Italic], ": Hamlet is ",
    "a playwright who wrote the play in 1763.\n",
    StyleBox["Assistant", FontWeight -> Bold],
    StyleBox[" (Alpaca SFT, this chapter)", FontSlant -> Italic],
    ": Shakespeare was a Roman Catholic poet, dramatist, and playwright ",
    "from about 586 to 1140."
  }],

  para[{
    "Both are wrong on dates.  The Alpaca model is right on ",
    StyleBox["author and role", FontSlant -> Italic],
    " (\"poet, dramatist, and playwright\") in a way the toy SFT isn't.  ",
    "It is also obviously not GPT-4.  This is roughly what 355M parameters ",
    "and a 20k-instruction SFT buys: chat fluency and rough fact recall, ",
    "with regular errors in dates, names, and reasoning chains.  Bigger ",
    "base + more data + RL would push each of those metrics."
  }],

  section["10.6  Why we can't (easily) train GPT-2 medium on the Mac"],

  figureCell["discussion-model/figures/oom_humor.png",
    "Figure 10.2.  Out-of-memory: trying to save a 3 GB GPT-2 large checkpoint on a 24 GiB single-GPU L4 after training fills the device.  Training itself succeeded; the post-training NetTake + CloudPut cascade did not.  Twice."],

  para[{
    "The same surgery code that works on GPT-2 small (124M) hits a ",
    "specific low-level bug on GPT-2 medium when run on Mac CPU.  The ",
    "error is at the MXNet operator level:"
  }],
  para[{
    StyleBox["MXNetError: Check failed: assign(&dattr, vec.at(i)): \n",
             FontFamily -> "Courier"],
    StyleBox["Incompatible attr in node _copyto at 0-th output: \n",
             FontFamily -> "Courier"],
    StyleBox["expected [4096], got [1024]\n", FontFamily -> "Courier"]
  }],
  para[{
    "The numbers 1024 and 4096 are GPT-2 medium's ",
    StyleBox["d_model", FontFamily -> "Courier"],
    " and ",
    StyleBox["d_ff", FontFamily -> "Courier"],
    ".  Something in the gradient routing across our reconstructed chain ",
    "gets confused at this exact size.  Same code, same call, works ",
    "perfectly on cuDNN (cloud GPU) and on MXNet-CPU at small.  Not ",
    "fixable from the application layer.  Practical implication: ",
    "anything bigger than small needs the cloud."
  }],

  section["10.7  Inference is slow in Wolfram Language on Mac"],

  figureCell["discussion-model/figures/chat_thinking.png",
    "Figure 10.3.  Inference: the model reads the prompt, thinks, replies.  At 5 tokens/s on Mac CPU through WL, an 80-token answer takes 15\[Dash]20 seconds; PyTorch with Metal Performance Shaders does the same work at 27 tokens/s on the same machine."],

  para[{
    "Even after training, ",
    StyleBox["running", FontSlant -> Italic],
    " GPT-2 medium on Mac CPU through Wolfram Language is around 5 tokens ",
    "per second.  Replies of ~80 tokens take 15-20 seconds.  The bottleneck ",
    "is not the architecture or the weights; it's the inference path.  WL's ",
    "neural-network stack uses MXNet without an Apple Silicon Metal backend.  ",
    "Same model, same hardware, in PyTorch:"
  }],
  para[{
    "\[Bullet] ", StyleBox["PyTorch CPU", FontWeight -> Bold],
    ": 14.4 tok/s (~3x WL).\n",
    "\[Bullet] ", StyleBox["PyTorch MPS", FontWeight -> Bold],
    " (Metal Performance Shaders on Apple Silicon GPU): 27.3 tok/s ",
    "(~5x WL)."
  }],
  para[{
    "Benchmark code: ",
    StyleBox["discussion-model/src/bench_gpt2_python.py", FontFamily -> "Courier"],
    ".  Two implications for anyone reading along:"
  }],
  para[{
    "1. ",
    StyleBox["For inference-heavy demos", FontWeight -> Bold],
    ", deploy the trained model via PyTorch + MPS rather than serving it ",
    "from WL.  Architecture is the same; only the weights differ from the ",
    "stock HF checkpoint.  Our repo includes a ",
    StyleBox["weight bridge", FontFamily -> "Courier"],
    " that maps the trained ",
    StyleBox[".wlnet", FontFamily -> "Courier"],
    " into a HuggingFace state-dict (see ",
    StyleBox["discussion-model/src/export_wl_to_hf.wls", FontFamily -> "Courier"],
    " and ",
    StyleBox["chat_python.py", FontFamily -> "Courier"],
    ").\n",
    "2. ",
    StyleBox["For chat-from-the-notebook", FontWeight -> Bold],
    ", stream tokens and use shorter context.  ",
    StyleBox["discussion-model/src/chat_cli.wls", FontFamily -> "Courier"],
    " does this: it writes each generated token to stdout as soon as it's ",
    "produced, and stops cleanly on " ,
    StyleBox["\" User:\"", FontFamily -> "Courier"], " / ",
    StyleBox["\" Assistant:\"", FontFamily -> "Courier"],
    " / 3-token repetition.  Wall time is still 15-20 s for an 80-token ",
    "reply, but the user sees the answer ",
    StyleBox["forming", FontSlant -> Italic],
    " rather than waiting in silence."
  }],

  section["10.8  Summary"],

  para[{
    "\[FilledSquare] ", StyleBox["355M-parameter pretrained base + 20k ",
                                  FontWeight -> Bold],
    StyleBox["real instructions = a model that holds the chat format and ",
             FontWeight -> Bold],
    StyleBox["gets simple facts right.", FontWeight -> Bold], "  ",
    "Same surgery as Chapter 7, same NetTrain loop, three different ",
    "knobs turned at once: base size, dataset size, compute target.\n",
    "\[FilledSquare] ",
    StyleBox["RemoteBatchSubmit is the production path.", FontWeight -> Bold],
    "  ~\[Sterling]13 buys you a 53-minute SFT epoch on GPT-2 medium.  ",
    "The pattern that worked: machine class ",
    StyleBox["GPU4xL4", FontFamily -> "Courier"],
    ", batch size 2-4, no periodic CloudPut inside ",
    StyleBox["TrainingProgressFunction", FontFamily -> "Courier"],
    " (the leak that bit us first).\n",
    "\[FilledSquare] ",
    StyleBox["Large is harder than it looks.", FontWeight -> Bold],
    "  Training itself works; saving the trained 3 GB net OOMs on a ",
    "single L4's 24 GiB.  Solvable with multi-GPU NetTrain ",
    "configuration or local-disk-then-upload, but past the budget here.\n",
    "\[FilledSquare] ",
    StyleBox["For interactive use, leave WL for the inference step.",
             FontWeight -> Bold],
    "  Architecture is identical to HuggingFace's; the weight bridge is ",
    "two short scripts.  PyTorch + MPS is ~5x WL on the same machine."
  }],

  (* ============ Closing ============ *)

  (* Numbered 12 because Chapter 11 (pretraining from scratch) is inserted
     before this closing chapter by withPart9[] (part9_cells.wl). *)
  chapter["12.  Where to go from here"],

  para[{
    "The natural next steps, after Chapter 10's discussion-model arc:"
  }],
  para[{
    "\[FilledSquare] ",
    StyleBox["Multi-turn SFT data.", FontWeight -> Bold],
    " Alpaca-cleaned is single-turn.  SmolTalk (~150k multi-turn ",
    "dialogues, HuggingFace) would teach the model to actually keep ",
    "context across turns instead of treating every reply as the start ",
    "of a new conversation.\n",
    "\[FilledSquare] ",
    StyleBox["Proper end-of-turn token.", FontWeight -> Bold],
    " The plain-text ",
    StyleBox["\"User: \" \[Ellipsis] \"Assistant: \" \[Ellipsis]", FontFamily -> "Courier"],
    " template has no learned EOS, so the model babbles past the answer ",
    "(see Chapter 10.5).  Extending GPT-2's tied vocabulary with three ",
    "special tokens \[LongDash] ",
    StyleBox["<|user|>", FontFamily -> "Courier"], ", ",
    StyleBox["<|assistant|>", FontFamily -> "Courier"], ", ",
    StyleBox["<|eot|>", FontFamily -> "Courier"],
    " \[LongDash] is the canonical fix.  The challenge is the weight ",
    "tying: an embedding+classifier resize needs to preserve ",
    StyleBox["SharedArrays", FontFamily -> "Courier"],
    ", which NetReplacePart doesn't do cleanly today.\n",
    "\[FilledSquare] ",
    StyleBox["Reasoning-style RL.", FontWeight -> Bold],
    " GRPO (Chapter 8) on top of the Alpaca-trained medium, with GSM8K ",
    "rollouts and a ground-truth answer-matching reward.  Same ",
    "policy-gradient loop, just a verifiable task instead of the toy ",
    "end-with-EOT.\n",
    "\[FilledSquare] ",
    StyleBox["Better tokeniser.", FontWeight -> Bold],
    " Implement byte-level BPE matching GPT-2 exactly, with the ",
    StyleBox["\[CapitalGamma]", FontFamily -> "Courier"],
    " space-prefix convention.  Currently the WL embedder we built in ",
    "Chapter 2 uses a different convention for spaces.\n",
    "\[FilledSquare] ",
    StyleBox["RoPE instead of learned absolute positional embeddings.",
             FontWeight -> Bold],
    "  Su et al. 2021; modern transformers all use this."
  }],
  para[{
    "Every one of these is a single-package edit away.  The whole ",
    "scaffolding is yours."
  }],

  section["References"],
  para[{
    "Gage, P. (1994). A New Algorithm for Data Compression. C Users Journal 12(2).\n",
    "Sennrich, R., Haddow, B., Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.\n",
    "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.\n",
    "Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2 technical report.)\n",
    "Holtzman, A. et al. (2020). The Curious Case of Neural Text Degeneration. (Top-p / nucleus sampling.)\n",
    "Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.\n",
    "Taori, R. et al. (2023). Stanford Alpaca: An Instruction-Following LLaMA Model. (The dataset SFT'd in Chapter 10.)\n",
    "yahma. (2023). alpaca-cleaned (HuggingFace dataset, Apache 2.0). The cleaned-up variant we actually used.\n",
    "Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.\n",
    "Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models. (Chinchilla.)\n",
    "Leviathan, Y., Kalman, M., Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding.\n",
    "Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning. (GRPO.)\n",
    "Xiao, G. et al. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR.\n",
    "Raschka, S. (2024). Build a Large Language Model from Scratch. Manning.\n",
    "Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks. char-rnn.\n",
    "Karpathy, A. (2025). nanochat."
  }]
};

nb = Notebook[cells, WindowSize -> {1000, 800}, Saveable -> True];
out = FileNameJoin[{Directory[],
  If[$evalMode, "LLM_A_to_Z_evaluated.nb", "LLM_A_to_Z.nb"]}];
Export[out, nb, "NB"];
Print["Wrote ", out, " (", FileByteCount[out], " bytes, ",
      Length[cells], " cells, ", If[$evalMode, "evaluated", "plain"], ")"];
