(* part9_cells.wl  --  shared Part-9 content for the published notebooks.

   Defines:
     flowchartGraphics       the native "map of the notebook" Graphics
     part9FlowCells          flowchart cells (CellTags PART9-FLOWCHART)
     part9ChapterCells       the from-scratch chapter, as Chapter 11, with each
                             detailed section in a COLLAPSED CellGroupData so
                             the full transcripts/tables live in the notebook
                             without making it long (CellTags PART9-CHAPTER)
     withPart9[cellList]     insert flowchart before the first Chapter, and the
                             from-scratch chapter immediately BEFORE the closing
                             "Where to go from here" chapter (renumbered to 12)
                             so the summary stays last.  Idempotent.
     insertPart9File[path]   apply withPart9 to an existing .nb on disk

   Used two ways:
     - build_master_notebook.wls wraps its assembled cells with withPart9[].
     - insert_part9_section.wls calls insertPart9File[] to patch existing .nb.

   All numbers/transcripts are the REAL verified results (see PART9_RESULTS.md):
   v2 ppl 55, v3 ppl 52; ~$2.6 & ~10.4h per pretrain; verbatim chat outputs.
   The detailed sections are collapsed by default; the reader expands them. *)

part9MarkerFlow = "PART9-FLOWCHART";
part9MarkerChap = "PART9-CHAPTER";
part9ClosingTitle = "Where to go from here";

(* locate build_flowchart.wls next to this file, regardless of CWD *)
part9SrcDir = DirectoryName[$InputFileName];
Get[FileNameJoin[{part9SrcDir, "build_flowchart.wls"}]];
If[Head[flowchartGraphics] =!= Graphics,
  Print["part9_cells.wl: WARNING flowchartGraphics not built"]];

part9FlowCells = {
  Cell["A map of the whole notebook", "Subsection", CellTags -> part9MarkerFlow],
  Cell["The notebook follows three paths to a working model. Path A trains the \
original GPT architecture on Tiny Shakespeare; Path B reuses OpenAI's \
pretrained GPT-2 weights for fine-tuning, RL, and the discussion model; Path C \
(Chapter 11, new) pretrains a model from scratch on cloud GPUs. The chart \
shows each model and the chapter where it is built.", "Text",
    CellTags -> part9MarkerFlow],
  Cell[BoxData[ToBoxes[Show[flowchartGraphics, ImageSize -> 850]]],
    "Output", CellTags -> part9MarkerFlow, TextAlignment -> Center]
};

(* a Section group, OPEN/expanded by default (the reader can collapse it).
   Header carries the marker so the whole group is stripped on a re-run. *)
closedSection[header_String, body_List] :=
  Cell[CellGroupData[
    Prepend[body, Cell[header, "Section", CellTags -> part9MarkerChap]],
    Open]];

part9ChapterCells = {
  Cell["11.  Pretraining a model from scratch", "Chapter",
    CellTags -> part9MarkerChap],

  Cell["Every model so far has either been trained on a few megabytes of Tiny \
Shakespeare (Chapters 5\[Dash]6) or built on top of OpenAI's pretrained GPT-2 \
weights (Chapters 7\[Dash]10). Neither is pretraining in the modern sense: \
starting from random numbers and a large corpus of ordinary web text, and \
spending real compute until the network learns English on its own. That last \
step is the one piece of the modern LLM recipe the series had not yet shown \
end to end in the Wolfram Language. This chapter does it \[Dash] for about the \
price of a sandwich.", "Text", CellTags -> part9MarkerChap],

  Cell["The plan: build a small but modern transformer (the same architecture \
family as LLaMA, Mistral, Qwen and Karpathy's nanochat), rent a single cloud \
GPU by the hour, stream a few hundred million tokens of educational web text \
through it, and watch the loss fall from \"random guessing\" to \"writes \
fluent English\". A short instruction-tuning pass then turns the text \
predictor into something you can ask questions. The detailed sections below \
are collapsed to keep the chapter short on screen \[Dash] click any heading to \
expand the full derivations, tables and transcripts.", "Text",
    CellTags -> part9MarkerChap],

  closedSection["11.1  A modern architecture: RMSNorm, RoPE, SwiGLU", {
    Cell["The transformer block from Chapter 4 is the original 2017 design: \
LayerNorm, learned absolute position embeddings, and a GELU feed-forward \
network. Every open-weights model released since 2023 has quietly replaced all \
three components. Chapter 11 rebuilds the block with the modern choices, each \
of which is a small, well-motivated change.", "Text"],
    Cell["RMSNorm instead of LayerNorm. LayerNorm subtracts the mean and \
divides by the standard deviation, then applies a learned scale and shift. \
RMSNorm keeps only the rescaling: it divides each vector by its \
root-mean-square, x / Sqrt[Mean[x^2] + \[Epsilon]]. Dropping the \
mean-subtraction removes a reduction per call and, empirically, costs nothing \
in quality. Our implementation also drops the learnable gain, which the LLaMA \
ablations show is worth well under 1% \[Dash] so this normaliser has zero \
trainable parameters.", "Text"],
    Cell["RoPE instead of learned positional embeddings. The original model \
adds a learned vector to each token to tell it where it sits in the sequence. \
Rotary position embedding (Su et al. 2021) instead rotates the query and key \
vectors inside each attention head by an angle proportional to position. \
Concretely, each consecutive pair of feature dimensions is treated as a point \
in a 2D plane and spun by angle m\[CenterDot]\[Theta] at position m; because \
both query and key are rotated by their own positions, their dot product keeps \
only the angle difference (i - j)\[CenterDot]\[Theta], so the attention score \
between positions i and j depends only on the offset i - j \[Dash] exactly what \
a position code should encode. The payoff: no positional parameters to learn \
(saving maxSeqLen \[Times] dModel weights), and \[Dash] for implementations \
that extend the rotation tables \[Dash] graceful generalisation past the \
training context length. (Our build compiles fixed sine/cosine tables for \
context 512, so running longer would mean rebuilding them.)", "Text"],
    Cell["SwiGLU instead of a GELU MLP. The classic feed-forward block is \
Linear \[Rule] GELU \[Rule] Linear. SwiGLU (Shazeer 2020) uses two parallel \
input projections, applies a SiLU gate to one and multiplies it against the \
other before the output projection: down(SiLU(gate(x)) \[CenterDot] up(x)). It \
adds a third linear projection, so production models usually shrink the \
feed-forward width to keep the parameter budget matched; at that matched budget \
it is a small but consistent quality gain, which is why it is now standard.",
      "Text"],
    Cell["The configuration used here \[Dash] call it the \"nano-30M\" spec: \
dModel 256, 8 attention heads, 4 layers, feed-forward width 1024, GPT-2 BPE \
vocabulary of 50,257, context length 512. The parameter count follows the \
closed form 2 V d + L (4 d^2 + 3 d f), with V the vocabulary, d the model \
width, L the layers and f the feed-forward width. Plugging in: the four \
transformer blocks hold 4 \[Times] (4\[CenterDot]256^2 + 3\[CenterDot]256\
\[CenterDot]1024) \[TildeEqual] 4.2M, and the token embedding and untied \
output projection are V \[Times] d = 12.9M each. Total: 29,925,888 \
\[TildeEqual] 29.9M trainable parameters \[Dash] hence \"nano-30M\". (The full \
NetChain reports ~40M arrays, but ~10M of those are the frozen RoPE sine/cosine \
and causal-mask tables, which carry no gradient.) The embedding and output \
projection together are 26 of the 30M trainable parameters; tying them into one \
shared matrix \[Dash] the standard GPT-2 trick \[Dash] would remove ~12.9M of \
them, the obvious next optimisation.", "Text"]
  }],

  closedSection["11.2  Why train from scratch, and on what data", {
    Cell["Fine-tuning a pretrained model (Chapters 7\[Dash]10) is cheap \
because someone else already paid for the hard part: the hundreds of \
GPU-years that taught GPT-2 the statistics of English. Pretraining is that \
hard part. Doing it ourselves, even at toy scale, is what turns the series \
from \"how to use a language model\" into \"how a language model comes to \
exist\".", "Text"],
    Cell["The corpus is FineWeb-Edu, HuggingFace's education-quality, \
deduplicated filter of Common Crawl. It is clean enough that a small model \
spends its limited capacity learning language rather than boilerplate and \
spam. We stream it, tokenise with the GPT-2 byte-pair encoder (the same \
vocabulary the model's output layer predicts), and write the token IDs to flat \
binary shards.", "Text"],
    Cell["How many tokens? The Chinchilla scaling result (Hoffmann et al. \
2022) says a compute-optimal model sees roughly 20 tokens per parameter. For \
~30M parameters that is ~600M tokens \[Dash] one pass over a 600M-token slice, \
which is what we use. More tokens would help, but 600M is the point where the \
budget and the scaling rule agree, and it keeps a full run near $3.", "Text"],
    Cell["Tokenisation runs locally in Python (tiktoken) at a few hundred \
thousand tokens per second; the resulting ~1.2 GB of uint16 IDs is split into \
thirty 20M-token shards and uploaded to cloud storage. A 20M-token shard takes \
about twenty minutes to train, so losing one to an interrupted GPU costs at \
most that \[Dash] which matters for the spot-instance strategy in 11.3.", "Text"]
  }],

  closedSection["11.3  Running it on rented GPUs (AWS Batch spot)", {
    Cell["A single NVIDIA T4 rents for about $0.25/hour on the AWS spot market \
\[Dash] roughly half the on-demand price. The catch is that spot instances can \
be reclaimed by AWS at any moment, so a multi-hour training run has to assume \
it will be interrupted and survive it.", "Text"],
    Cell["The harness is built around that assumption. One self-contained job \
loops over the shards in order, keeping the model resident on the GPU. After \
each shard it writes a checkpoint and a tiny progress marker to cloud storage. \
If the instance is reclaimed mid-run, AWS Batch automatically resubmits the \
job; on restart it reads the progress marker and the latest checkpoint and \
resumes from the next shard. There is no long-lived process on the local \
machine \[Dash] all of the resilience lives in the cloud.", "Text"],
    Cell["This design was not the first attempt. An earlier version drove the \
run from a script on a Mac, checkpointing only at the end of large chunks; a \
spot reclamation 43 minutes in lost the lot, and the orchestrating script \
itself later died. Moving the loop and the checkpointing entirely into the \
cloud job, with small shards, is what made the full run reliable: the \
production run then completed all thirty shards in a single attempt with zero \
interruptions.", "Text"],
    Cell["Throughput came out at about 16,100 tokens/second on the T4, so 600M \
tokens take ~10.4 hours and cost ~$2.60. That is a respectable rate for a \
framework-level implementation in single precision, without the hand-written \
mixed-precision kernels a dedicated training stack would use.", "Text"]
  }],

  closedSection["11.4  The training curve, and a learning-rate lesson", {
    Cell["The loss starts near ln(50257) \[TildeEqual] 10.83. That is not an \
accident: a freshly initialised model assigns roughly uniform probability \
1/V over the V = 50,257-token vocabulary, and the cross-entropy of the uniform \
distribution is -log(1/V) = log V, exactly the log of the vocabulary size. \
Watching the very first batch print ~10.8 is the quickest confirmation that the \
loss is wired up correctly (in WL, Log[50257] is 10.825).", "Text"],
    Cell["Throughout we report perplexity = exp(cross-entropy), which has a \
concrete reading: it is the effective number of equally-likely next tokens the \
model is choosing among. At initialisation that is exp(10.83) \[TildeEqual] \
50,257 \[Dash] genuinely no idea, all 50k tokens equally likely. A final \
perplexity of 52 means the model has narrowed 50,000-way confusion down to \
roughly 52-way: still uncertain, but a thousandfold sharper than chance.",
      "Text"],
    Cell["We ran the identical model on the identical data twice, changing only \
the learning-rate schedule. Run v2 used NetTrain's defaults on each shard \
\[Dash] thirty separate training calls, each with its own internal warm-up and \
decay, so the schedule restarts thirty times and keeps forgetting where the \
last left off. Run v3 explicitly used the ADAM optimiser with a single global \
cosine learning-rate decay keyed to the shard index, lr_i = lrMin + \
0.5 (lrMax - lrMin)(1 + cos(\[Pi] (i-1)/(N-1))), from 2\[Times]10^-3 down to \
2\[Times]10^-4. (One honest caveat: only the learning rate is global \[Dash] \
because NetTrain does not expose optimiser state between calls, ADAM's moment \
estimates still reset at each shard boundary.) Per-shard tail perplexity \
(lower is better):", "Text"],
    Cell["   tokens seen   v2 (per-shard LR)   v3 (global cosine LR)\n\
   20M             160                 157\n\
   60M              91                  89\n\
   100M             75                  76\n\
   200M             63                  63\n\
   300M             56                  58\n\
   500M             53                  53\n\
   600M             55                  52", "Text", FontFamily -> "Courier"],
    Cell["These are tail perplexities \[Dash] each row is the exponentiated \
mean of the last hundred batch losses on that shard. Because we train for a \
single epoch, every shard is fresh data the moment it is trained on, so its \
tail loss is effectively a held-out measurement: the model is being scored on \
tokens it is seeing for the first time, not memorised ones.", "Text"],
    Cell["v2 flattens out around 300M tokens \[Dash] the repeated schedule \
resets keep nudging it off the descent, and it even ticks back up at the very \
end (55 after 53) as the final shard's fresh warm-up disturbs the converged \
weights. v3 keeps improving to the end. The honest caveat: the final gap is \
small (perplexity 52 vs 55). The global schedule is clearly the right thing to \
do, but at this size and token budget the model is close to the best it can do \
regardless, so the schedule buys a cleaner curve rather than a dramatically \
better model. Both runs cost the same ~$2.60 and ~10.4 hours.", "Text"],
    Cell["For reference, GPT-2 small (124M parameters, far more training) sits \
in the low-30s perplexity on comparable text. A 30M model reaching the low-50s \
on a single 600M-token epoch is squarely where the scaling laws predict it \
should be.", "Text"]
  }],

  closedSection["11.5  What the base model writes", {
    Cell["With weights in hand, generation reuses the temperature / top-k \
machinery from Chapter 6 unchanged. These are verbatim continuations from the \
pretrained-only model (top-k with k = 40, temperature 0.9); the opening phrase \
is the prompt, the rest is the model.", "Text"],
    Cell["The history of the American Revolution, and the history of slavery \
in America, was brought to life by the American Revolution. The Civil War was \
based on the battles of the Civil War and the battlefield that led to the \
Civil War\[Ellipsis] the battle of Fort Knox, the Battle\[Ellipsis]", "Text"],
    Cell["Water is important because it can be used to make a smooth, clear \
and clean surface and remove and maintain the sealant and/or other parts of \
the body. It protects the sealant from injury and reduces the risk of \
damage.", "Text"],
    Cell["The sun is a star that is found in the eastern sky\[Ellipsis] There \
are three types of stars called supernovae, and you can get the number of \
star-determinants.", "Text"],
    Cell["My favorite food is pizza because they can be baked at the grocery \
store\[Ellipsis] it's been linked to health issues, your taste can be\
\[Ellipsis]", "Text"],
    Cell["This is the moment the whole exercise is about: grammatical, \
punctuated, on-topic English produced by a network that two days earlier was \
random numbers. The topical association is real \[Dash] history pulls in the \
Revolution and the Civil War, water pulls in cleaning and the body, the sun \
pulls in stars and supernovae. The weaknesses are equally clear: it loops \
(\"the Civil War\[Ellipsis] the Civil War\"), wanders, and has thin, often \
wrong, world knowledge. Both the strength and the weakness are exactly what a \
perplexity around 55 at 30M parameters predicts.", "Text"]
  }],

  closedSection["11.6  From text predictor to chat: instruction tuning", {
    Cell["The base model continues text; it cannot answer a question, because \
nothing in pretraining taught it that a question should be followed by an \
answer rather than by more questions. Supervised fine-tuning (SFT) fixes this \
by training on instruction/response pairs formatted with a fixed template. We \
use the Alpaca dataset (~52k examples) for three epochs \[Dash] about 18 \
minutes and $0.10 on the same T4.", "Text"],
    Cell["The fixed template is just a wrapper around each instruction/response \
pair, ending in the end-of-text token:", "Text"],
    Cell["Below is an instruction that describes a task. Write a response that \
appropriately completes the request.\n\n\
### Instruction:\nGive three tips for staying healthy.\n\n\
### Response:\n1. Eat a balanced diet. 2. Exercise regularly. 3. Sleep well.\
<|endoftext|>", "Text", FontFamily -> "Courier"],
    Cell["That template also solves a subtler problem: when to stop. Because \
every example ends with the end-of-text token, the model learns to emit it when \
its answer is done; at generation time we stop as soon as that token appears \
\[Dash] which is why the answers below terminate cleanly instead of running on \
as the base model does.", "Text"],
    Cell["Verbatim answers from the instruction-tuned model:", "Text"],
    Cell["Q: What is the capital of France?\n\
A: the capital of France is Britannica, France.", "Text",
      FontFamily -> "Courier"],
    Cell["Q: Give three tips for staying healthy.\n\
A: 1. Know your body and mind about all foods\n\
   2. Get enough sleep\n\
   3. Take regular breaks\n\
   4. Get enough sleep\n\
   5. Exercise regularly", "Text", FontFamily -> "Courier"],
    Cell["Q: Write a short poem about the sea.\n\
A: The ocean of the sea was a peaceful place. The sea was so dry. The sea was \
so thick that it slowly crept in the ocean\[Ellipsis] For ever again and \
again, the sea was so warm, soft surf, and a wonderful place.", "Text",
      FontFamily -> "Courier"],
    Cell["The transformation is real and the right shape: it gives a list when \
asked for a list, a poem when asked for a poem, and it stops. The content is \
another matter \[Dash] \"the capital of France is Britannica\" has the correct \
form and the wrong fact, and the health tips repeat themselves. SFT teaches \
behaviour, not knowledge; the knowledge has to come from a bigger base model.",
      "Text"]
  }],

  closedSection["11.7  Does a better base help? v2 vs v3, head to head", {
    Cell["The v3 base had the better perplexity (52 vs 55), so one might hope \
its chat model is noticeably sharper. It is not. Instruction-tuning both bases \
identically and asking the same questions gives the right shape from each and \
unreliable content from each. These are the verbatim outputs:", "Text"],
    Cell["Q: What is the capital of France?\n\
   v2: the capital of France is Britannica, France.\n\
   v3: The capital of France is Austria.", "Text", FontFamily -> "Courier"],
    Cell["Q: Explain what a star is in one sentence.\n\
   v2: A star is in one sentence.\n\
   v3: A star is in another sentence.", "Text", FontFamily -> "Courier"],
    Cell["Q: Give three tips for staying healthy.\n\
   v2: a 5-item list that repeats \"get enough sleep\".\n\
   v3: an 11-item list that repeats \"practice relaxation\".", "Text",
      FontFamily -> "Courier"],
    Cell["Both get the capital of France wrong (just differently), both \
collapse the one-sentence star definition, both pad their lists with repeats. \
The lesson is the useful, unglamorous one: a small improvement in base-model \
perplexity does not buy a meaningfully better assistant. The ceiling here is \
capacity and data, not tuning.", "Text"]
  }],

  closedSection["11.8  The honest result, and what would move it", {
    Cell["What this chapter demonstrates, end to end and reproducibly, is the \
complete modern pipeline working in the Wolfram Language: random \
initialisation \[Rule] a modern transformer \[Rule] hundreds of millions of \
tokens of real web text \[Rule] fluent English \[Rule] instruction-following \
behaviour, for about $3 and ~11 hours per version (pretraining ~$2.60 / \
~10.4h, instruction tuning ~$0.10 / ~18min).", "Text"],
    Cell["What it does not produce is a reliable chatbot, and it is worth \
being clear about why. At 30M parameters on 600M tokens the model is near the \
limit of what it can know. The levers that would actually move quality, in \
rough order of impact: (1) a larger model \[Dash] 100M\[Dash]200M parameters, \
which the same harness already supports given more GPU budget; (2) more tokens \
\[Dash] several epochs or a larger slice of FineWeb-Edu; (3) weight tying \
between the embedding and the output projection \[Dash] sharing one matrix \
instead of two removes ~12.9M parameters and lets that capacity be spent on \
depth or width; (4) multi-turn and higher-quality instruction data. The \
learning-rate schedule, by contrast, is already close to optimal.", "Text"]
  }],

  closedSection["11.9  Engineering notes (the parts that actually took the time)", {
    Cell["The model was the easy part. Making unattended spot-GPU training \
reliable, and getting the Wolfram kernel to talk to the cloud correctly, was \
most of the work. Four lessons stood out.", "Text"],
    Cell["Credentials inside the container. Every cloud job failed at first \
with \"no AWS credentials\", even though the instance role was correct. The \
cause was the instance-metadata hop limit: from inside a container the \
metadata service is one network hop further away, and the default limit of 1 \
silently blocks it. Raising it to 2 was the single change that let the very \
first job run.", "Text"],
    Cell["Logits versus probabilities. CrossEntropyLossLayer[\"Index\"] \
expects a probability vector, but the model's output layer emits raw logits. \
Feeding logits straight in makes the loss run unbounded-negative (the network \
is rewarded for ever-larger logits). The fix is a SoftmaxLayer between the \
model and the loss for training; the saved checkpoint keeps the bare logit \
head so sampling is unaffected. The diagnostic was the loss diving below zero \
instead of toward ln(vocab).", "Text"],
    Cell["No command-line tools in the image. The Wolfram engine container has \
no aws CLI, so all cloud storage I/O is done natively with \
ServiceExecute[\[Ellipsis], \"GetObject\"/\"PutObject\"]. Binary .wlnet \
checkpoints round-trip through this path byte-for-byte, which is what makes the \
resume-from-checkpoint mechanism trustworthy.", "Text"],
    Cell["Checkpoint between shards, never inside training. Saving from within \
the training callback leaks memory (each save copies the network); on a long \
run that ends in an out-of-memory crash near the finish. Checkpointing only at \
shard boundaries, and dropping the references in between, keeps memory flat. \
The full code, costs and transcripts live in the part-09-from-scratch folder \
and in PART9_RESULTS.md.", "Text"]
  }]
};

(* Renumber the closing chapter's title to "12.  <title>" whatever its current
   number.  Matches a Chapter cell whose text contains part9ClosingTitle. *)
part9RenumberClosing[cells_List] := cells /.
  Cell[t_String /; StringContainsQ[t, part9ClosingTitle], "Chapter", o___] :>
    Cell["12.  " <> part9ClosingTitle, "Chapter", o];

(* withPart9: strip any prior insert, renumber the closing chapter to 12,
   insert the from-scratch chapter (Chapter 11) right before it, and insert
   the flowchart before the first chapter.  If the closing chapter is not
   found, append the from-scratch chapter at the end (fallback). *)
withPart9[cells_List] := Module[{stripped, renum, closeIdx, withChap, firstChap},
  (* strip any previously-inserted Part-9 cells, matching ANY tag that starts
     with "PART9" (covers older markers too, so re-running never duplicates). *)
  stripped = DeleteCases[cells,
    c_Cell /; ! FreeQ[c, s_String /; StringStartsQ[s, "PART9"]]];
  renum = part9RenumberClosing[stripped];
  closeIdx = FirstPosition[renum,
    Cell["12.  " <> part9ClosingTitle, "Chapter", ___], Missing[]][[1]];
  withChap = If[MissingQ[closeIdx],
    Join[renum, part9ChapterCells],
    Join[Take[renum, closeIdx - 1], part9ChapterCells, Drop[renum, closeIdx - 1]]];
  firstChap = FirstPosition[withChap, Cell[_, "Chapter", ___],
    {Length[withChap] + 1}][[1]];
  Join[Take[withChap, firstChap - 1], part9FlowCells, Drop[withChap, firstChap - 1]]
];

insertPart9File[file_String] := Module[{nb, cells, out},
  If[! FileExistsQ[file], Print["insertPart9File: missing ", file]; Return[$Failed]];
  CopyFile[file, file <> ".bak", OverwriteTarget -> True];
  nb = Import[file];
  If[Head[nb] =!= Notebook, Print["insertPart9File: not a Notebook ", file];
    Return[$Failed]];
  cells = First[nb];
  out = Notebook[withPart9[cells], Sequence @@ Rest[nb]];
  Export[file, out, "NB"];
  Print["insertPart9File: ", file, "  ", Length[cells], " -> ",
    Length[withPart9[cells]], " top-level cells"];
  file
];
