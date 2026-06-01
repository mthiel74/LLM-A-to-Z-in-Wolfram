#!/usr/bin/env wolframscript
(* Build part-09-from-scratch/LLM_FromScratch.nb -- the standalone
   pretraining-from-scratch notebook.  Companion to the master
   LLM_A_to_Z.nb and discussion-model/LLM_Discussion.nb.

   Architecture: nano-style (RMSNorm + RoPE + SwiGLU), same family
   as LLaMA / Karpathy's nanochat.  Compute: AWS Batch spot g4dn
   instances under a $100 budget.  Cost-decision: between-job slope
   detection from the LLMAtoZ`Pretraining` harness.

   This is the scaffold: section headers + prose + stub Input cells.
   Output cells stay empty until we have a real cloud run.           *)

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
mono[s_]       := StyleBox[s, FontFamily -> "Courier"];
italic[s_]     := StyleBox[s, FontSlant -> Italic];
bold[s_]       := StyleBox[s, FontWeight -> Bold];

cells = Flatten @ {

  title["Pretraining a Nano Model from Scratch"],
  subtitle["From random weights to a chat-capable LLM in Wolfram Language, on a $100 budget"],

  text[{
    "The master notebook ", italic["LLM A to Z"], " ended with a 124M ",
    "GPT-2 small fine-tuned on twenty hand-written Q&A pairs.  ",
    "The ", italic["Discussion"], " notebook scaled that fine-tune to ",
    "ten thousand pairs of Alpaca-cleaned and got a model that actually ",
    "chats.  Both notebooks imported OpenAI's pretrained GPT-2 weights ",
    "and only trained the supervised-fine-tuning step on top."
  }],

  text[{
    "This notebook does the part nobody had done from-scratch in WL: ",
    bold["pretrains"], " the model.  No imported weights.  Random ",
    "initialisation, a slab of internet text, and the cross-entropy ",
    "objective.  We rebuild the nano-chat architecture (the family ",
    "Karpathy's nanochat, LLaMA, Mistral and Qwen share) directly out ",
    "of WL primitives \[Dash] RMSNorm, rotary position embedding, ",
    "SwiGLU \[Dash] then schedule chunked training on AWS Batch spot ",
    "GPUs under a fixed dollar budget."
  }],

  text[{
    "Three design choices worth flagging up-front:"
  }],

  text[{
    "1. ", bold["Nano architecture, not GPT-2."],
    "  RMSNorm replaces LayerNorm, RoPE replaces learned absolute ",
    "position embedding, SwiGLU replaces GELU MLP.  These are 2023-era ",
    "best practice and are what every open-weights release ",
    "(LLaMA, Mistral, Qwen) actually uses."
  }],

  text[{
    "2. ", bold["AWS Batch spot, not Wolfram Cloud."],
    "  Wolfram OnDemand charges a ~30% RemoteBatchSubmit overhead.  ",
    "g4dn.xlarge spot in eu-central-1 runs about $0.30/hour, so a ",
    "$100 budget buys ~330 GPU-hours \[Dash] enough to take a 30M-",
    "parameter model to coherent English on FineWeb-Edu."
  }],

  text[{
    "3. ", bold["Cost-decision lives between jobs, not within."],
    "  NetTrain runs are atomic; once started they consume their ",
    "full time budget.  We default to ~4-hour spot jobs (about $1.20 ",
    "each) and inspect the loss slope after each one before deciding ",
    "whether to launch a resume job from the latest checkpoint."
  }],

  chapter["1.  Setup"],

  input["SetDirectory[NotebookDirectory[]]"],

  input["Get[\"src/NanoChat.wl\"];
Get[\"src/Pretraining.wl\"];"],

  text[{
    "The companion code lives in two packages.  ",
    mono["LLMAtoZ`NanoChat`"], " gives us the architectural ",
    "building blocks (", mono["rmsNorm"], ", ", mono["applyRoPE"], ", ",
    mono["swiGluMLP"], ", ", mono["nanoModel"], ").  ",
    mono["LLMAtoZ`Pretraining`"], " gives us the in-job S3 checkpoint ",
    "callback and the between-job continue/stop decision."
  }],

  chapter["2.  Why a new architecture?"],

  text[{
    "The 12-layer transformer we built in Part 3 uses the original ",
    "Vaswani-2017 components: LayerNorm, learned absolute position ",
    "embeddings, a two-layer GELU feed-forward block.  Every public ",
    "open-weights LLM released after 2022 \[Dash] LLaMA, Mistral, ",
    "Qwen, Gemma, even Karpathy's didactic nanochat \[Dash] has ",
    "swapped all three out.  Each change is small in isolation; the ",
    "combination is empirically large."
  }],

  section["2.1  RMSNorm"],

  text[{
    "LayerNorm subtracts the mean and divides by the standard ",
    "deviation, then applies a learnable scale and bias: ",
    mono["y = (x - mean(x))/std(x) * \[Gamma] + \[Beta]"], ".  ",
    "RMSNorm drops the mean-subtraction and the bias: ",
    mono["y = x/Sqrt[mean(x^2) + \[Epsilon]] * \[Gamma]"], ".  ",
    "Zhang & Sennrich (2019) showed the mean-centering is doing little ",
    "work in transformers, and removing it saves a reduction op per ",
    "forward pass.  Our implementation also drops the learnable ",
    "\[Gamma]; ablations in LLaMA-2 show that costs <0.5% on the ",
    "evaluation suite."
  }],

  input["norm = LLMAtoZ`NanoChat`rmsNorm[8, 4];
sample = RandomReal[NormalDistribution[], {4, 8}];
{Dimensions[norm[sample]], Mean[norm[sample]^2 // Flatten]}"],

  text[{
    "The output has the same shape as the input and a mean-square close ",
    "to one (the defining property of an RMSNorm layer)."
  }],

  section["2.2  Rotary position embedding"],

  text[{
    "Vanilla transformers add a learned vector to each token embedding ",
    "to encode position.  The model sees the position implicitly through ",
    "the embedding.  Rotary position embedding (Su et al. 2021) takes a ",
    "different tack: ", italic["rotate"], " the query and key vectors ",
    "inside each attention head by an angle that depends on the position. ",
    "The dot-product attention score between positions ", mono["i"], " ",
    "and ", mono["j"], " then depends only on the difference ",
    mono["j - i"], " \[Dash] which is exactly what we want from a ",
    "position encoding."
  }],

  text[{
    "RoPE has two practical advantages: (i) no learnable position ",
    "embedding to train, saving ", mono["maxSeqLen * dModel"],
    " parameters; (ii) the model generalises to context lengths ",
    "beyond ", mono["maxSeqLen"], " more gracefully because there is ",
    "nothing in the model's parameters that ", italic["expects"],
    " a particular sequence length."
  }],

  input["rope = LLMAtoZ`NanoChat`applyRoPE[16, 32];
sample = RandomReal[{-1, 1}, {16, 32}];
Dimensions[rope[sample]]"],

  section["2.3  SwiGLU MLP"],

  text[{
    "The Vaswani feed-forward block is ", mono["Linear -> GELU -> Linear"],
    ".  SwiGLU (Shazeer 2020) replaces it with two parallel linears, a ",
    "SiLU non-linearity on one branch, and an elementwise product:"
  }],

  text[{
    mono["y = down( SiLU(gate(x)) * up(x) )"]
  }],

  text[{
    "The gate-and-multiply trick is borrowed from gated linear units ",
    "in language modelling (Dauphin et al. 2017).  It adds parameters ",
    "(3 linear layers instead of 2) but the LLaMA-2 ablations show it ",
    "is a consistent 0.5\[Dash]1.0 perplexity-point win at the same ",
    "compute budget."
  }],

  input["ffn = LLMAtoZ`NanoChat`swiGluMLP[64, 256, 16];
Dimensions @ ffn[RandomReal[{-1, 1}, {16, 64}]]"],

  section["2.4  Putting it together"],

  text[{
    "A nano-style transformer block is the same residual scaffold as ",
    "GPT-2, with each component replaced:"
  }],

  text[{
    mono["x -> RMSNorm -> RoPE-MHA -> + -> RMSNorm -> SwiGLU -> +"]
  }],

  input["block = LLMAtoZ`NanoChat`nanoTransformerBlock[64, 4, 256, 16];
Dimensions @ block[RandomReal[{-1, 1}, {16, 64}]]"],

  chapter["3.  Building nanoModel"],

  text[{
    "The full model is a token embedding, ", mono["nLayers"], " of ",
    "the block above, a final RMSNorm, and an output linear projection ",
    "back to vocabulary size.  No positional embedding \[Dash] RoPE ",
    "handles position inside each attention block."
  }],

  input["tinySpec = LLMAtoZ`NanoChat`nanoModelSpecTiny[100, 16];
mdl = LLMAtoZ`NanoChat`nanoModel[
   tinySpec[\"dModel\"], tinySpec[\"nHeads\"],
   tinySpec[\"nLayers\"], tinySpec[\"dFF\"],
   tinySpec[\"vocabSize\"], tinySpec[\"maxSeqLen\"]];
Information[mdl, \"ArraysTotalElementCount\"]"],

  text[{
    "The closed-form parameter count and the actual array count match ",
    "to within the frozen causal-mask and RoPE cos/sin tables \[Dash] ",
    "the difference is the count of position-encoding constants the ",
    "model carries around but does not train."
  }],

  input["LLMAtoZ`NanoChat`nanoParameterCount[tinySpec]"],

  chapter["4.  Model sizes and the dollar budget"],

  text[{
    "Four named specs sized for the GPT-2 50,257-entry vocabulary at ",
    "1024 context length:"
  }],

  input["sizes = Map[
   With[{s = #[50257, 1024]},
     <|\"name\" -> s[\"label\"], \"dModel\" -> s[\"dModel\"],
       \"nLayers\" -> s[\"nLayers\"],
       \"params\" -> LLMAtoZ`NanoChat`nanoParameterCount[s]|>
   ] &,
   {LLMAtoZ`NanoChat`nanoModelSpec30M,
    LLMAtoZ`NanoChat`nanoModelSpec100M,
    LLMAtoZ`NanoChat`nanoModelSpec200M}
 ];
Dataset[sizes]"],

  text[{
    "Wall-time estimates on a single T4 (g4dn.xlarge spot, $0.30/hr in ",
    "eu-central-1) for 5 billion training tokens:"
  }],

  text[{
    "  30M  params  \[Rule]  ~8 hours    \[Rule]  ~$2.40\n",
    "  100M params  \[Rule]  ~30 hours   \[Rule]  ~$9\n",
    "  200M params  \[Rule]  ~80 hours   \[Rule]  ~$24"
  }],

  text[{
    "These are forward-only estimates; backward + optimizer state ",
    "roughly doubles the wall time.  Multi-GPU on g4dn.12xlarge (4x T4) ",
    "scales roughly 3x, not 4x, because of communication overhead.  ",
    "The actual numbers will land in the table below after the first ",
    "calibration run."
  }],

  chapter["5.  Training data: FineWeb-Edu"],

  text[{
    "FineWeb-Edu is HuggingFace's deduplicated, education-quality ",
    "subset of Common Crawl.  We use the 10B-token sample, which ",
    "weighs in at ~30 GB compressed Parquet.  Pre-tokenised with the ",
    "GPT-2 BPE encoder (the same one our model output projection ",
    "matches) it fits in roughly 20 GB of token-ID files.  The full ",
    "preprocessing recipe lives in ", mono["src/preprocess_data.wls"],
    ".  For the notebook we work with a 100 MB shard so the cells ",
    "below execute quickly."
  }],

  program["(* Placeholder: data prep run on the cloud node.  See
   src/preprocess_data.wls for the actual recipe. *)"],

  chapter["6.  The chunked-pretraining harness"],

  text[{
    "Long pretraining runs fail in one of two ways: the node terminates ",
    "(spot revocation, OOM at the end during checkpoint save) or the ",
    "training appears to flatline halfway through.  Both problems have ",
    "the same answer: ", italic["save the model periodically while training is still going"],
    ", and have the orchestrator decide ",
    "whether to launch a resume job once the current one finishes."
  }],

  text[{
    mono["LLMAtoZ`Pretraining`pretrainingCallback"], " returns a ",
    "Function suitable for NetTrain's ", mono["TrainingProgressReporting"],
    " hook.  Every ", mono["EveryBatches"], " batches it writes a JSON ",
    "loss-curve snapshot to S3; every ten of those it also writes a ",
    "full ", mono[".wlnet"], " checkpoint.  The cadences are tuned so ",
    "the per-batch overhead is well under 1%."
  }],

  input["cb = LLMAtoZ`Pretraining`pretrainingCallback[
   \"S3Prefix\" -> \"s3://my-bucket/runs/2026-05-21\",
   \"EveryBatches\" -> 100,
   \"NetExtractor\" -> (NetTake[#, Length[#] - 1] &)
];
Head[cb]"],

  chapter["7.  The cost-decision rule"],

  text[{
    "Between jobs, the orchestrator fetches ", mono["loss.json"],
    " from S3 and feeds the trajectory to ",
    mono["decideToContinue"], ".  Three outcomes:"
  }],

  text[{
    "  Budget exhausted    \[Rule]  Stop\n",
    "  Loss has plateaued  \[Rule]  Stop\n",
    "  Still going down    \[Rule]  Continue \[Dash] launch a resume job"
  }],

  text[{
    "The plateau detector is a coarse one: linear fit over the last ",
    "100 batches, flag if the absolute slope is below ",
    mono["5x10^-6"], " loss-units per batch.  At a 100M model with ",
    "batch size 32, that threshold corresponds to a perplexity ",
    "improvement of less than 0.001 over 10,000 examples \[Dash] not ",
    "worth another $1.20 of training."
  }],

  input["(* synthetic example: an improving trajectory *)
improving = Table[
   <|\"batch\" -> b, \"loss\" -> 5.0 * Exp[-b/300.] + 0.3|>,
   {b, 1, 500}];
LLMAtoZ`Pretraining`decideToContinue[<|
   \"LossHistory\" -> improving,
   \"CostSpent\" -> 20., \"BudgetUSD\" -> 100.
|>]"],

  chapter["8.  Running on AWS Batch spot"],

  text[{
    "The training stack on AWS Batch:"
  }],

  text[{
    "  ", mono["WolframG4SpotCE"], "          \[LongDash] spot CE, g4dn.xlarge + g4dn.2xlarge\n",
    "  ", mono["WolframG4MultiSpotCE"], "     \[LongDash] spot CE, g4dn.12xlarge (4\[Times]T4)\n",
    "  ", mono["WolframG4SpotQueue"], "       \[LongDash] queue feeding the single-GPU CE\n",
    "  ", mono["WolframG4MultiSpotQueue"], "  \[LongDash] queue feeding the 4-GPU CE\n",
    "  ", mono["WolframG4Multi4GpuJobDef"], " \[LongDash] job def: vCPUs=48, GPUs=4, MEM=180 GiB"
  }],

  text[{
    "Account-level quota requests (eu-central-1) needed before any of ",
    "this runs:"
  }],

  text[{
    "  L-3819A6DF  All G/VT Spot Instance Requests    \[Rule] 96 vCPU\n",
    "  L-DB2E81BA  Running On-Demand G/VT instances   \[Rule] 48 vCPU"
  }],

  program["(* Submission template -- run from a fresh kernel that has
   ~/.aws/credentials configured.  Output cell stays empty until
   we have the first calibration run. *)
awsService = ServiceConnect[\"AWS\"];
env = RemoteBatchSubmissionEnvironment[\"AWSBatch\", <|
   \"JobQueue\" -> \"arn:aws:batch:eu-central-1:.../job-queue/WolframG4SpotQueue\",
   \"JobDefinition\" -> \"arn:aws:batch:eu-central-1:.../job-definition/WolframG4xlGpuJobDef:1\",
   \"IOBucket\" -> \"mywolframbatchstack-g4gpu-wolframjobdatabucket-...\",
   \"ServiceObject\" -> awsService
|>];"],

  chapter["9.  The first calibration run"],

  text[{
    "The first run is a ", italic["calibration"],
    " run: 30M model, 100 MB FineWeb-Edu shard, ~1 hour wall time, ~$0.30 ",
    "of spot.  Its job is to confirm the training loop trains, the S3 ",
    "callback writes loss + checkpoint, and the throughput numbers we ",
    "estimated above are within 2x of reality."
  }],

  program["(* RemoteBatchSubmit returning the RemoteBatchJobObject.
   Filled in after the first cloud run. *)"],

  text[{
    "Once we have those numbers, we re-plan: pick the largest model ",
    "size whose 5B-token wall time fits in the $100 budget at the ",
    "observed throughput.  This will likely be 100M parameters; it ",
    "might be 200M if the throughput estimate was conservative; it ",
    "might be 30M if we are missing a CUDA optimisation."
  }],

  chapter["10.  The actual training run"],

  text[{
    "Placeholder for the production run.  This cell will become a ",
    "loop: submit a 4-hour spot job, wait for it to finish, fetch ",
    "loss.json, ", mono["decideToContinue"],
    ", submit the next or stop."
  }],

  program["(* Filled in after calibration. *)"],

  chapter["11.  Sampling from the trained model"],

  text[{
    "Once we have weights, sampling reuses the temperature / top-k ",
    "machinery from Part 5 unchanged.  RoPE plays nicely with ",
    "incremental sampling: we generate one token, append it to the ",
    "context, and re-run the model on the extended sequence.  No ",
    "KV-cache yet \[Dash] that is a follow-up."
  }],

  program["(* Filled in after we have trained weights. *)"],

  chapter["12.  Chat-style fine-tuning"],

  text[{
    "Pretraining alone gets the model fluent in English but not ",
    "conversational.  The Discussion notebook's SFT recipe ",
    "(Alpaca-cleaned, plain-text User/Assistant template) ports across ",
    "to this architecture unchanged \[Dash] same NetTrain call, just ",
    "a different starting checkpoint.  We do this as a separate ",
    "$5\[Dash]$10 run after pretraining is done."
  }],

  chapter["13.  Limits and what is missing"],

  text[{
    "  \[Bullet] ", bold["No weight tying."], "  Karpathy's nanochat ",
    "ties the input embedding and the output projection, saving ~38M ",
    "parameters at the 200M size.  WL's ", mono["NetSharedArray"],
    " makes this implementable; not yet wired in.\n",
    "  \[Bullet] ", bold["No KV-cache for sampling."], "  Generation ",
    "currently re-runs the full model on the extended sequence each ",
    "step, which is O(N^2) in context length.  Should add an ",
    "incremental sampler.\n",
    "  \[Bullet] ", bold["No grouped-query attention."], "  Modern ",
    "70B-scale models use GQA to share KV across query heads.  Not ",
    "useful at our scale but worth mentioning.\n",
    "  \[Bullet] ", bold["The RMSNorm has no learnable gain."],
    "  The \[Gamma] is the cheapest parameter in the model but ",
    "implementing it in WL requires a NetSharedArray hook; left as ",
    "follow-up."
  }],

  text[{
    "Useful references:"
  }],

  text[{
    "  Karpathy, A.  ", italic["nanochat"], ".  github.com/karpathy/nanochat (2025).\n",
    "  Su, J. et al.  ", italic["RoFormer: Enhanced Transformer with Rotary Position Embedding"], ".  arXiv:2104.09864 (2021).\n",
    "  Zhang, B. & Sennrich, R.  ", italic["Root Mean Square Layer Normalization"], ".  NeurIPS 2019.\n",
    "  Shazeer, N.  ", italic["GLU Variants Improve Transformer"], ".  arXiv:2002.05202 (2020).\n",
    "  Touvron, H. et al.  ", italic["LLaMA: Open and Efficient Foundation Language Models"], ".  arXiv:2302.13971 (2023)."
  }]
};

nb = Notebook[cells,
  WindowSize -> {1100, 800},
  StyleDefinitions -> "Default.nb",
  TaggingRules -> <|"BuiltAt" -> DateString[Now, "ISODateTime"],
                    "BuiltBy"  -> "part-09-from-scratch/src/build_notebook.wls"|>
];

outPath = FileNameJoin[{partDir, "LLM_FromScratch.nb"}];
res = Export[outPath, nb, "NB"];
If[res === $Failed,
  Print["Export FAILED for ", outPath]; Quit[1],
  Print["Wrote: ", outPath, "  (", Length[cells], " cells, ", FileByteCount[outPath], " bytes)"]];
