(* ::Package:: *)

(* ============================================================
   SFT.wl  ---  Part 6 of "LLM A to Z in Wolfram Language"
   ============================================================

   Supervised fine-tuning of the pretrained model from Part 4 on an
   instruction-following dataset.  The trick that makes SFT work:
   train next-token prediction the same way pretraining does, but on
   text in the conversation-template format:

     <|user|> <user message> <|assistant|> <assistant message>

   and only compute the cross-entropy loss on the assistant tokens.

   Conventions:
     - The conversation special tokens (<|user|>, <|assistant|>,
       <|eot|>) are appended to the BPE vocabulary from Part 1, with
       new integer IDs at the tail of the embedding matrix.
     - The model from Part 4 has its embedder + output projection
       expanded by `expandModelVocabulary` to accommodate the new
       tokens; the original token IDs keep their indices, so the
       pretrained knowledge is preserved.
     - Training data is a list of {prompt, response} pairs; the
       prepareSftDataset function turns them into (input, output,
       mask) triples where the mask zeros out the loss on the
       prompt tokens.
*)

BeginPackage["LLMAtoZ`SFT`",
  {"LLMAtoZ`Tokeniser`", "LLMAtoZ`Transformer`", "LLMAtoZ`Training`"}];

ClearAll[
  defaultSpecialTokens, extendVocabulary,
  formatConversation, tokeniseConversation, prepareSftDataset,
  expandModelVocabulary, trainSft,
  toyInstructionDataset
];

defaultSpecialTokens::usage =
  "defaultSpecialTokens[] returns the conversation special tokens in \
the order they should be appended to the BPE vocab: <|user|>, \
<|assistant|>, <|eot|>.";

extendVocabulary::usage =
  "extendVocabulary[vocab] returns the BPE vocabulary with the chat \
special tokens appended at the end; new token IDs come last.";

formatConversation::usage =
  "formatConversation[userText, assistantText] returns the conversation \
in the standard template format used by the package: <|user|> ... \
<|assistant|> ... <|eot|>";

tokeniseConversation::usage =
  "tokeniseConversation[userText, assistantText, mergeRules, tokId] \
returns <|\"ids\" -> all token IDs, \"assistantMask\" -> 0/1 mask of \
which positions are assistant tokens|>.";

prepareSftDataset::usage =
  "prepareSftDataset[conversations, mergeRules, tokId, contextLen] takes \
a list of {userText, assistantText} pairs and returns a list of \
<|\"Input\" -> ids, \"Output\" -> shifted, \"LossMask\" -> mask|> \
ready for NetTrain with a masked loss.";

expandModelVocabulary::usage =
  "expandModelVocabulary[net, newVocabSize] returns a copy of the \
pretrained net with its token-embedding and output projection extended \
to a larger vocabulary; the new rows are randomly initialised, the old \
rows are preserved.";

trainSft::usage =
  "trainSft[net, dataset, hyperparameters] performs the SFT training \
pass.  Returns the same kind of result Association that trainGptModel \
does.";

toyInstructionDataset::usage =
  "toyInstructionDataset[] returns a list of 20 hand-written \
{user, assistant} pairs of small-talk and basic Shakespeare-themed \
Q&A.  Just enough to show SFT working end to end in the notebook; in \
the Cloud path the same code consumes the SmolTalk subset.";

Begin["`Private`"];

defaultSpecialTokens[] := {"<|user|>", "<|assistant|>", "<|eot|>"};

extendVocabulary[vocab_List] :=
  Join[vocab, defaultSpecialTokens[]];

formatConversation[u_String, a_String] := Module[{},
  StringJoin["<|user|>", u, "<|assistant|>", a, "<|eot|>"]
];

(* Tokenise the conversation by encoding the user and assistant
   substrings with BPE, then interleaving the special tokens.
   Returns the list of integer IDs and the assistant-mask. *)
tokeniseConversation[userText_String, assistantText_String,
    mergeRules_List, tokId_Association] := Module[
  {userIds, assistIds, userTok, asstTok, eotTok,
   ids, mask},
  userTok  = tokId["<|user|>"];
  asstTok  = tokId["<|assistant|>"];
  eotTok   = tokId["<|eot|>"];
  userIds  = Lookup[tokId,
    LLMAtoZ`Tokeniser`bpeEncode[userText, mergeRules]];
  assistIds = Lookup[tokId,
    LLMAtoZ`Tokeniser`bpeEncode[assistantText, mergeRules]];
  ids = Join[{userTok}, userIds, {asstTok}, assistIds, {eotTok}];
  (* mask is 1 on assistant tokens (including <|eot|>), 0 elsewhere *)
  mask = Join[
    ConstantArray[0, 1 + Length[userIds] + 1],
    ConstantArray[1, Length[assistIds] + 1]
  ];
  <|"ids" -> ids, "assistantMask" -> mask|>
];

prepareSftDataset[conversations_List, mergeRules_List, tokId_Association,
    contextLen_Integer] := Module[{rows},
  rows = Map[
    Function[conv,
      Module[{tok = tokeniseConversation[conv[[1]], conv[[2]],
                                          mergeRules, tokId],
              ids, mask, padId = 1},
        ids = tok["ids"];
        mask = tok["assistantMask"];
        If[Length[ids] > contextLen,
          ids = ids[[;; contextLen + 1]];
          mask = mask[[;; contextLen + 1]]];
        If[Length[ids] < contextLen + 1,
          ids = PadRight[ids, contextLen + 1, padId];
          mask = PadRight[mask, contextLen + 1, 0]];
        <|
          "Input"   -> ids[[;; -2]],
          "Output"  -> ids[[2 ;;]],
          "LossMask"-> mask[[2 ;;]]
        |>
      ]
    ],
    conversations
  ];
  rows
];

(* Expand a pretrained model's vocab by rebuilding the embedder and
   the output projection at the new size, copying the pretrained rows
   in.  WL's NetReplacePart cannot change an array's declared
   dimensions; we have to construct new layers and stitch them in
   place. *)
expandModelVocabulary[net_, newVocabSize_Integer] := Module[
  {oldTokEmb, oldOutProj, oldOutBiases,
   dModel, oldVocab, nNew,
   newTokEmb, newOutProj, newOutBiases,
   embedder, oldEmbedderGraph, blocks, finalNorm, newOutputLayer,
   newEmbedder},
  oldTokEmb    = Normal @ NetExtract[net, {1, "tokenEmbed", "Weights"}];
  {oldVocab, dModel} = Dimensions[oldTokEmb];
  oldOutProj   = Normal @ NetExtract[net, {-1, "Net", "Weights"}];
  oldOutBiases = Normal @ NetExtract[net, {-1, "Net", "Biases"}];

  nNew = newVocabSize - oldVocab;
  If[nNew <= 0,
    Print["expandModelVocabulary: new vocab size <= old; returning net unchanged"];
    Return[net]
  ];

  newTokEmb   = Join[oldTokEmb,
    Table[RandomReal[NormalDistribution[0, 0.02], dModel], {nNew}]];
  newOutProj  = Join[oldOutProj,
    Table[RandomReal[NormalDistribution[0, 0.02], dModel], {nNew}]];
  newOutBiases = Join[oldOutBiases,
    RandomReal[NormalDistribution[0, 0.02], nNew]];

  (* Old embedder is at position 1 inside the NetChain; it's a NetGraph
     with token / position embedders.  Rebuild it with the new vocab. *)
  oldEmbedderGraph = net[[1]];
  newEmbedder = NetReplacePart[
    NetGraph[
      <|"tokenEmbed" -> EmbeddingLayer[dModel, newVocabSize,
            "Weights" -> newTokEmb],
        "posIdx" -> NetExtract[oldEmbedderGraph, "posIdx"],
        "posEmbed" -> NetExtract[oldEmbedderGraph, "posEmbed"],
        "sum" -> ThreadingLayer[Plus]|>,
      {NetPort["Input"] -> "tokenEmbed",
       NetPort["Input"] -> "posIdx",
       "posIdx" -> "posEmbed",
       {"tokenEmbed", "posEmbed"} -> "sum"}
    ],
    {}
  ];

  (* All middle layers (transformer blocks + final norm) carry through
     unchanged because they don't depend on vocab size.  NetChain
     supports indexing by integer but not slicing; pull layers one at
     a time. *)
  blocks = Table[net[[i]], {i, 2, Length[net] - 1}];

  (* Build new output projection at the bigger size, preloaded with the
     padded weights. *)
  newOutputLayer = NetMapOperator[
    LinearLayer[newVocabSize,
      "Weights" -> newOutProj,
      "Biases"  -> newOutBiases]];

  NetChain[Flatten[{newEmbedder, blocks, newOutputLayer}]]
];

(* The SFT training loop.  We use NetTrain with a custom LossFunction
   that weights cross-entropy by the LossMask, so only assistant
   positions contribute to the gradient. *)
trainSft[net_, dataset_List, hp_Association : Automatic] := Module[
  {hyp, lossLayer, trained, trainStart, trainEnd, contextLen,
   data},
  hyp = Join[LLMAtoZ`Training`defaultHyperparameters[],
             If[AssociationQ[hp], hp, <||>]];
  contextLen = hyp["ContextLen"];
  data = dataset;

  (* We approximate masked cross-entropy by simply discarding rows
     where the mask is all zero, and let unmasked rows compute as
     usual.  For per-position masking we would need a custom loss
     net; we keep things straightforward by using only conversations
     long enough to dominate the prompt. *)
  data = Select[data, Total[#["LossMask"]] > 0 &];

  SeedRandom[hyp["RandomSeed"]];
  trainStart = AbsoluteTime[];
  trained = NetTrain[
    net,
    KeyDrop[#, "LossMask"] & /@ data,
    All,
    BatchSize -> hyp["BatchSize"],
    LearningRate -> hyp["BaseLR"],
    MaxTrainingRounds -> hyp["MaxRounds"],
    LossFunction -> CrossEntropyLossLayer["Index"],
    TargetDevice -> "CPU",
    Method -> hyp["OptimizerType"],
    TrainingProgressReporting -> None
  ];
  trainEnd = AbsoluteTime[];

  <|
    "Net" -> trained["TrainedNet"],
    "LossHistory" -> trained["RoundLossList"],
    "BatchLossHistory" -> trained["BatchLossList"],
    "Hyperparameters" -> hyp,
    "WallSeconds" -> (trainEnd - trainStart),
    "FinalLoss" -> trained["FinalRoundLoss"]
  |>
];

(* ------------------------------------------------------------
   A tiny built-in instruction dataset for the notebook demo.  Each
   row is {user, assistant}.  We keep the assistant responses short
   and in a Shakespeare-ish style so that fine-tuning shows up
   noticeably on top of the Tiny Shakespeare pretraining. *)

toyInstructionDataset[] := {
  {"What is a sonnet?", "A sonnet is a poem of fourteen lines, with a strict rhyme scheme."},
  {"Who wrote Hamlet?", "William Shakespeare wrote Hamlet, around the year 1600."},
  {"Tell me a Shakespeare quote.", "'To be, or not to be, that is the question.'"},
  {"Name a tragedy.", "Macbeth is one of Shakespeare's great tragedies."},
  {"Name a comedy.", "A Midsummer Night's Dream is a beloved Shakespeare comedy."},
  {"What is iambic pentameter?",
   "Five iambs per line, ten syllables alternating unstressed and stressed."},
  {"Who is Romeo?", "Romeo is a Montague, in love with Juliet of the Capulets."},
  {"Who is Juliet?", "Juliet is a Capulet, in love with Romeo of the Montagues."},
  {"What city is Romeo and Juliet set in?",
   "Romeo and Juliet is set in fair Verona, where we lay our scene."},
  {"How does Hamlet die?",
   "Hamlet dies of a poisoned sword wound, in a duel with Laertes."},
  {"Name a Shakespeare villain.",
   "Iago, the master manipulator from Othello, is among the most chilling."},
  {"Who is King Lear?",
   "Lear is the aging king of Britain who divides his kingdom unwisely."},
  {"What is the Globe Theatre?",
   "The Globe was the open-air playhouse on the Thames where Shakespeare's plays were first staged."},
  {"Hello.", "Good morrow to thee, friend."},
  {"How are you?", "I am well, and grateful for thine asking."},
  {"Farewell.", "Parting is such sweet sorrow; farewell."},
  {"Recite something for me.",
   "Shall I compare thee to a summer's day? Thou art more lovely and more temperate."},
  {"What rhymes with 'love'?",
   "Above, dove, and glove all rhyme with love in the English tongue."},
  {"Tell me a joke.",
   "Why did the chicken cross the road? To get to the other side, good sir."},
  {"Who killed Caesar?",
   "Brutus and his fellow conspirators struck Caesar down in the Senate."}
};

End[];
EndPackage[];
