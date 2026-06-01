(* ::Package:: *)

(* ============================================================
   Chat.wl  ---  Part 8 of "LLM A to Z in Wolfram Language"
   ============================================================

   The interactive chat widget that closes the series.  Loads the
   final post-trained model from Part 7, formats the conversation
   in the same template the SFT pass used, streams a continuation
   token-by-token into a Dynamic-rendered transcript.

   The widget runs in the notebook front end out of the box.  The
   same chat function can also be deployed to the Wolfram Cloud
   via CloudDeploy[Delayed[chatWith[...]], Permissions -> "Public"]
   to give the model a permanent URL.
*)

BeginPackage["LLMAtoZ`Chat`",
  {"LLMAtoZ`Tokeniser`", "LLMAtoZ`Transformer`",
   "LLMAtoZ`Sampling`", "LLMAtoZ`SFT`"}];

ClearAll[
  chatTurn, streamChatTurn, chatWidget
];

chatTurn::usage =
  "chatTurn[net, transcript, userMessage, mergeRules, tokId, idTok, opts] \
formats the existing transcript plus the new user message in the SFT \
chat template, runs the model autoregressively until <|eot|> or the \
max-tokens limit is hit, and returns the updated transcript with the \
assistant response appended.";

streamChatTurn::usage =
  "streamChatTurn[net, transcript, userMessage, mergeRules, tokId, idTok, \
streamFn, opts] is the streaming variant: it calls streamFn[token] each \
time the model emits a token, suitable for live front-end rendering.";

chatWidget::usage =
  "chatWidget[net, mergeRules, tokId, idTok] returns a Dynamic \
front-end widget: an input field for the user's next message, a button \
to send, a button to reset, and the live transcript.  Drop it into a \
notebook cell.";

Begin["`Private`"];

Options[chatTurn] = Options[streamChatTurn] = Options[chatWidget] = {
  "MaxNew"      -> 80,
  "Temperature" -> 0.8,
  "TopK"        -> 40,
  "ContextLen"  -> 64
};

(* Format the running transcript into the SFT chat template.  Each
   transcript entry is an Association with role (user or assistant)
   and content (string).  We feed the model a flat token sequence:
   <|user|> u_1 <|assistant|> a_1 <|user|> u_2 <|assistant|> ... *)

renderTranscript[transcript_List, userMessage_String,
    mergeRules_List, tokId_Association] := Module[
  {parts, ids},
  parts = Map[
    Function[turn,
      Switch[turn["role"],
        "user",      "<|user|>" <> turn["content"],
        "assistant", "<|assistant|>" <> turn["content"] <> "<|eot|>"]
    ],
    transcript
  ];
  AppendTo[parts, "<|user|>" <> userMessage];
  AppendTo[parts, "<|assistant|>"];
  StringJoin[parts]
];

tokeniseChat[text_String, mergeRules_List, tokId_Association] := Module[
  {parts, ids, t},
  (* split by special tokens to keep them intact *)
  ids = {};
  parts = StringSplit[text,
    {"<|user|>" -> "<|user|>",
     "<|assistant|>" -> "<|assistant|>",
     "<|eot|>" -> "<|eot|>"}];
  Do[
    If[MemberQ[{"<|user|>", "<|assistant|>", "<|eot|>"}, t],
      AppendTo[ids, tokId[t]],
      ids = Join[ids,
        Lookup[tokId,
          LLMAtoZ`Tokeniser`bpeEncode[t, mergeRules]]]
    ],
    {t, parts}
  ];
  ids
];

chatTurn[net_, transcript_List, userMessage_String,
    mergeRules_List, tokId_Association, idTok_Association,
    opts : OptionsPattern[]] := Module[
  {prompt, promptIds, contextLen, eotId, maxNew, ids, generated,
   asstStr, T, topK, ctx, pos, logits, sampled},
  contextLen = OptionValue["ContextLen"];
  maxNew     = OptionValue["MaxNew"];
  T          = OptionValue["Temperature"];
  topK       = OptionValue["TopK"];
  eotId      = tokId["<|eot|>"];

  prompt = renderTranscript[transcript, userMessage, mergeRules, tokId];
  promptIds = tokeniseChat[prompt, mergeRules, tokId];

  ids = promptIds;
  generated = {};
  Do[
    ctx = If[Length[ids] >= contextLen, ids[[-contextLen ;;]],
              PadRight[ids, contextLen, 1]];
    pos = Min[Length[ids], contextLen];
    logits = net[ctx][[pos, All]];
    sampled = LLMAtoZ`Sampling`topKSample[logits, topK, T];
    AppendTo[ids, sampled];
    AppendTo[generated, sampled];
    If[sampled === eotId, Break[]],
    {maxNew}
  ];

  asstStr = StringReplace[
    StringJoin @ Lookup[idTok, DeleteCases[generated, eotId]],
    "</w>" -> " "];
  Join[transcript,
    {<|"role" -> "user",      "content" -> userMessage|>,
     <|"role" -> "assistant", "content" -> asstStr|>}]
];

streamChatTurn[net_, transcript_List, userMessage_String,
    mergeRules_List, tokId_Association, idTok_Association,
    streamFn_, opts : OptionsPattern[]] := Module[
  {prompt, promptIds, contextLen, eotId, maxNew, ids, generated,
   T, topK, ctx, pos, logits, sampled, asstStr},
  contextLen = OptionValue["ContextLen"];
  maxNew     = OptionValue["MaxNew"];
  T          = OptionValue["Temperature"];
  topK       = OptionValue["TopK"];
  eotId      = tokId["<|eot|>"];

  prompt = renderTranscript[transcript, userMessage, mergeRules, tokId];
  promptIds = tokeniseChat[prompt, mergeRules, tokId];
  ids = promptIds;
  generated = {};
  Do[
    ctx = If[Length[ids] >= contextLen, ids[[-contextLen ;;]],
              PadRight[ids, contextLen, 1]];
    pos = Min[Length[ids], contextLen];
    logits = net[ctx][[pos, All]];
    sampled = LLMAtoZ`Sampling`topKSample[logits, topK, T];
    AppendTo[ids, sampled];
    AppendTo[generated, sampled];
    streamFn[StringReplace[idTok[sampled], "</w>" -> " "]];
    If[sampled === eotId, Break[]],
    {maxNew}
  ];
  asstStr = StringReplace[
    StringJoin @ Lookup[idTok, DeleteCases[generated, eotId]],
    "</w>" -> " "];
  Join[transcript,
    {<|"role" -> "user",      "content" -> userMessage|>,
     <|"role" -> "assistant", "content" -> asstStr|>}]
];

(* The interactive widget. *)
chatWidget[net_, mergeRules_List, tokId_Association, idTok_Association,
    opts : OptionsPattern[]] := DynamicModule[
  {transcript = {}, userInput = "", busy = False, streamed = ""},
  Panel[
    Column[{
      Style["Tiny Shakespeare Chat", Bold, 18],
      Dynamic @ Pane[
        Column[
          Map[
            Function[turn,
              Style[
                Row[{Style[turn["role"] <> ": ", Bold,
                           If[turn["role"] === "user", Blue, Darker[Green]]],
                     turn["content"]}],
                "Text"]
            ],
            transcript
          ]
        ],
        {500, 300}, Scrollbars -> True],
      Row[{
        InputField[Dynamic[userInput], String,
          FieldSize -> {40, 1},
          FieldHint -> "Type a message and press Enter"],
        Spacer[8],
        Button["send",
          If[!busy && StringLength[userInput] > 0,
            busy = True;
            transcript = chatTurn[net, transcript, userInput,
                                  mergeRules, tokId, idTok, opts];
            userInput = "";
            busy = False],
          ImageSize -> 80],
        Spacer[4],
        Button["reset", transcript = {}; userInput = "",
          ImageSize -> 80]
      }],
      Dynamic[If[busy, ProgressIndicator[Appearance -> "Indeterminate"], ""]]
    }],
    Background -> RGBColor[0.98, 0.98, 0.96]
  ]
];

End[];
EndPackage[];
