#!/usr/bin/env wolframscript
(* build_evaluated_notebook.wls

   Opens LLM_A_to_Z.nb in a hidden Wolfram Front End session, evaluates
   the Input cells whose code is *safe* (no training loops, no 500 MB
   NetModel pulls, no cloud submissions), and saves the result as
   LLM_A_to_Z_evaluated.nb.

   Heavy cells (matching $skipPatterns below) are left as plain Input
   cells with no Output -- the reader can run them manually when
   they're willing to wait.

   The Front End handles graphics rendering, Manipulate
   serialization, and image compression natively, so we get textbook-
   style baked-in outputs without the ~200 MB bloat we had when we
   tried to do the same thing inline in the build kernel.            *)

SetDirectory[DirectoryName[$InputFileName]];

srcPath = FileNameJoin[{Directory[], "LLM_A_to_Z.nb"}];
outPath = FileNameJoin[{Directory[], "LLM_A_to_Z_evaluated.nb"}];

If[!FileExistsQ[srcPath],
  Print["source notebook not found: ", srcPath]; Quit[1]];

(* Cells whose code text contains any of these are skipped at
   evaluation time.  They're the ones that need network / a GPU / many
   minutes of CPU.                                                   *)
$skipPatterns = {
  "NetTrain[",                       (* training loops *)
  "loadGpt2Small",                   (* 498 MB NetModel pull *)
  "loadGpt2Medium",                  (* 1.4 GB NetModel pull *)
  "loadPrebuiltGpt2",                (* same *)
  "buildGpt2Architecture",           (* from-scratch GPT-2 build *)
  "RemoteBatchSubmit",               (* cloud submission *)
  "RemoteBatchJobObject",            (* cloud queries *)
  "trainGptModel",                   (* training entry *)
  "trainOvernight",                  (* training entry *)
  "trainSft",                        (* SFT training *)
  "runGrpoEpoch",                    (* RL post-training *)
  "discussion_medium_20k.wlnet",     (* large checkpoint import *)
  "discussion_large_10k.wlnet",
  "gpt2_sft_chat.wlnet",
  "gpt2_attention_all_layers.wxf",   (* big WXF (~50 MB) *)
  "tiny_gpt_trained.wlnet",
  "smallcpu_gpt_trained.wlnet",
  "charlevel_shakespeare_trained.wlnet",
  "submitRemotePretraining",
  "CloudConnect",
  "URLDownload",                     (* network *)
  "fetchAlpacaCleaned"               (* 44 MB download *)
};

shouldEvaluateQ[code_String] := !AnyTrue[$skipPatterns,
  StringContainsQ[code, #] &];

Print["opening source notebook ..."];

UsingFrontEnd @ Module[
  {nb, inputCells, total, skipped = 0, t0},

  nb = NotebookOpen[srcPath, Visible -> False];
  inputCells = Cells[nb, CellStyle -> "Input"];
  total = Length[inputCells];
  Print["found ", total, " Input cells; tagging the heavies"];

  (* First pass: mark cells we don't want NotebookEvaluate to touch. *)
  Do[
    Module[{cell = inputCells[[i]], boxes, codeStr},
      boxes = NotebookRead[cell];
      codeStr = Quiet @ Check[
        ToString[boxes[[1]] /. (Cell[s_, ___] :> s) /. BoxData[x_] :> x,
                  InputForm],
        ""];
      If[!shouldEvaluateQ[codeStr],
        SetOptions[cell, Evaluatable -> False];
        skipped += 1;
        Print["  [", i, "/", total, "] SKIP -- ",
              StringTake[codeStr, UpTo[60]]]]],
    {i, total}];

  Print["skipped ", skipped, " heavy cells; will NotebookEvaluate the ",
        total - skipped, " remaining"];
  t0 = AbsoluteTime[];

  (* Synchronous: walks evaluatable cells in order, inserts Output cells. *)
  NotebookEvaluate[nb, InsertResults -> True];

  Print["evaluation finished in ", Round[(AbsoluteTime[] - t0)/60., 0.1],
        " min"];
  Print["saving ", outPath];
  NotebookSave[nb, outPath];
  NotebookClose[nb]];

Print["done.  ", outPath, " is ",
  Round[FileByteCount[outPath] / 2^20, 0.1], " MB"];
