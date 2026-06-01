#!/usr/bin/env wolframscript
(* Insert the Part-9 flowchart + Chapter 11 into a published notebook USING THE
   FRONT END (NotebookWrite/NotebookSave), which preserves the existing cells'
   CompressedData images losslessly.

   DO NOT use kernel Import[nb] / Export[nb,"NB"] for this -- a kernel round-trip
   decompresses every embedded image (CompressedData -> raw), bloats the file,
   and makes the front end render the figures as red boxes.

   Idempotent: removes any prior Part-9 cells (CellTags starting "PART9") first.

   Usage:  wolframscript -file insert_part9_frontend.wls file.nb *)

file = $ScriptCommandLine[[2]];
If[! FileExistsQ[file], Print["missing: ", file]; Quit[1]];

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
Get[FileNameJoin[{repo, "part-09-from-scratch", "src", "part9_cells.wl"}]];
If[Head[flowchartGraphics] =!= Graphics, Print["flowchart not built"]; Quit[1]];

(* flatten closedSection wrappers into a flat cell list for NotebookWrite:
   each closedSection is Cell[CellGroupData[{header, body...}, Open]] -- the
   front end will regroup; we just write the inner cells in order. *)
flatten9[cells_] := Flatten[cells //. Cell[CellGroupData[inner_List, _], ___] :> inner];

flowFlat = flatten9[part9FlowCells];
chapFlat = flatten9[part9ChapterCells];

res = UsingFrontEnd[
  Module[{nb, found},
    nb = NotebookOpen[file, Visible -> False];

    (* 0. strip any previous Part-9 cells by tag *)
    NotebookFind[nb, "PART9", All, CellTags, AutoScroll -> False];
    If[MatchQ[SelectedCells[nb], {__CellObject}],
      NotebookDelete[nb]];

    (* 1. renumber the closing chapter 11 -> 12 *)
    found = NotebookFind[nb, "11.  Where to go from here", All,
      AutoScroll -> False];
    If[found =!= $Failed,
      NotebookWrite[nb, Cell["12.  Where to go from here", "Chapter"]]];

    (* 2. put Chapter 11 (from-scratch) right before the closing chapter.
       Re-find the (now "12.") closing chapter, move selection before it. *)
    NotebookFind[nb, "12.  Where to go from here", All, AutoScroll -> False];
    SelectionMove[nb, Before, Cell];
    NotebookWrite[nb, chapFlat];

    (* 3. flowchart before the first chapter (Tokenisation = chapter 1) *)
    NotebookFind[nb, "1.  Tokenisation", All, AutoScroll -> False];
    SelectionMove[nb, Before, Cell];
    NotebookWrite[nb, flowFlat];

    NotebookSave[nb];
    NotebookClose[nb];
    "ok"]];

Print["result: ", res];
