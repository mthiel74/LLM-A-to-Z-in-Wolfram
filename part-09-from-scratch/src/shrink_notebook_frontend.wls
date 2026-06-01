#!/usr/bin/env wolframscript
(* Shrink a notebook's embedded raster figures THROUGH THE FRONT END so the
   images stay valid (no red boxes) and get re-compressed on save.

   Method: open in a hidden front end, NotebookGet the expression, replace each
   large RasterBox's pixel data with a resized Image wrapped so the front end
   re-serialises it as CompressedData, NotebookPut it back, NotebookSave.
   Saving via the front end (not kernel Export) is what keeps the images intact.

   Usage:  wolframscript -file shrink_notebook_frontend.wls file.nb [minMaxDim] [scale]
*)

file = $ScriptCommandLine[[2]];
minMaxDim = If[Length[$ScriptCommandLine] >= 3, ToExpression[$ScriptCommandLine[[3]]], 200];
scale = If[Length[$ScriptCommandLine] >= 4, ToExpression[$ScriptCommandLine[[4]]], 0.6];
If[! FileExistsQ[file], Print["missing: ", file]; Quit[1]];

n = 0;
(* turn a RasterBox into a resized image's boxes (GraphicsBox via ToBoxes of a
   compressed Image), so the front end stores it compactly. Decide on
   ImageDimensions, not ByteCount. *)
resizeBox[box_RasterBox] := Module[{im, sm},
  im = Quiet @ Check[Image[box[[1]]], $Failed];
  If[Head[im] =!= Image || Max[ImageDimensions[im]] <= minMaxDim, Return[box]];
  sm = ImageResize[im, Scaled[scale]];
  n = n + 1;
  (* ToBoxes of an Image gives a GraphicsBox[TagBox[RasterBox[...]]] that the FE
     compresses on save *)
  ToBoxes[sm]
];

res = UsingFrontEnd[
  Module[{nb, expr, pos, boxes, newExpr},
    nb = NotebookOpen[file, Visible -> False];
    expr = NotebookGet[nb];
    pos = Position[expr, _RasterBox];
    boxes = Extract[expr, pos];
    newExpr = ReplacePart[expr, Thread[pos -> (resizeBox /@ boxes)]];
    NotebookPut[newExpr, nb];
    NotebookSave[nb];
    NotebookClose[nb];
    "ok"]];

Print["result:  ", res];
Print["resized: ", n];
Print["size MB: ", NumberForm[FileByteCount[file]/1048576., {6, 2}]];
