#!/usr/bin/env wolframscript
(* Downsample embedded raster figures in a notebook so the file stays under a
   size budget, keeping each figure's on-page size (only its pixel resolution
   drops).

   Notes learned the hard way:
     - .nb deflate-compresses internally, so JPEG re-encoding does nothing for
       file size; only fewer pixels help.
     - RasterBox[data, coordbox, range, opts]: data (arg 1) is the pixel array.
     - Decide whether to resize from ImageDimensions, then rebuild the box with
       the resized pixel data kept as a NumericArray.

   Usage:  wolframscript -file shrink_notebook.wls file.nb [minMaxDim] [scale]
*)

args = $ScriptCommandLine[[2 ;;]];
If[Length[args] < 1, Print["usage: shrink_notebook.wls file.nb [minMaxDim] [scale]"]; Quit[1]];
file = args[[1]];
minMaxDim = If[Length[args] >= 2, ToExpression[args[[2]]], 200];
scale = If[Length[args] >= 3, ToExpression[args[[3]]], 0.75];

If[! FileExistsQ[file], Print["missing: ", file]; Quit[1]];
nb = Import[file];
If[Head[nb] =!= Notebook, Print["not a Notebook: ", file]; Quit[1]];

before = FileByteCount[file];

(* Resize one box.  Decide from ImageDimensions (ByteCount of the extracted
   pixel array under-reports), keep the resized data as a NumericArray.  Takes
   the box as an argument (NOT a pattern in a ReplaceAll rule -- in-structure
   pattern replacement silently fails to match these boxes; Extract +
   ReplacePart by explicit Position is what actually works). *)
resize[box_RasterBox] := Module[{d, im, sm},
  d = box[[1]];
  im = Quiet @ Check[Image[d], $Failed];
  If[Head[im] =!= Image || Max[ImageDimensions[im]] <= minMaxDim, Return[box]];
  sm = ImageResize[im, Scaled[scale]];
  ReplacePart[box, 1 -> NumericArray[ImageData[sm, "Byte"], "UnsignedInteger8"]]
];

pos = Position[nb, _RasterBox];
boxes = Extract[nb, pos];
newBoxes = resize /@ boxes;
n = Count[Transpose[{boxes, newBoxes}], {a_, b_} /; a =!= b];
nb2 = ReplacePart[nb, Thread[pos -> newBoxes]];
Export[file, nb2, "NB"];

after = FileByteCount[file];
reimport = Import[file];
Print["file:        ", file];
Print["resized:     ", n, " of ",
  Length[Cases[nb, _RasterBox, Infinity]], " RasterBoxes (maxdim>", minMaxDim,
  ", x", scale, ")"];
Print["size:        ", NumberForm[before/1048576., {6, 2}], " MB -> ",
  NumberForm[after/1048576., {6, 2}], " MB"];
Print["under 20 MB: ", after/1048576. < 20];
Print["reimport ok: ", Head[reimport] === Notebook];
Print["rasterboxes: ", Length[Cases[reimport, _RasterBox, Infinity]]];
Print["data heads:  ", Tally[Cases[reimport, RasterBox[d_, ___] :> Head[d], Infinity]]];
