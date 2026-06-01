#!/usr/bin/env wolframscript
(* Patch existing published notebooks with the Part-9 flowchart + Chapter 12.

   Thin driver over part9_cells.wl (the single source of truth, also used by
   build_master_notebook.wls so a fresh build includes Part 9 natively).
   Idempotent: re-running replaces the inserted cells, never duplicates.

   Usage:  wolframscript -file insert_part9_section.wls [file1.nb file2.nb] *)

repo = FileNameJoin[{$HomeDirectory, "GitHub", "LLM-A-to-Z-in-WL"}];
Get[FileNameJoin[{repo, "part-09-from-scratch", "src", "part9_cells.wl"}]];

files = $ScriptCommandLine[[2 ;;]];
If[files === {}, files = {
  FileNameJoin[{repo, "LLM_A_to_Z.nb"}],
  FileNameJoin[{repo, "LLM_A_to_Z_evaluated.nb"}]}];

Scan[insertPart9File, files];
Print["done"];
