#!/usr/bin/env wolframscript
(* Export the trained Wolfram-format GPT-2 medium SFT model into one .npy file
   per HuggingFace state-dict key. The .npy files are read back in
   load_into_hf.py to produce a real HF model directory.

   Pipeline order (each step writes to data/hf_weights/ or data/hf_model/):
     1. dump_wl_vocab.wls   -> data/hf_weights/wl_vocab.json
     2. export_wl_to_hf.wls -> data/hf_weights/*.npy
     3. load_into_hf.py     -> data/hf_model/*
     4. chat_python.py      -> interactive REPL.

   Layer mapping (discovered via probe_layers.wls):

   Outer NetChain wraps an inner NetChain[NetGraph "embedding",
   NetChain "decoder" (24 transformer blocks + final NormalizationLayer),
   NetMapOperator "classifier" wrapping a LinearLayer with weight tied to
   the token embedding].

   For each transformer block i (1..24), the WL path is
     {1, 2, i, 1, ...}  -> attention sub-NetGraph
     {1, 2, i, 2, ...}  -> MLP sub-NetGraph
   Attention nodes (NetMapOperator wrapping LinearLayer accessed via "Net"):
     "attention" 1: Q   (1024x1024)
     "attention" 2: K   (1024x1024)
     "attention" 4: V   (1024x1024)
     "attention" 6: Out (1024x1024)
   MLP nodes:
     "linear1" (Net): up-projection (4096x1024)
     "linear2" (Net): down-projection (1024x4096)
   Final LayerNorm: {1, 2, 25, "Scaling"/"Biases"}.

   Wolfram's LinearLayer stores Weights as (out, in).  HF's Conv1D linears
   (c_attn, c_proj, c_fc, mlp.c_proj) store weight as (in, out), so we
   transpose. *)

SetDirectory["/Users/thiel/GitHub/LLM-A-to-Z-in-WL/discussion-model"];
outDir = "data/hf_weights";
If[!DirectoryQ[outDir], CreateDirectory[outDir]];

(* ------------------------------------------------------------------ *)
(* tiny .npy writer (float32 only — sufficient here) *)
(* ------------------------------------------------------------------ *)
(* .npy header v1.0:
     6 bytes magic = "\x93NUMPY"
     1 byte  major version (1)
     1 byte  minor version (0)
     2 bytes (little endian) header_len
     header_len bytes ASCII Python dict-literal padded with spaces to a
       16-byte boundary (so total header_len is divisible by 16),
       terminated with a newline.
   followed by raw float32 data in C order. *)

shapeString[dims_List] := Module[{parts},
  parts = Map[ToString, dims];
  If[Length[dims] == 1, parts = {parts[[1]], ""}];
  "(" <> StringJoin[Riffle[parts, ", "]] <> ")"];

writeNpyFloat32[file_String, array_] := Module[
  {arr = N[Normal[array]], dims, header, headerLen, pad, byteCount, fp, dictPart},
  arr = If[Head[arr] === Real, {{arr}}, arr];
  dims = Dimensions[arr];
  (* Build header dict text *)
  dictPart = "{'descr': '<f4', 'fortran_order': False, 'shape': " <> shapeString[dims] <> ", }";
  (* total header bytes = 10 (magic + version + lenfield) + len(dict) + 1 newline.
     pad with spaces to make total length divisible by 64 (numpy default). *)
  byteCount = 10 + StringLength[dictPart] + 1;
  pad = Mod[64 - Mod[byteCount, 64], 64];
  header = dictPart <> StringRepeat[" ", pad] <> "\n";
  headerLen = StringLength[header];
  fp = OpenWrite[file, BinaryFormat -> True];
  (* magic *)
  BinaryWrite[fp, ToCharacterCode["\[Backslash]"][[1]] /. _ -> 0]; (* unused *)
  Close[fp];
  (* Restart with BinaryWrite *)
  fp = OpenWrite[file, BinaryFormat -> True];
  BinaryWrite[fp, {147}, "UnsignedInteger8"]; (* 0x93 *)
  BinaryWrite[fp, ToCharacterCode["NUMPY"], "UnsignedInteger8"];
  BinaryWrite[fp, {1, 0}, "UnsignedInteger8"];   (* version 1.0 *)
  BinaryWrite[fp, {headerLen}, "UnsignedInteger16"];  (* little-endian by default *)
  BinaryWrite[fp, ToCharacterCode[header], "UnsignedInteger8"];
  (* Data: flatten in row-major order and write as Real32 *)
  Module[{flat = Flatten[arr]},
    BinaryWrite[fp, flat, "Real32"]];
  Close[fp];];

(* ------------------------------------------------------------------ *)
(* Load trained net *)
(* ------------------------------------------------------------------ *)
Print["Loading discussion_medium_20k.wlnet..."];
t0 = AbsoluteTime[];
net = Import["data/discussion_medium_20k.wlnet"];
Print["  loaded in ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print["  total params = ", Information[net, "ArraysTotalElementCount"]];

(* ------------------------------------------------------------------ *)
(* Convenience: extract & convert a tensor to a real array *)
(* ------------------------------------------------------------------ *)
get[path_List] := Normal[NetExtract[net, path]];

(* Track shape stats *)
totalParams = 0;
writeTensor[name_String, arr_] := Module[{d},
  d = Dimensions[arr];
  totalParams += Times @@ d;
  Print["  ", StringPadRight[name, 40], " ", d];
  writeNpyFloat32[FileNameJoin[{outDir, name <> ".npy"}], arr]];

Print["\nWriting tensors to ", outDir, "/ ..."];

(* Token & positional embeddings — keep as (vocab, dim) and (max, dim);
   HF wte.weight is (50257, 1024), wpe.weight is (1024, 1024).  Both shapes
   match what WL stores, no transpose needed. *)
writeTensor["transformer.wte.weight", get[{1, 1, "embeddingtokens", "Weights"}]];
writeTensor["transformer.wpe.weight", get[{1, 1, "embeddingpos", "Weights"}]];

(* Per-block params *)
Do[
  Module[{prefix = "transformer.h." <> ToString[i - 1] <> ".",
          attnPath = {1, 2, i, 1}, mlpPath = {1, 2, i, 2},
          wq, wk, wv, wo, bq, bk, bv, bo, cAttnW, cAttnB},

    (* LN1 (Wolfram NormalizationLayer "Scaling"/"Biases" -> HF "weight"/"bias") *)
    writeTensor[prefix <> "ln_1.weight",
                get[Join[attnPath, {"norm", "Scaling"}]]];
    writeTensor[prefix <> "ln_1.bias",
                get[Join[attnPath, {"norm", "Biases"}]]];

    (* Build c_attn (in, 3*out) by concatenating Q,K,V on the out dim.
       WL stores Weights as (out, in)=1024x1024. HF c_attn.weight is
       (in=1024, out=3072) -> Conv1D. So:
         - take WL (1024,1024) for each of Q, K, V
         - HF row of c_attn.weight at out index j corresponds to in dim.
         - HF: c_attn.weight = Transpose[Q] | Transpose[K] | Transpose[V]
       where | is column-concat (along axis=1, out dim).
       For biases, c_attn.bias = Join[bQ, bK, bV] (length 3072).

       GOTCHA: in the WL attention sub-NetGraph the linear at node 1 feeds
       the AttentionLayer's Key port and the linear at node 2 feeds its
       Query port (verified via NetGraph edge inspection). So WL's
       "attention,1" is K and "attention,2" is Q — opposite of the
       intuitive numbering. The 1/sqrt(64) scaling at node 3 is in the
       Query path; HF applies 1/sqrt(d_k) inside its attention so the
       overall factor is identical. *)
    wq = get[Join[attnPath, {"attention", 2, "Net", "Weights"}]];
    bq = get[Join[attnPath, {"attention", 2, "Net", "Biases"}]];
    wk = get[Join[attnPath, {"attention", 1, "Net", "Weights"}]];
    bk = get[Join[attnPath, {"attention", 1, "Net", "Biases"}]];
    wv = get[Join[attnPath, {"attention", 4, "Net", "Weights"}]];
    bv = get[Join[attnPath, {"attention", 4, "Net", "Biases"}]];
    (* Conv1D weight shape (in_features, out_features) = (1024, 3072).
       Each WL piece (1024, 1024) is (out, in); we want (in, out_concat). *)
    cAttnW = ArrayFlatten[{{Transpose[wq], Transpose[wk], Transpose[wv]}}];
    (* cAttnW now is (1024, 3072) *)
    cAttnB = Join[bq, bk, bv];
    writeTensor[prefix <> "attn.c_attn.weight", cAttnW];
    writeTensor[prefix <> "attn.c_attn.bias",   cAttnB];

    (* Attention output projection: WL (out=1024, in=1024) -> HF c_proj
       weight (in=1024, out=1024) = Transpose. *)
    wo = get[Join[attnPath, {"attention", 6, "Net", "Weights"}]];
    bo = get[Join[attnPath, {"attention", 6, "Net", "Biases"}]];
    writeTensor[prefix <> "attn.c_proj.weight", Transpose[wo]];
    writeTensor[prefix <> "attn.c_proj.bias",   bo];

    (* LN2 *)
    writeTensor[prefix <> "ln_2.weight",
                get[Join[mlpPath, {"norm", "Scaling"}]]];
    writeTensor[prefix <> "ln_2.bias",
                get[Join[mlpPath, {"norm", "Biases"}]]];

    (* MLP up: WL linear1 weights (4096, 1024) -> HF c_fc (1024, 4096) *)
    writeTensor[prefix <> "mlp.c_fc.weight",
                Transpose[get[Join[mlpPath, {"linear1", "Net", "Weights"}]]]];
    writeTensor[prefix <> "mlp.c_fc.bias",
                get[Join[mlpPath, {"linear1", "Net", "Biases"}]]];

    (* MLP down: WL linear2 weights (1024, 4096) -> HF (4096, 1024) *)
    writeTensor[prefix <> "mlp.c_proj.weight",
                Transpose[get[Join[mlpPath, {"linear2", "Net", "Weights"}]]]];
    writeTensor[prefix <> "mlp.c_proj.bias",
                get[Join[mlpPath, {"linear2", "Net", "Biases"}]]];
  ],
  {i, 1, 24}];

(* Final LayerNorm (at decoder index 25) *)
writeTensor["transformer.ln_f.weight", get[{1, 2, 25, "Scaling"}]];
writeTensor["transformer.ln_f.bias",   get[{1, 2, 25, "Biases"}]];

(* lm_head.weight: tied to wte.weight (already in shared array). HF saves
   lm_head.weight as (vocab=50257, dim=1024). We write the same tensor again
   to lm_head.weight for explicitness; the load script can either set it
   or rely on the wte tie. *)
writeTensor["lm_head.weight", get[{1, 3, "Net", "Weights"}]];

Print["\nTotal exported tensor element count: ", totalParams];
Print["  (model total: 354823168; difference is lm_head tied-vs-explicit)"];
Print["\nFiles written to ", outDir];
Print["Done."];
