(* ::Package:: *)

(* ============================================================
   CharTokeniser.wl  ---  the char-level alternative to Part 1's BPE
   ============================================================

   For very small models on Shakespeare-class corpora, a character-
   level vocabulary turns out to be the better choice: the per-token
   capacity gain (vocab ~65 vs BPE 4000) more than compensates for
   the longer sequences.  Karpathy's char-rnn (2015) at ~1M LSTM
   parameters produced surprisingly coherent Shakespeare output on
   exactly this trade-off.

   Drop-in replacement for the BPE pieces of Part 1:
     - charVocabulary[text]      sorted character list
     - encodeChars[text, tokId]  raw text -> token-id sequence
     - decodeChars[ids, idTok]   token-id sequence -> raw text *)

BeginPackage["LLMAtoZ`CharTokeniser`"];

ClearAll[charVocabulary, charTokenToIdAssociation,
        charIdToTokenAssociation, encodeChars, decodeChars];

charVocabulary::usage =
  "charVocabulary[text] returns the sorted list of distinct \
characters in text, ready to be used as the model's vocab.";

charTokenToIdAssociation::usage =
  "charTokenToIdAssociation[vocab] returns an Association from each \
character to its 1-based integer ID.";

charIdToTokenAssociation::usage =
  "charIdToTokenAssociation[vocab] returns the inverse association.";

encodeChars::usage =
  "encodeChars[text, tokId] returns the list of integer IDs for \
each character of text.  Characters absent from the vocab map to \
ID 1 (a safe fall-back; in practice we train on the full corpus \
so out-of-vocab chars are rare).";

decodeChars::usage =
  "decodeChars[ids, idTok] joins the per-character lookups back \
into a string.";

Begin["`Private`"];

charVocabulary[text_String] := Union[Characters[text]];

charTokenToIdAssociation[vocab_List] :=
  AssociationThread[vocab -> Range[Length[vocab]]];

charIdToTokenAssociation[vocab_List] :=
  AssociationThread[Range[Length[vocab]] -> vocab];

encodeChars[text_String, tokId_Association] :=
  Lookup[tokId, Characters[text], 1];

decodeChars[ids_List, idTok_Association] :=
  StringJoin @ Lookup[idTok, ids, ""];

End[];
EndPackage[];
