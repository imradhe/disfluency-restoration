"""Interleaved word / phone token sequence for the transformer tagger.

For an utterance with ASR words w_1..w_n the model reads::

    [CLS]  <gap 0 phones> w_1  <gap 1 phones> w_2 ... w_n  <gap n phones>  [SEP]

where ``<gap g phones>`` are the recognised phones that the aligner could not explain by any ASR word (one token per
phone, new vocabulary entries ``[ph_xx]``), and ``w_k`` is the word's word-pieces.  Every token also carries

* a *kind* embedding (special / first word-piece / later word-piece / gap phone), and
* the numeric alignment features of ``features.py`` (word: how well its G2P phones were found; gap phone: duration,
  pauses, position in the run) through a small MLP,

both added to the (pretrained) token embedding, so the encoder reads text, sounds and misalignment together.
Labels are BIO tags over :data:`TAGS`; a word's label sits on its first word-piece.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FG, FW
from .labeler import TAGS
from .phones import PHONES

LABELS = ["O"] + [f"{b}-{t}" for t in TAGS for b in "BI"]
L2I = {l: i for i, l in enumerate(LABELS)}
FMAX = max(FW, FG)
KIND_SPECIAL, KIND_WORD, KIND_CONT, KIND_PHONE = 0, 1, 2, 3
PHONE_TOKENS = [f"[ph_{p}]" for p in PHONES]
IGNORE = -100


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.add_tokens(PHONE_TOKENS, special_tokens=True)
    return tok


@dataclass
class Encoded:
    ids: list[int]
    kind: list[int]
    feats: np.ndarray  # (L, FMAX)
    labels: list[int]
    word_pos: list[int]  # token index of each ASR word's first piece
    gap_pos: list[list[int]]  # token indices of each gap's phones


class Encoder:
    def __init__(self, tokenizer, max_len: int = 320):
        self.tok = tokenizer
        self.max_len = max_len
        self.phone_id = {p: tokenizer.convert_tokens_to_ids(f"[ph_{p}]") for p in PHONES}
        self._pieces: dict[str, list[int]] = {}

    def pieces(self, word: str) -> list[int]:
        ids = self._pieces.get(word)
        if ids is None:
            ids = self.tok(word, add_special_tokens=False)["input_ids"] or [self.tok.unk_token_id]
            self._pieces[word] = ids
        return ids

    def encode(self, ex: dict, with_labels: bool = True) -> Encoded:
        tok = self.tok
        ids = [tok.cls_token_id]
        kind = [KIND_SPECIAL]
        feats = [np.zeros(FMAX, dtype=np.float32)]
        labels = [IGNORE]
        word_pos: list[int] = []
        gap_pos: list[list[int]] = []
        n = len(ex["words"])

        def add_gap(g: int):
            pos = []
            for k, ph in enumerate(ex["gp"][g]):
                pos.append(len(ids))
                ids.append(self.phone_id[ph])
                kind.append(KIND_PHONE)
                f = np.zeros(FMAX, dtype=np.float32)
                f[:FG] = ex["gf"][g][k]
                feats.append(f)
                labels.append(L2I[ex["gl"][g][k]] if with_labels and "gl" in ex else IGNORE)
            gap_pos.append(pos)

        for k in range(n):
            add_gap(k)
            word_pos.append(len(ids))
            for pi, pid in enumerate(self.pieces(ex["words"][k])):
                ids.append(pid)
                kind.append(KIND_WORD if pi == 0 else KIND_CONT)
                f = np.zeros(FMAX, dtype=np.float32)
                f[:FW] = ex["wf"][k]
                feats.append(f)
                labels.append(L2I[ex["wl"][k]] if (pi == 0 and with_labels and "wl" in ex) else IGNORE)
        add_gap(n)
        ids.append(tok.sep_token_id)
        kind.append(KIND_SPECIAL)
        feats.append(np.zeros(FMAX, dtype=np.float32))
        labels.append(IGNORE)
        if len(ids) > self.max_len:
            raise ValueError(f"sequence of {len(ids)} tokens exceeds max_len {self.max_len}")
        return Encoded(ids, kind, np.stack(feats), labels, word_pos, gap_pos)


def collate(batch: list[Encoded], pad_id: int):
    import torch

    L = max(len(b.ids) for b in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    kind = torch.zeros((B, L), dtype=torch.long)
    feats = torch.zeros((B, L, FMAX), dtype=torch.float32)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b.ids)
        ids[i, :n] = torch.tensor(b.ids)
        kind[i, :n] = torch.tensor(b.kind)
        feats[i, :n] = torch.from_numpy(b.feats)
        labels[i, :n] = torch.tensor(b.labels)
        attn[i, :n] = 1
    return {"ids": ids, "kind": kind, "feats": feats, "labels": labels, "attn": attn}


def decode_spans(label_ids: list[int], positions: list[int]) -> list[tuple[int, int, str]]:
    """BIO label ids over ``positions`` (token indices of one run) -> [(first, last, tag)] indices into ``positions``.
    An ``I-X`` that does not continue an ``X`` span starts a new one."""
    spans: list[tuple[int, int, str]] = []
    cur: list | None = None
    for i, p in enumerate(positions):
        lab = LABELS[label_ids[p]]
        if lab == "O":
            cur = None
            continue
        bi, tag = lab.split("-", 1)
        if bi == "I" and cur is not None and cur[2] == tag:
            cur[1] = i
        else:
            cur = [i, i, tag]
            spans.append(cur)
    return [tuple(s) for s in spans]
