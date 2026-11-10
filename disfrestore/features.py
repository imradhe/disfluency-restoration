"""Numeric features that describe how the recognised phones relate to the ASR words (the "misalignment").

Word token (FW = 11 numbers)
    0 n_ref/8        number of G2P phones of the word
    1 match          fraction of the word's G2P phones found exactly in the recognised phones
    2 sub            ... found as a substitution
    3 del            ... not found at all
    4 ins            recognised phones inserted *inside* the word, relative to n_ref
    5 sub_cost       mean articulatory distance of the substitutions (0..1)
    6 pause_before   silence before the word's first recognised phone (s / 2)
    7 pause_after    silence after its last recognised phone (s / 2)
    8 rate           word duration / (0.08 s * n_ref), clipped 0..3, / 3
    9 absent         1 if none of the word's phones was recognised
   10 chunk_cost     alignment cost per reference phone of the whole chunk (how much to trust the phones), clipped 0..1
Gap phone token (FG = 9 numbers)
    0 dur/0.4   1 pause_after/2   2 pause_before/2   3 position in run   4 first in run   5 last in run
    6 run length/10   7 run duration/2   8 chunk_cost

All times are relative: a recogniser phone's duration is the time to the next phone start capped at 0.4 s, the rest is
"pause".  This works the same for simulated phones (word alignments) and for CTC phones from real audio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import Alignment

FW = 11
FG = 9
MAX_DUR = 0.4


@dataclass
class RecPhone:
    ph: str
    start: float
    end: float
    conf: float = 1.0
    src: int = -1  # simulation only: unit the phone came from


def _timing(rec: list[RecPhone]) -> tuple[np.ndarray, np.ndarray]:
    """(duration, pause_after) per recognised phone, seconds."""
    n = len(rec)
    dur = np.zeros(n)
    pause = np.zeros(n)
    for j, p in enumerate(rec):
        nxt = rec[j + 1].start if j + 1 < n else max(p.end, p.start + 0.05)
        gap = max(nxt - p.start, 0.0)
        dur[j] = min(gap, MAX_DUR)
        pause[j] = max(gap - MAX_DUR, 0.0) if j + 1 < n else 0.0
    return dur, pause


def compute_features(rec: list[RecPhone], al: Alignment):
    """-> (word_feats [n_words][FW], gap_feats [n_words+1][run][FG])"""
    dur, pause = _timing(rec)
    n_ref = sum(w.n_ref for w in al.words)
    chunk_cost = float(min(1.0, al.cost / max(n_ref, 1)))
    wf: list[list[float]] = []
    for w in al.words:
        n = max(w.n_ref, 1)
        if w.rec_first >= 0:
            a, b = w.rec_first, w.rec_last
            before = pause[a - 1] if a > 0 else 0.0
            after = pause[b]
            span = (rec[b].start + dur[b]) - rec[a].start
        else:
            before = after = span = 0.0
        wf.append([
            min(w.n_ref / 8.0, 1.5), w.match / n, w.sub / n, w.dele / n, w.ins / n,
            (w.sub_cost / w.sub / 1.2) if w.sub else 0.0,
            min(before / 2.0, 1.0), min(after / 2.0, 1.0),
            min(span / (0.08 * n), 3.0) / 3.0 if w.rec_first >= 0 else 0.0,
            1.0 if w.rec_first < 0 and w.n_ref > 0 else 0.0,
            chunk_cost,
        ])
    gf: list[list[list[float]]] = []
    for run in al.gaps:
        L = len(run)
        total = float(sum(dur[j] for j in run)) if L else 0.0
        feats = []
        for k, j in enumerate(run):
            before = pause[j - 1] if j > 0 else 0.0
            feats.append([
                dur[j] / MAX_DUR, min(pause[j] / 2.0, 1.0), min(before / 2.0, 1.0),
                k / max(L - 1, 1), float(k == 0), float(k == L - 1), min(L / 10.0, 1.5), min(total / 2.0, 1.0),
                chunk_cost,
            ])
        gf.append(feats)
    return wf, gf
