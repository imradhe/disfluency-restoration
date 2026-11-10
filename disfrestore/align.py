"""Alignment of the recognised phone sequence against the G2P phones of the ASR transcript.

The ASR words are turned into one reference phone string (word boundaries remembered).  A global alignment (Needleman-
Wunsch) against the recognised phones gives, per ASR word, how well its phones were found (match / substitution /
deletion / insertion inside the word) and, per *gap* (the slot before word k, k = 0..n), the recognised phones that no
word explains.  Those unexplained runs are the evidence for sounds the ASR dropped: fillers, repeated words, fragments.

Costs: substitution = articulatory distance (phones.SUB, 0.25..1.2), deletion 0.75, insertion between words 0.45 (cheap:
that is where dropped material appears) and inside a word 0.9.  Ties are resolved so that insertions are placed as far
left as possible, i.e. a repeated word shows up as an extra copy *before* the retained one (reparandum before repair).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .phones import P2I, SUB

C_DEL = 0.75
C_INT = 0.90
C_GAP = 0.45
C_INDEL = 0.75  # plain edit distance (restoration)


@dataclass
class WordAlign:
    n_ref: int
    match: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0  # recognised phones inserted inside the word
    sub_cost: float = 0.0
    rec_first: int = -1  # first / last recognised phone aligned to this word (or inserted inside it)
    rec_last: int = -1


@dataclass
class Alignment:
    words: list[WordAlign]
    gaps: list[list[int]]  # n_words + 1 slots; recognised-phone indices unexplained by any word
    rec_word: list[int]  # per recognised phone: index of the word it belongs to, -1 if in a gap
    cost: float
    ops: list[tuple[str, int, int]] = field(default_factory=list)  # (M|S|D|I, ref phone idx, rec phone idx)


def align_phones(ref_words: list[list[str]], rec: list[str]) -> Alignment:
    n_words = len(ref_words)
    g = np.array([P2I[p] for w in ref_words for p in w], dtype=np.int64)
    r = np.array([P2I[p] for p in rec], dtype=np.int64)
    N, M = len(g), len(r)
    ends = np.cumsum([len(w) for w in ref_words]) if n_words else np.zeros(0, dtype=np.int64)
    boundary = np.zeros(N + 1, dtype=bool)
    boundary[0] = True
    boundary[ends] = True
    word_of = np.repeat(np.arange(n_words), [len(w) for w in ref_words]) if N else np.zeros(0, dtype=np.int64)
    ci_row = np.where(boundary, C_GAP, C_INT)

    D = np.empty((N + 1, M + 1), dtype=np.float64)
    idx = np.arange(M + 1, dtype=np.float64)
    D[0] = idx * ci_row[0]
    for i in range(1, N + 1):
        prev = D[i - 1]
        cand = np.empty(M + 1)
        cand[0] = prev[0] + C_DEL
        if M:
            cand[1:] = np.minimum(prev[:-1] + SUB[g[i - 1]][r], prev[1:] + C_DEL)
        ci = ci_row[i]
        D[i] = np.minimum.accumulate(cand - ci * idx) + ci * idx

    # backtrace, preferring diagonal, then deletion, then insertion -> insertions end up leftmost
    ops: list[tuple[str, int, int]] = []
    i, j = N, M
    eps = 1e-7
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = SUB[g[i - 1], r[j - 1]]
            if abs(D[i, j] - (D[i - 1, j - 1] + s)) < eps:
                ops.append(("M" if g[i - 1] == r[j - 1] else "S", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(D[i, j] - (D[i - 1, j] + C_DEL)) < eps:
            ops.append(("D", i - 1, -1))
            i -= 1
            continue
        ops.append(("I", i, j - 1))  # i = row = number of ref phones consumed so far
        j -= 1
    ops.reverse()

    words = [WordAlign(len(w)) for w in ref_words]
    gaps: list[list[int]] = [[] for _ in range(n_words + 1)]
    rec_word = [-1] * M
    for op, a, b in ops:
        if op in "MS":
            w = words[word_of[a]]
            if op == "M":
                w.match += 1
            else:
                w.sub += 1
                w.sub_cost += float(SUB[g[a], r[b]])
            rec_word[b] = int(word_of[a])
            w.rec_first = b if w.rec_first < 0 else min(w.rec_first, b)
            w.rec_last = max(w.rec_last, b)
        elif op == "D":
            words[word_of[a]].dele += 1
        else:  # insertion at row a
            if boundary[a]:
                gap = int(np.searchsorted(ends, a, side="right")) if n_words else 0
                gaps[gap].append(b)
            else:
                k = int(word_of[a - 1])
                w = words[k]
                w.ins += 1
                rec_word[b] = k
                w.rec_first = b if w.rec_first < 0 else min(w.rec_first, b)
                w.rec_last = max(w.rec_last, b)
    return Alignment(words, gaps, rec_word, float(D[N, M]), ops)


def edit_cost(a: list[str], b: list[str]) -> float:
    """Feature-weighted edit distance between two phone strings (indel 0.75, substitution phones.SUB)."""
    if not a or not b:
        return C_INDEL * (len(a) + len(b))
    ai = np.array([P2I[p] for p in a])
    bi = np.array([P2I[p] for p in b])
    prev = np.arange(len(bi) + 1, dtype=np.float64) * C_INDEL
    for i in range(1, len(ai) + 1):
        cur = np.empty_like(prev)
        cur[0] = i * C_INDEL
        cur[1:] = np.minimum(prev[:-1] + SUB[ai[i - 1]][bi], prev[1:] + C_INDEL)
        # horizontal moves (insert b phone) are the running closure
        idx = np.arange(len(bi) + 1)
        cur = np.minimum.accumulate(cur - C_INDEL * idx) + C_INDEL * idx
        prev = cur
    return float(prev[-1])


def norm_cost(a: list[str], b: list[str]) -> float:
    """edit_cost divided by the longer string: ~0 for equal, ~1 for unrelated."""
    return edit_cost(a, b) / (C_INDEL * max(len(a), len(b), 1))
