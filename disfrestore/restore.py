"""Phones -> words.  Turns the typed gap spans predicted by the tagger into restored words, using the ASR transcript.

* FP / EX / BC  nearest entry of the filler inventory (uh um hm ah eh | huh ooh uh-oh | uh-huh um-hum huh-uh), by
                feature-weighted phone distance plus a frequency prior (Switchboard counts).
* REP           a copy of the neighbouring ASR words whose G2P best matches the phones (the n words before the gap or the
                n words after it, n <= 5); falls back to the generic decoder when nothing is close.
* PW            a fragment: the prefix of the next (or previous) word whose phone prefix best matches, spelled with the
                same proportion of letters  ->  ``th-``.  Falls back to ``[ph ph]-``.
* other types   a small lexicon decoder: segment the phones into words by dynamic programming, each word costing its edit
                distance to the pronunciation + a word penalty + a unigram prior (top-N Switchboard words).
* NS            kept as an event (``[noise]``), never inserted into the transcript.
"""
from __future__ import annotations

import math

import numpy as np

from .align import C_INDEL, edit_cost, norm_cost
from .lexicon import G2P
from .phones import N_PHONES, P2I, SUB
from .swb import SwbDict

FILLERS = {"uh": ["ah"], "um": ["ah", "m"], "hm": ["hh", "m"], "ah": ["aa"], "eh": ["eh"]}
EXCLAMATIONS = {"huh": ["hh", "ah"], "ooh": ["uw"], "uh-oh": ["ah", "ow"]}
BACKCHANNELS = {"uh-huh": ["ah", "hh", "ah"], "um-hum": ["ah", "m", "hh", "ah", "m"], "huh-uh": ["hh", "ah", "ah"]}
LMAX = 12
SUBX = np.full((N_PHONES, N_PHONES + 1), 1.0)
SUBX[:, :N_PHONES] = SUB
FRAGMENT_FAIL = 0.55  # normalised phone distance above which a fragment/repetition candidate is rejected
REP_FAIL = 0.55


class Restorer:
    def __init__(self, swb: SwbDict | None, g2p: G2P, counts: dict[str, int], n_lex: int = 6000,
                 word_penalty: float = 0.55, prior_weight: float = 0.10):
        self.g2p = g2p
        self.counts = counts
        self.total = float(sum(counts.values())) or 1.0
        skip = set(FILLERS) | set(EXCLAMATIONS) | set(BACKCHANNELS) | {"hum", "mm"}
        words, prons = [], []
        for w, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
            if w in skip or not w.replace("'", "").isalpha():
                continue
            pr = None
            if swb is not None:
                lk = swb.lookup(w)
                if lk:
                    pr = list(lk[0])
            if pr is None:
                pr = g2p.word(w)
            if 0 < len(pr) <= LMAX:
                words.append(w)
                prons.append(pr)
            if len(words) >= n_lex:
                break
        self.words = words
        self.lex = np.full((len(words), LMAX), N_PHONES, dtype=np.int64)
        for i, pr in enumerate(prons):
            self.lex[i, :len(pr)] = [P2I[p] for p in pr]
        self.lens = np.array([len(p) for p in prons])
        self.wpen = word_penalty
        self.prior = np.array([-math.log((counts[w] + 1) / self.total) for w in words]) * prior_weight
        self.rows = np.arange(len(words))

    # -- generic phones -> words ------------------------------------------------------------------------------
    def _span_costs(self, q: np.ndarray, s: int, maxlen: int) -> dict[int, np.ndarray]:
        n = len(self.words)
        idx = np.arange(LMAX + 1)
        prev = np.broadcast_to(idx * C_INDEL, (n, LMAX + 1)).copy()
        out = {}
        for i in range(1, min(maxlen, len(q) - s) + 1):
            sub = SUBX[q[s + i - 1]][self.lex]
            cur = np.empty_like(prev)
            cur[:, 0] = i * C_INDEL
            cur[:, 1:] = np.minimum(prev[:, :-1] + sub, prev[:, 1:] + C_INDEL)
            prev = np.minimum.accumulate(cur - C_INDEL * idx, axis=1) + C_INDEL * idx
            out[s + i] = prev[self.rows, self.lens]
        return out

    def decode(self, phones: list[str]) -> tuple[list[str], float]:
        """Best word sequence for a phone string and its cost per phone (lower = better, ~0.8+ means unreliable)."""
        m = len(phones)
        if m == 0 or not self.words:
            return [], 1.0
        q = np.array([P2I[p] for p in phones])
        best = [math.inf] * (m + 1)
        best[0] = 0.0
        back: list[tuple[int, int] | None] = [None] * (m + 1)
        for s in range(m):
            if best[s] == math.inf:
                continue
            for e, c in self._span_costs(q, s, LMAX + 2).items():
                tot = c + self.wpen + self.prior
                k = int(np.argmin(tot))
                if best[s] + tot[k] < best[e]:
                    best[e] = best[s] + float(tot[k])
                    back[e] = (s, k)
        out, e = [], m
        while e > 0:
            s, k = back[e]
            out.append(self.words[k])
            e = s
        return out[::-1], best[m] / m

    # -- type-specific restorers ------------------------------------------------------------------------------
    def _nearest(self, phones: list[str], inventory: dict[str, list[str]]) -> tuple[str, float]:
        best, bc = None, math.inf
        for w, pr in inventory.items():
            prior = -math.log((self.counts.get(w, 0) + 5) / self.total) * 0.04
            c = edit_cost(phones, pr) + prior
            if c < bc:
                best, bc = w, c
        return best, norm_cost(phones, inventory[best])

    def repetition(self, phones: list[str], words: list[str], gap: int) -> tuple[str, float, str] | None:
        best = None
        for side in ("next", "prev"):
            for n in range(1, 6):
                cand = words[gap:gap + n] if side == "next" else words[max(gap - n, 0):gap]
                if len(cand) != n:
                    continue
                pr = [p for w in cand for p in self.g2p.word(w)]
                if not pr:
                    continue
                c = norm_cost(phones, pr)
                if best is None or c < best[1] - 1e-9:
                    best = (" ".join(cand), c, side)
        return best if best and best[1] <= REP_FAIL else None

    def fragment(self, phones: list[str], words: list[str], gap: int) -> tuple[str, float] | None:
        best = None
        for w in (words[gap:gap + 1] + words[max(gap - 1, 0):gap]):
            pr = self.g2p.word(w)
            if not pr or not w.isalpha():
                continue
            for m in range(1, len(pr) + 1):
                c = edit_cost(phones, pr[:m]) / (C_INDEL * max(len(phones), m))
                if best is None or c < best[1] - 1e-9:
                    best = (w, c, m, len(pr))
        if best is None or best[1] > FRAGMENT_FAIL:
            return None
        w, c, m, n = best
        L = max(1, min(len(w) - 1, round(len(w) * m / n))) if len(w) > 1 else 1
        return w[:L] + "-", c

    def restore_span(self, tag: str, phones: list[str], words: list[str], gap: int) -> dict:
        item = {"gap": gap, "tag": tag, "phones": phones}
        if tag == "NS":
            item.update(text="[noise]", score=0.0, how="event")
        elif tag in ("FP", "EX", "BC"):
            inv = FILLERS if tag == "FP" else EXCLAMATIONS if tag == "EX" else BACKCHANNELS
            w, c = self._nearest(phones, inv)
            item.update(text=w, score=c, how="inventory")
        elif tag == "PW":
            fr = self.fragment(phones, words, gap)
            if fr:
                item.update(text=fr[0], score=fr[1], how="fragment")
            else:
                item.update(text="[" + " ".join(phones) + "]-", score=1.0, how="phones")
        else:
            rep = self.repetition(phones, words, gap) if tag in ("REP",) else None
            if rep:
                item.update(text=rep[0], score=rep[1], how=f"copy-{rep[2]}")
            else:
                ws, c = self.decode(phones)
                if ws and c <= 0.75:
                    item.update(text=" ".join(ws), score=c, how="lexicon")
                else:
                    item.update(text="[" + " ".join(phones) + "]", score=c, how="phones")
        return item


# --------------------------------------------------------------------------------------------------------------
def spans_from_predictions(enc, label_ids: list[int], phones_per_gap: list[list[str]]):
    """Predicted gap spans -> [{gap, tag, phones, idx}] using the BIO decoding of encoding.decode_spans."""
    from .encoding import decode_spans

    out = []
    for g, pos in enumerate(enc.gap_pos):
        for a, b, tag in decode_spans(label_ids, pos):
            out.append({"gap": g, "tag": tag, "phones": phones_per_gap[g][a:b + 1], "idx": (a, b)})
    return out


def word_spans(enc, label_ids: list[int]):
    from .encoding import decode_spans

    return [(a, b, tag) for a, b, tag in decode_spans(label_ids, enc.word_pos)]


def render(words: list[str], items: list[dict], wspans: list[tuple[int, int, str]] | None = None,
           tagged: bool = False, with_events: bool = False) -> str:
    """ASR words with restored items inserted into their gaps.  ``tagged`` marks restored items as {TAG:text} and
    disfluent words already in the ASR text as [TAG:words].  In the plain transcript, items whose phones could not be turned
    into words (``how == "phones"``, shown as ``[d]``) are left out: a bracketed phone string is never the right word; they
    stay visible in the tagged view and in the JSON output.  The exception is an unresolved partial word (``[k ah]-``), which
    is kept as a fragment marker."""
    by_gap: dict[int, list[dict]] = {}
    for it in items:
        by_gap.setdefault(it["gap"], []).append(it)
    starts = {a: (b, t) for a, b, t in (wspans or [])}
    out: list[str] = []
    k = 0
    for g in range(len(words) + 1):
        for it in by_gap.get(g, []):
            if it["tag"] == "NS" and not with_events:
                continue
            if not tagged and it.get("how") == "phones" and it["tag"] != "PW":
                continue
            out.append("{%s:%s}" % (it["tag"], it["text"]) if tagged else it["text"])
        if g < len(words):
            if tagged and g in starts:
                b, t = starts[g]
                out.append("[%s:%s]" % (t, " ".join(words[g:b + 1])))
                k = b + 1
            elif g >= k:
                out.append(words[g])
    return " ".join(out)
