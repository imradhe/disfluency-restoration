"""Turn verbatim Switchboard units into training examples that look like the pipeline's real inputs.

The free corpus has no audio, so both inputs of the pipeline are *simulated* from the verbatim transcript:

* **ASR transcript** (``words``): the verbatim words with the disfluent ones removed the way a normalising ASR (Whisper,
  wav2vec2+LM) removes them.  Each tag has a drop probability scaled by a per-example ``verbosity`` (0 = very clean ASR,
  ~0.9 = nearly verbatim ASR, in which case the disfluent words stay in the text and must be *tagged* instead of
  restored).  A little ASR noise (near-homophone substitutions, deletions, insertions) is added.
* **Recognised phones** (``rec``): the dictionary pronunciation (``sw-ms98-dict.text``, which has reduced variants and the
  *spoken part* of fragments) of every verbatim unit, spread over the unit's real time span from the word alignments,
  corrupted by a recogniser noise model (feature-aware substitutions, deletions, insertions, junk on laughter/noise).

Because the phones are generated from the units, every recognised phone knows its source unit.  That gives exact gold
labels for the unexplained phones the aligner finds (type + BIO span) and a gold list of dropped *items* with their words.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .align import align_phones
from .features import RecPhone, compute_features
from .labeler import RESTORABLE, Unit
from .lexicon import G2P
from .phones import DIST, PHONES, VOWELS
from .swb import SwbDict

# probability that an ASR drops a unit of this tag when it is "clean" (verbosity 0)
BASE_DROP = {"FP": 0.95, "EX": 0.90, "BC": 0.60, "PW": 0.90, "REP": 0.75, "RPR": 0.85, "FS": 0.90, "NS": 1.0,
             "DM": 0.05, "EET": 0.10}
ITEM_TAGS = set(RESTORABLE) | {"DM", "EET"}
JUNK = ["hh", "ah", "m", "n", "ae", "s", "sh", "ng", "eh", "aa", "uw", "f"]  # what recognisers hallucinate on laughter/noise
FUNCTION_WORDS = ["the", "a", "and", "to", "of", "i", "you", "it", "that", "in", "is", "so"]
WEAK = {"ah", "ih", "t", "d", "hh", "dh"}  # recognisers lose these most easily
# systematic Indian-English realisations a recogniser reports (phone -> (replacement, probability at accent strength 1)):
# dental fricatives become stops, v/w merge, z devoices, tense/lax and open/mid vowels are confused
ACCENT = {"th": ("t", 0.6), "dh": ("d", 0.6), "v": ("w", 0.35), "w": ("v", 0.2), "z": ("s", 0.3), "ae": ("eh", 0.2),
          "ao": ("aa", 0.25), "ey": ("eh", 0.2), "ow": ("ao", 0.2), "ih": ("iy", 0.15), "uh": ("uw", 0.2)}


@dataclass
class Profile:
    verbosity: float
    noise: float
    asr_sub: float
    asr_del: float
    asr_ins: float
    accent: float = 0.0  # 0 = none; otherwise the strength of systematic accent substitutions (see ACCENT)


def sample_profile(rng: random.Random) -> Profile:
    r = rng.random()
    verb = 0.0 if r < 0.5 else rng.uniform(0.1, 0.5) if r < 0.85 else rng.uniform(0.5, 0.9)
    accent = 0.0 if rng.random() < 0.4 else rng.uniform(0.3, 1.0)
    return Profile(verb, rng.random(), rng.uniform(0, 0.08), rng.uniform(0, 0.03), rng.uniform(0, 0.01), accent)


# --------------------------------------------------------------------------------------------------------------
# Near homophones for ASR substitution errors (edit distance 1 over G2P phones, found with deletion hashing)
# --------------------------------------------------------------------------------------------------------------
class Neighbors:
    def __init__(self, counts: dict[str, int], g2p: G2P, top: int = 20000):
        words = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top] if w.isalpha() or "'" in w]
        self.freq = {w: counts[w] for w in words}
        index: dict[tuple, list[str]] = {}
        self.pron = {}
        for w in words:
            p = tuple(g2p.word(w))
            if not p:
                continue
            self.pron[w] = p
            index.setdefault(p, []).append(w)
            for i in range(len(p)):
                index.setdefault(p[:i] + ("*",) + p[i + 1:], []).append(w)  # substitution
                index.setdefault(p[:i] + p[i + 1:], []).append(w)  # deletion
        self.index = index

    def near(self, w: str) -> list[str]:
        p = self.pron.get(w)
        if not p:
            return []
        out: set[str] = set()
        for i in range(len(p)):
            out.update(self.index.get(p[:i] + ("*",) + p[i + 1:], ()))
            out.update(self.index.get(p[:i] + p[i + 1:], ()))
        out.update(self.index.get(p, ()))
        out.discard(w)
        return sorted(out)


# --------------------------------------------------------------------------------------------------------------
# Phone-recogniser model
# --------------------------------------------------------------------------------------------------------------
_SUB_CHOICES: dict[str, tuple[list[str], list[float]]] = {}
for _i, _p in enumerate(PHONES):
    _others = [q for q in PHONES if q != _p]
    _w = [math.exp(-float(DIST[_i, PHONES.index(q)]) / 0.18) for q in _others]
    _SUB_CHOICES[_p] = (_others, _w)
_FREQ = ["ah", "ih", "n", "t", "r", "s", "l", "d", "k", "iy", "m", "ae", "eh", "z", "ay", "ow", "w", "b", "p", "dh"]


def _pronunciation(u: Unit, swb: SwbDict, g2p: G2P, rng: random.Random) -> list[str]:
    prons = swb.lookup(u.text)
    if prons:
        return list(prons[0] if len(prons) == 1 else rng.choice(prons))
    return g2p.word(u.word or u.spoken.strip("-"))


def simulate_recognizer(units: list[Unit], swb: SwbDict, g2p: G2P, prof: Profile, rng: random.Random) -> list[RecPhone]:
    n = prof.noise
    # real accented speech through a multilingual CTC recogniser is substitution-heavy and deletion-light
    p_sub, p_del, p_ins = 0.05 + 0.22 * n, 0.02 + 0.10 * n, 0.005 + 0.04 * n
    out: list[RecPhone] = []
    for u in units:
        if u.tag == "NS":
            phones = [rng.choice(JUNK) for _ in range(rng.randint(1, 4))] if rng.random() < 0.7 else []
        else:
            phones = []
            for p in _pronunciation(u, swb, g2p, rng):
                if prof.accent and p in ACCENT and rng.random() < ACCENT[p][1] * prof.accent:
                    p = ACCENT[p][0]
                if rng.random() < p_del * (1.5 if p in WEAK else 1.0):
                    continue
                if rng.random() < p_sub:
                    cand, w = _SUB_CHOICES[p]
                    p = rng.choices(cand, w)[0]
                phones.append(p)
                if rng.random() < p_ins:
                    phones.append(rng.choice(_FREQ) if rng.random() < 0.5 else p)
        if not phones:
            continue
        dt = max(u.end - u.start, 0.02 * len(phones)) / len(phones)
        for k, p in enumerate(phones):
            s = u.start + k * dt
            out.append(RecPhone(p, s, s + dt, 1.0, u.idx))
    return out


# --------------------------------------------------------------------------------------------------------------
# ASR model
# --------------------------------------------------------------------------------------------------------------
def asr_word(u: Unit) -> str:
    """The word a verbose ASR would write for a unit that it keeps."""
    if u.cat in ("partial_word", "cutoff_word"):
        return u.spoken if u.spoken.endswith("-") else u.spoken + "-"
    return u.word


def simulate_asr(units: list[Unit], prof: Profile, rng: random.Random, nb: Neighbors | None):
    """-> (words, word_unit) where word_unit[k] is the index of the unit behind ASR word k (-1 for an inserted word)."""
    words: list[str] = []
    wu: list[int] = []
    for u in units:
        if u.tag == "NS":
            continue
        base = BASE_DROP.get(u.tag)
        if base is not None and rng.random() < base * (1.0 - prof.verbosity):
            continue
        w = asr_word(u)
        if not w:
            continue
        r = rng.random()
        if u.tag == "O" and r < prof.asr_del:
            continue
        if u.tag == "O" and r < prof.asr_del + prof.asr_sub and nb is not None:
            cands = nb.near(w)
            if cands:
                w = rng.choices(cands, [nb.freq.get(c, 1) for c in cands])[0]
        words.append(w)
        wu.append(u.idx)
        if rng.random() < prof.asr_ins:
            words.append(rng.choice(FUNCTION_WORDS))
            wu.append(-1)
    return words, wu


# --------------------------------------------------------------------------------------------------------------
# Example assembly
# --------------------------------------------------------------------------------------------------------------
def _bio(keys: list[tuple]) -> list[str]:
    """keys: (tag, span_key) or ('O', None) -> BIO labels."""
    out, prev = [], None
    for k in keys:
        if k[0] == "O":
            out.append("O")
        else:
            out.append(("I-" if prev == k else "B-") + k[0])
        prev = k
    return out


def make_example(units: list[Unit], swb: SwbDict, g2p: G2P, rng: random.Random, nb: Neighbors | None = None,
                 prof: Profile | None = None, ex_id: str = "") -> dict | None:
    prof = prof or sample_profile(rng)
    words, wu = simulate_asr(units, prof, rng, nb)
    rec = simulate_recognizer(units, swb, g2p, prof, rng)
    if not words or not rec:
        return None
    kept = {k for k in wu if k >= 0}

    # --- gold items: dropped units with a restorable tag, in reference order, with the ASR gap they belong to ------
    gtype: dict[int, str] = {}
    for u in units:
        if u.group != -1 and u.tag in ("REP", "RPR", "FS"):
            gtype[u.group] = u.tag
    items: list[dict] = []
    item_of: dict[int, int] = {}
    # number of ASR words emitted up to and including each kept unit (ASR-inserted words attach to the unit before them)
    upto: dict[int, int] = {}
    last_k = -1
    for i, k in enumerate(wu):
        if k >= 0:
            last_k = k
        if last_k >= 0:
            upto[last_k] = i + 1
    unit_gap: dict[int, int] = {}
    cnt = 0
    for u in units:  # gap = slot before ASR word `cnt`: dropped units sit before the next kept word
        unit_gap[u.idx] = cnt
        if u.idx in upto:
            cnt = upto[u.idx]
    for u in units:
        if u.idx in kept or u.tag not in ITEM_TAGS:
            continue
        text = u.word or u.spoken
        if u.tag == "NS":
            text = u.text
        last = items[-1] if items else None
        if (last and u.tag in ("REP", "RPR", "FS") and last["tag"] == u.tag and last["group"] == u.group
                and last["gap"] == unit_gap[u.idx] and last["units"][-1] == u.idx - 1):
            last["text"] += " " + text
            last["units"].append(u.idx)
            item_of[u.idx] = len(items) - 1
            continue
        items.append({"gap": unit_gap[u.idx], "tag": u.tag, "text": text, "group": u.group, "units": [u.idx]})
        item_of[u.idx] = len(items) - 1

    ref_words = g2p.words(words)
    al = align_phones(ref_words, [p.ph for p in rec])
    wf, gf = compute_features(rec, al)

    # --- gold labels on ASR words ------------------------------------------------------------------------------
    wkeys = []
    for k in wu:
        u = units[k] if k >= 0 else None
        wkeys.append((u.tag, u.group if u.tag in ("REP", "RPR", "FS") else k) if u is not None and u.tag != "O"
                     else ("O", None))
    wl = _bio(wkeys)

    # --- gold labels on unexplained recognised phones ------------------------------------------------------------
    def rec_key(p: RecPhone):
        """(tag, span key) of the unit behind a recognised phone.  Reparandum and repair copy of one REP/RPR/FS group share
        a key: which of the two copies the aligner leaves in the gap is a tie, the span is the same."""
        u = units[p.src]
        if p.src in item_of:
            it = items[item_of[p.src]]
            if it["tag"] in ("REP", "RPR", "FS") and it["group"] != -1:
                return (it["tag"], ("g", it["group"]))
            return (it["tag"], item_of[p.src])
        if u.group != -1 and gtype.get(u.group) in ("REP", "RPR"):
            return (gtype[u.group], ("g", u.group))
        return ("O", None)

    gp, gfeat, gl = [], [], []
    keys_in_gap: list[list[tuple]] = []
    for run in al.gaps:
        ks = [rec_key(rec[j]) for j in run]
        keys_in_gap.append(ks)
        gp.append([rec[j].ph for j in run])
        gl.append(_bio(ks))
    gfeat = gf

    # coverage diagnostic = ceiling for restoration: share of a gold item's phones that the aligner left in a gap
    # (counting the other copy of a repetition, which is equivalent)
    n_in_gap: dict[tuple, int] = {}
    for ks in keys_in_gap:
        for k in ks:
            n_in_gap[k] = n_in_gap.get(k, 0) + 1
    for ix, it in enumerate(items):
        it["n_ph"] = sum(1 for p in rec if p.src in it["units"])
        key = (it["tag"], ("g", it["group"])) if it["tag"] in ("REP", "RPR", "FS") and it["group"] != -1 else (it["tag"], ix)
        it["covered"] = it["n_ph"] > 0 and n_in_gap.get(key, 0) >= 0.6 * it["n_ph"]

    ref = [w for w in (u.word if u.tag == "MP" else asr_word(u) for u in units if u.tag != "NS") if w]
    return {
        "id": ex_id, "words": words, "wl": wl, "wf": [[round(x, 4) for x in f] for f in wf],
        "gp": gp, "gf": [[[round(x, 4) for x in f] for f in run] for run in gfeat], "gl": gl,
        "items": [{k: v for k, v in it.items() if k not in ("units",)} for it in items],
        "ref": ref, "meta": {"verbosity": round(prof.verbosity, 3), "noise": round(prof.noise, 3), "accent": round(prof.accent, 3),
                              "n_rec": len(rec), "cost": round(al.cost, 2)},
    }
