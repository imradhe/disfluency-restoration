"""Disfluency taxonomy and labelling for Switchboard (MS98) transcripts.

Tag set (one tag per token; ``O`` = fluent)::

    FP   filled pause                    uh um hm hum ah eh                         native in MS98
    EX   other hesitation / exclamation  huh ooh uh-oh                              native
    BC   backchannel (conversational)    uh-huh um-hum huh-uh                       native (not a disfluency proper)
    PW   partial word / cut-off word     y[ou]-  -[o]kay  i-                        native
    MP   mispronounced word              [tranged/changed]                          native; the ASR word is the intended one
    NS   non-speech event                [laughter] [noise] [vocalized-noise]       native; NOT a disfluency, must not be restored
    REP  repetition (reparandum)         "the the", "i don't i don't"               derived by rule
    RPR  repair / substitution (reparandum)  "i went to the, i went to a store"     derived by rule
    FS   false start (abandoned words before a fragment)                            derived by rule
    EET  explicit editing term           "i mean", "you know" inside a repair       derived by rule (closed list)
    DM   discourse marker                "you know", "i mean" at clause edge        derived by rule (closed list)

The free release has no Treebank-3 ``EDITED`` layer, so REP/RPR/FS/EET/DM are *silver* labels from the rules below.
Reparandum = the part a clean-up would delete.  For REP/RPR the repair copy keeps tag ``O`` but shares the group id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .swb import Segment, Side, Tok, VARIANT

TAGS = ["FP", "EX", "BC", "PW", "MP", "NS", "REP", "RPR", "FS", "EET", "DM"]
# tags that a verbatim reference contains and a "clean" ASR transcript typically lacks (candidates for restoration)
RESTORABLE = ["FP", "EX", "BC", "PW", "REP", "RPR", "FS", "NS"]
DISFLUENCY = ["FP", "EX", "PW", "MP", "REP", "RPR", "FS", "EET", "DM"]  # BC and NS are not disfluencies

LEXICAL_CATS = {"word", "pron_variant", "laughter_word", "coinage", "mispronounced"}
FILLER_CATS = {"filled_pause", "other_hesitation", "backchannel"}
FRAGMENT_CATS = {"partial_word", "cutoff_word"}
NOISE_CATS = {"noise", "vocalized_noise", "laughter"}
LEGIT_REPEATS = {"bye", "very", "tut", "hee", "ha", "ho", "no no"}  # intensifying / lexicalised repeats are not disfluencies

_BRACKETS = re.compile(r"\[[^\]]*\]")


@dataclass(slots=True)
class Unit:
    idx: int
    text: str  # token as it appears in the corpus
    cat: str  # swb.classify() category
    start: float
    end: float
    seg: int  # index of the (merged) segment the unit came from
    word: str = ""  # surface form an ASR would write ("" for fragments and non-speech)
    spoken: str = ""  # what was actually said (differs from word for MP and fragments)
    tag: str = "O"
    group: int = -1  # disfluency instance this unit belongs to (reparandum, interregnum and repair share it)
    role: str = ""  # "rm" reparandum | "ip" interregnum | "rr" repair | ""


def surface(text: str, cat: str) -> tuple[str, str]:
    """(word, spoken) for a corpus token.  ``word`` is what a transcriber/ASR would write, ``spoken`` the realisation."""
    if cat in ("word", "pron_variant"):
        w = VARIANT.sub("", text).lower()
        return w, w
    if cat in FILLER_CATS:
        w = text.lower()
        return w, w
    if cat == "laughter_word":
        w = VARIANT.sub("", text[len("[laughter-"):].rstrip("]")).lower()
        return w, w
    if cat == "coinage":
        w = text.strip("{}").lower()
        return w, w
    if cat == "mispronounced":
        inner = text[1:-1]
        spoken, _, intended = inner.rpartition("/")
        return intended.lower(), _BRACKETS.sub("", spoken).lower()
    if cat in FRAGMENT_CATS:
        return "", _BRACKETS.sub("", text).lower()
    return "", ""


def units_from_segments(segs: list[Segment]) -> list[Unit]:
    """Flatten (merged) segments into labelled-able units; silence and aside markers are dropped (pauses are kept in the times)."""
    out: list[Unit] = []
    for si, seg in enumerate(segs):
        for t in seg.toks:
            if t.cat in ("silence", "aside_marker"):
                continue
            w, sp = surface(t.text, t.cat)
            out.append(Unit(len(out), t.text, t.cat, t.start, t.end, si, w, sp))
    return out


# --------------------------------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------------------------------
MAX_REP_N = 5  # longest repeated n-gram
MAX_RPR_RM = 6  # longest reparandum of a substitution repair
MAX_REPAIR_GAP = 1.5  # s; copies further apart than this are separate utterances, not a repair
CUE_PAUSE = 0.25  # s; pause that counts as a disfluency cue before a restart
FS_MAX_WORDS = 4
CLAUSE_STARTERS = {"and", "but", "so", "because", "or", "then", "that", "if", "when", "while"}
EET_PHRASES = [("you", "know"), ("i", "mean"), ("you", "see"), ("or", "rather"), ("excuse", "me"), ("sorry",)]
DM_PHRASES = [("you", "know"), ("i", "mean"), ("you", "see")]


def _native_tags(units: list[Unit]) -> None:
    for u in units:
        if u.cat == "filled_pause":
            u.tag = "FP"
        elif u.cat == "other_hesitation":
            u.tag = "EX"
        elif u.cat == "backchannel":
            u.tag = "BC"
        elif u.cat in FRAGMENT_CATS:
            u.tag = "PW"
        elif u.cat == "mispronounced":
            u.tag = "MP"
        elif u.cat in NOISE_CATS:
            u.tag = "NS"


def _lexical_index(units: list[Unit]) -> list[int]:
    """Indices of units that are words for repetition purposes (fillers, fragments, noise are skipped, not matched)."""
    return [u.idx for u in units if u.cat in LEXICAL_CATS]


def _cue_between(units: list[Unit], a: int, b: int) -> bool:
    """A disfluency cue between lexical units a < b: an interregnum unit, or a pause/segment break before b."""
    if any(units[k].cat in FILLER_CATS | FRAGMENT_CATS | {"mispronounced"} for k in range(a + 1, b)):
        return True
    return units[b].start - units[b - 1].end >= CUE_PAUSE or units[b].seg != units[b - 1].seg


def _mark(units: list[Unit], rm: list[int], rr_start: int, rr_end: int, tag: str, gid: int) -> None:
    """rm = lexical unit indices of the reparandum; repair spans rr_start..rr_end (lexical unit indices, inclusive)."""
    for k in rm:
        units[k].tag, units[k].group, units[k].role = tag, gid, "rm"
    first, last = rm[0], rr_start
    for k in range(first, last):
        u = units[k]
        if k not in rm and u.cat not in LEXICAL_CATS:
            u.group = gid
            u.role = "ip" if u.role == "" else u.role
    for k in range(rr_start, rr_end + 1):
        if units[k].cat in LEXICAL_CATS and units[k].group == -1:
            units[k].group, units[k].role = gid, "rr"


RESTART_STARTERS = {
    "i", "we", "you", "they", "he", "she", "it", "the", "a", "an", "there", "that", "this", "so", "and", "but", "well",
    "my", "our", "what", "if", "when", "how", "i'm", "it's", "that's", "we're", "they're", "you're", "i've", "i'd",
    "we've", "i'll", "we'll", "there's", "he's", "she's", "then", "because", "or", "oh", "yeah", "no", "not",
}


def _eet_occurrences(units: list[Unit]) -> list[tuple[int, int, tuple[str, ...]]]:
    """Positions (k, length, phrase) of closed-list editing-term phrases made of adjacent lexical units."""
    found = []
    n = len(units)
    k = 0
    while k < n:
        for phrase in sorted(EET_PHRASES, key=len, reverse=True):
            L = len(phrase)
            if k + L <= n and all(units[k + q].cat in LEXICAL_CATS and units[k + q].word == phrase[q] for q in range(L)):
                # "you know you know" is a repetition of the phrase, not an editing term between two copies of something
                if k + 2 * L <= n and all(units[k + L + q].word == phrase[q] for q in range(L)):
                    break
                found.append((k, L, phrase))
                k += L - 1
                break
        k += 1
    return found


def label_units(units: list[Unit]) -> list[Unit]:
    """Assign ``tag`` / ``group`` / ``role`` to every unit (in place) and return the list."""
    _native_tags(units)
    eet = _eet_occurrences(units)
    eet_idx = {k + q for k, L, _ in eet for q in range(L)}
    lex = [k for k in _lexical_index(units) if k not in eet_idx]  # editing terms are transparent to the repair rules
    words = [units[k].word for k in lex]
    gid = 0

    # --- 1. exact repetitions (longest first), chained: "the the the" -> first two copies are reparanda -----------
    i = 0
    while i < len(lex):
        hit = False
        for n in range(MAX_REP_N, 0, -1):
            j = i + n
            if j + n > len(lex):
                continue
            a, b = words[i:j], words[j:j + n]
            if a != b or " ".join(a) in LEGIT_REPEATS or (n == 1 and a[0] in LEGIT_REPEATS):
                continue
            if units[lex[j]].start - units[lex[j - 1]].end > MAX_REPAIR_GAP:
                continue
            if any(units[lex[k]].tag != "O" for k in range(i, j)):
                continue
            _mark(units, [lex[k] for k in range(i, j)], lex[j], lex[j + n - 1], "REP", gid)
            gid += 1
            i = j
            hit = True
            break
        if not hit:
            i += 1

    # --- 2. approximate restarts: the repair re-starts the reparandum's words but changes some ---------------------
    #     lead = words the repair shares with the start of the reparandum; accepted if the overlap is substantial
    #     (lead >= 2 and at most 2 words changed) or a cue (filler / fragment / pause) marks a short restart (lead == 1, n <= 2)
    for jj in range(1, len(lex)):
        j = lex[jj]
        if units[j].tag != "O" or units[j].group != -1:
            continue
        for n in range(2, MAX_RPR_RM + 1):  # n == 1 with lead == 1 would be an exact repetition (handled above)
            ii = jj - n
            if ii < 0:
                break
            i = lex[ii]
            if any(units[lex[k]].tag != "O" or units[lex[k]].group != -1 for k in range(ii, jj)):
                continue
            if words[ii] != words[jj] or words[ii] in LEGIT_REPEATS:
                continue
            if units[j].start - units[lex[jj - 1]].end > MAX_REPAIR_GAP:
                continue
            lead = 0
            while lead < n and jj + lead < len(lex) and words[ii + lead] == words[jj + lead]:
                lead += 1
            if lead >= n:
                continue  # whole reparandum repeated: that is a REP (or overlapping chain), not a substitution
            cue = _cue_between(units, lex[jj - 1], j)
            if (lead >= 2 and n - lead <= 2) or (lead == 1 and cue and n <= 2):
                span_end = min(jj + lead + 1, len(lex)) - 1
                _mark(units, [lex[k] for k in range(ii, jj)], j, lex[span_end], "RPR", gid)
                gid += 1
                break

    # --- 3. false starts: abandoned words before a fragment, when the speaker then starts afresh in the same segment
    for k, u in enumerate(units):
        if u.cat not in FRAGMENT_CATS:
            continue
        nxt = next((units[m] for m in range(k + 1, len(units)) if units[m].cat in LEXICAL_CATS | FRAGMENT_CATS), None)
        if nxt is None or nxt.seg != u.seg or nxt.cat in FRAGMENT_CATS:
            continue  # utterance-final cut-off (interruption) or a stutter on the same word
        frag = u.spoken.strip("-")
        if frag and nxt.word.startswith(frag[:2]):
            continue  # word-internal self-repair ("y- you"): the fragment alone is the disfluency
        before = [units[m].word for m in range(max(0, k - FS_MAX_WORDS), k)
                  if units[m].cat in LEXICAL_CATS and units[m].seg == u.seg]
        if nxt.word not in RESTART_STARTERS and nxt.word not in before:
            continue
        taken, m = 0, k - 1
        while m >= 0 and taken < FS_MAX_WORDS:
            v = units[m]
            if v.seg != u.seg or v.cat not in LEXICAL_CATS or v.tag != "O" or v.role != "":
                break
            taken += 1
            m -= 1
            if v.word in CLAUSE_STARTERS:
                break
        if taken:
            for q in range(m + 1, k):
                if units[q].cat in LEXICAL_CATS and units[q].tag == "O" and units[q].role == "":
                    units[q].tag, units[q].group, units[q].role = "FS", gid, "rm"
            u.group, u.role = gid, "rm"
            gid += 1

    # --- 4. editing terms (between a reparandum and its repair) and discourse markers (clause edge / cue) ------------
    def neighbour(k: int, step: int):
        m = k + step
        while 0 <= m < len(units):
            if units[m].cat in LEXICAL_CATS and m not in eet_idx:
                return units[m]
            if units[m].cat not in FILLER_CATS | FRAGMENT_CATS | NOISE_CATS and m not in eet_idx:
                return None
            m += step
        return None

    n = len(units)
    for k, L, phrase in eet:
        if any(units[k + q].tag != "O" for q in range(L)):
            continue
        prev, nxt = neighbour(k, -1), neighbour(k + L - 1, +1)
        between = (prev is not None and nxt is not None and prev.role == "rm" and nxt.role == "rr"
                   and prev.group == nxt.group)
        tag = ""
        if between:
            tag = "EET"
        else:
            before = units[k - 1] if k > 0 else None
            after = units[k + L] if k + L < n else None
            edge = before is None or after is None or before.seg != units[k].seg or after.seg != units[k + L - 1].seg
            near_filler = any(x is not None and x.cat in FILLER_CATS | FRAGMENT_CATS for x in (before, after))
            pause = (before is not None and units[k].start - before.end >= CUE_PAUSE) or \
                    (after is not None and after.start - units[k + L - 1].end >= CUE_PAUSE)
            if phrase in DM_PHRASES and (edge or near_filler or pause):
                tag = "DM"
            elif phrase not in DM_PHRASES and (near_filler or pause):
                tag = "EET"
        if tag:
            for q in range(L):
                units[k + q].tag = tag
                if between:
                    units[k + q].group, units[k + q].role = prev.group, "ip"
    return units


# --------------------------------------------------------------------------------------------------------------
# Chunking: merge neighbouring speech segments of one speaker into training chunks
# --------------------------------------------------------------------------------------------------------------
def has_speech(seg: Segment) -> bool:
    return any(t.cat not in ("silence", "aside_marker") for t in seg.toks)


def chunk_side(side: Side, max_gap: float = 1.0, max_words: int = 40) -> list[list[Segment]]:
    """Group consecutive speech-bearing segments whose gap is <= ``max_gap`` seconds, up to ``max_words`` tokens."""
    chunks: list[list[Segment]] = []
    cur: list[Segment] = []
    n = 0
    for seg in side.segments:
        if not has_speech(seg):
            continue
        k = sum(1 for t in seg.toks if t.cat not in ("silence", "aside_marker"))
        if cur and (seg.start - cur[-1].end > max_gap or n + k > max_words):
            chunks.append(cur)
            cur, n = [], 0
        cur.append(seg)
        n += k
    if cur:
        chunks.append(cur)
    return chunks
