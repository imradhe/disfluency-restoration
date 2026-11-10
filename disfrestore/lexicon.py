"""Grapheme-to-phoneme conversion for ASR transcripts, in the CMU39 inventory (no stress).

Order of lookup for a word:  special table (fillers, backchannels)  ->  CMUdict (first pronunciation)  ->  hyphen split
->  digits (spelled out by g2p_en's normaliser)  ->  g2p_en's neural model (OOV: names, jargon such as "NPTEL").
Results are cached; ``G2P.save``/``G2P.load`` persist the cache so worker processes do not re-run the neural model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .phones import P2I, strip_stress

# fillers and backchannels as they are pronounced (same phones the Switchboard dictionary gives them)
SPECIAL: dict[str, list[str]] = {
    "uh": ["ah"], "um": ["ah", "m"], "hm": ["hh", "m"], "hmm": ["hh", "m"], "mm": ["m"], "mmm": ["m"], "hum": ["hh", "m"],
    "ah": ["aa"], "eh": ["eh"], "er": ["er"], "erm": ["er", "m"], "huh": ["hh", "ah"], "ooh": ["uw"], "oh": ["ow"],
    "uh-huh": ["ah", "hh", "ah"], "um-hum": ["ah", "m", "hh", "ah", "m"], "huh-uh": ["hh", "ah", "ah"],
    "hum-um": ["hh", "m", "ah", "m"], "uh-hum": ["ah", "hh", "ah", "m"], "mm-hm": ["m", "hh", "m"],
    "mhm": ["m", "hh", "m"], "uh-oh": ["ah", "ow"],
}
_STRIP = re.compile(r"^[^\w']+|[^\w']+$")
_NOT_LETTER = re.compile(r"[^a-z']")


class G2P:
    def __init__(self, neural: bool = True):
        import cmudict

        self.cmu = cmudict.dict()
        self.use_neural = neural
        self._neural = None
        self.cache: dict[str, tuple[str, ...]] = {}

    # -- neural model, loaded on first use ------------------------------------------------------------------
    def _g2p_en(self):
        if self._neural is None:
            from g2p_en import G2p

            self._neural = G2p()
        return self._neural

    def _clean(self, phones) -> list[str]:
        out = []
        for p in phones:
            p = strip_stress(p)
            if p in P2I:
                out.append(p)
        return out

    def _lookup(self, w: str) -> list[str]:
        if w in SPECIAL:
            return list(SPECIAL[w])
        prons = self.cmu.get(w)
        if prons:
            return self._clean(prons[0])
        if "-" in w and w.strip("-"):
            out: list[str] = []
            for part in w.split("-"):
                if part:
                    out.extend(self._lookup(part))
            if out:
                return out
        if any(ch.isdigit() for ch in w) and self.use_neural:
            return self._clean(p for p in self._g2p_en()(w) if p.strip())
        letters = _NOT_LETTER.sub("", w)
        if not letters:
            return []
        if letters.endswith("'s") and letters[:-2] in self.cmu:  # possessive of a known word
            base = self._clean(self.cmu[letters[:-2]][0])
            last = base[-1] if base else ""
            return base + (["ah", "z"] if last in {"s", "z", "sh", "zh", "ch", "jh"} else
                           ["s"] if last in {"p", "t", "k", "f", "th"} else ["z"])
        if self.use_neural:
            return self._clean(self._g2p_en().predict(letters.replace("'", "")))
        return []

    def word(self, w: str) -> list[str]:
        key = _STRIP.sub("", w.lower())
        hit = self.cache.get(key)
        if hit is None:
            hit = tuple(self._lookup(key)) if key else ()
            self.cache[key] = hit
        return list(hit)

    def words(self, words: list[str]) -> list[list[str]]:
        return [self.word(w) for w in words]

    # -- persistence ----------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: list(v) for k, v in self.cache.items()}))

    def load_cache(self, path: Path) -> None:
        if Path(path).exists():
            for k, v in json.loads(Path(path).read_text()).items():
                self.cache[k] = tuple(v)
