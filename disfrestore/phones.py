"""Phone inventory shared by every stage of the pipeline.

Everything is mapped onto the 39 phones of CMUdict/ARPAbet (lower case, no stress):

* the Switchboard (MS98) dictionary uses the same set plus ``ax el en`` -> :func:`norm_swb_phone`
* the phone recogniser emits IPA (espeak style, e.g. ``facebook/wav2vec2-xlsr-53-espeak-cv-ft``) or
  TIMIT phones -> :func:`ipa_to_arpa`, :func:`timit_to_arpa`
* G2P gives ARPAbet with stress digits -> :func:`strip_stress`

It also provides ``SUB`` (a 39x39 articulatory-feature substitution cost matrix used by the phone aligner and by the
recogniser simulator) and a few phone classes.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

PHONES: list[str] = ("aa ae ah ao aw ay b ch d dh eh er ey f g hh ih iy jh k l m n ng ow oy p r s sh t th uh uw v w y z "
                     "zh").split()
P2I: dict[str, int] = {p: i for i, p in enumerate(PHONES)}
N_PHONES = len(PHONES)

# --------------------------------------------------------------------------------------------------------------
# Articulatory features
# --------------------------------------------------------------------------------------------------------------
# vowels: (height 0=high..1=low, backness 0=front..1=back, rounded)
_V = {
    "iy": (0.00, 0.00, 0), "ih": (0.20, 0.15, 0), "eh": (0.50, 0.10, 0), "ae": (0.85, 0.00, 0),
    "ah": (0.55, 0.50, 0), "aa": (1.00, 0.90, 0), "ao": (0.70, 1.00, 1), "uh": (0.20, 0.85, 1),
    "uw": (0.00, 1.00, 1), "er": (0.50, 0.50, 0),
}
_V["ey"] = tuple((a + b) / 2 for a, b in zip(_V["eh"], _V["ih"]))
_V["ay"] = tuple((a + b) / 2 for a, b in zip(_V["aa"], _V["ih"]))
_V["oy"] = tuple((a + b) / 2 for a, b in zip(_V["ao"], _V["ih"]))
_V["aw"] = tuple((a + b) / 2 for a, b in zip(_V["aa"], _V["uh"]))
_V["ow"] = tuple((a + b) / 2 for a, b in zip((0.45, 1.0, 1), _V["uh"]))
VOWELS = frozenset(_V)

# consonants: (manner, place, voiced); manner 0 stop 1 affricate 2 fricative 3 nasal 4 liquid 5 glide
_C = {
    "p": (0, 0, 0), "b": (0, 0, 1), "t": (0, 3, 0), "d": (0, 3, 1), "k": (0, 6, 0), "g": (0, 6, 1),
    "ch": (1, 4, 0), "jh": (1, 4, 1),
    "f": (2, 1, 0), "v": (2, 1, 1), "th": (2, 2, 0), "dh": (2, 2, 1), "s": (2, 3, 0), "z": (2, 3, 1),
    "sh": (2, 4, 0), "zh": (2, 4, 1), "hh": (2, 7, 0),
    "m": (3, 0, 1), "n": (3, 3, 1), "ng": (3, 6, 1),
    "l": (4, 3, 1), "r": (4, 4, 1),
    "w": (5, 0, 1), "y": (5, 5, 1),
}
CONSONANTS = frozenset(_C)
assert set(_V) | set(_C) == set(PHONES), set(PHONES) ^ (set(_V) | set(_C))

# vowel <-> consonant pairs that really are close (glides and rhotic/lateral/nasal quasi-vowels)
_CROSS = {("y", "iy"): .5, ("y", "ih"): .5, ("w", "uw"): .5, ("w", "uh"): .5, ("r", "er"): .45, ("l", "uh"): .8,
          ("l", "ah"): .85, ("n", "ah"): .85, ("m", "ah"): .85, ("hh", "ah"): .8}


def feature_distance(a: str, b: str) -> float:
    """Articulatory distance in [0, 1]; 0 only for identical phones."""
    if a == b:
        return 0.0
    va, vb = a in _V, b in _V
    if va and vb:
        (h1, b1, r1), (h2, b2, r2) = _V[a], _V[b]
        return min(0.9, float(np.sqrt((h1 - h2) ** 2 + (b1 - b2) ** 2 + (0.5 * (r1 - r2)) ** 2)) / 1.1 + 0.05)
    if va != vb:
        c, v = (b, a) if va else (a, b)
        return _CROSS.get((c, v), 1.0)
    (m1, p1, v1), (m2, p2, v2) = _C[a], _C[b]
    manner = min(1.0, abs(m1 - m2) / 4.0)
    return min(1.0, 0.45 * abs(p1 - p2) / 7.0 + 0.35 * manner + 0.20 * abs(v1 - v2) + 0.05)


# substitution cost used by the aligner: even the closest pair costs something, the worst pair stays below del+ins
SUB_MIN, SUB_SPAN = 0.25, 0.95
DIST = np.zeros((N_PHONES, N_PHONES), dtype=np.float32)
for _i, _a in enumerate(PHONES):
    for _j, _b in enumerate(PHONES):
        DIST[_i, _j] = feature_distance(_a, _b)
SUB = np.where(DIST == 0, 0.0, SUB_MIN + SUB_SPAN * DIST).astype(np.float32)


def is_vowel(p: str) -> bool:
    return p in _V


# --------------------------------------------------------------------------------------------------------------
# Normalising phone strings from other sources
# --------------------------------------------------------------------------------------------------------------
_STRESS = re.compile(r"[012]$")


def strip_stress(p: str) -> str:
    return _STRESS.sub("", p.lower())


def norm_swb_phone(p: str) -> list[str]:
    """Switchboard dictionary phone -> CMU39 phones (``ax`` is a schwa, ``el``/``en`` are syllabic l/n)."""
    p = p.lower()
    if p == "ax":
        return ["ah"]
    if p == "el":
        return ["ah", "l"]
    if p == "en":
        return ["ah", "n"]
    if p in P2I:
        return [p]
    return []


# eSpeak/IPA symbol -> ARPAbet phones.  Longest match first (see ipa_to_arpa).  Length, stress, aspiration, nasalisation and
# other diacritics are dropped.  Retroflex consonants of Indian English (ʈ ɖ ɳ ʂ ɽ) go to their alveolar counterparts.
_IPA = {
    # vowels
    "i": ["iy"], "iː": ["iy"], "ɪ": ["ih"], "e": ["ey"], "eː": ["ey"], "ɛ": ["eh"], "æ": ["ae"], "a": ["aa"],
    "ɑ": ["aa"], "ɒ": ["aa"], "ɔ": ["ao"], "o": ["ow"], "oː": ["ow"], "ʊ": ["uh"], "u": ["uw"], "ʌ": ["ah"],
    "ə": ["ah"], "ɐ": ["ah"], "ɚ": ["er"], "ɝ": ["er"], "ɜ": ["er"], "ɵ": ["ah"], "ɘ": ["ah"], "ʉ": ["uw"],
    "ɨ": ["ih"], "ø": ["er"], "œ": ["er"], "y": ["uw"], "ɯ": ["uw"], "ɤ": ["ah"], "ᵻ": ["ih"], "ᵿ": ["uh"],
    "aɪ": ["ay"], "aʊ": ["aw"], "eɪ": ["ey"], "oʊ": ["ow"], "əʊ": ["ow"], "ɔɪ": ["oy"], "ɪə": ["ih", "ah"],
    "ɛə": ["eh", "r"], "ʊə": ["uh", "ah"], "ɑɪ": ["ay"], "ɑʊ": ["aw"], "ɔə": ["ao", "r"], "ai": ["ay"], "au": ["aw"],
    "ei": ["ey"], "ou": ["ow"], "oi": ["oy"],
    # consonants
    "p": ["p"], "b": ["b"], "t": ["t"], "d": ["d"], "k": ["k"], "g": ["g"], "ɡ": ["g"], "c": ["k"], "ɟ": ["g"],
    "q": ["k"], "ʈ": ["t"], "ɖ": ["d"], "ʔ": [], "tʃ": ["ch"], "dʒ": ["jh"], "ʧ": ["ch"], "ʤ": ["jh"],
    "f": ["f"], "v": ["v"], "θ": ["th"], "ð": ["dh"], "s": ["s"], "z": ["z"], "ʃ": ["sh"], "ʒ": ["zh"], "ʂ": ["sh"],
    "ʐ": ["zh"], "ç": ["sh"], "x": ["k"], "ɣ": ["g"], "h": ["hh"], "ɦ": ["hh"], "ħ": ["hh"], "β": ["v"], "ʋ": ["v"],
    "m": ["m"], "n": ["n"], "ŋ": ["ng"], "ɳ": ["n"], "ɲ": ["n"], "ɱ": ["m"], "l": ["l"], "ɫ": ["l"], "ɭ": ["l"],
    "ʎ": ["l"], "ɹ": ["r"], "r": ["r"], "ɾ": ["d"], "ɽ": ["r"], "ɻ": ["r"], "ʀ": ["r"], "ʁ": ["r"],
    "w": ["w"], "j": ["y"], "ʍ": ["w"], "ɥ": ["w"],
}
_IPA_DROP = set("ːˑˈˌ.ʰʲʷˠˤ̩̯̥̃̊̚ʼ̪̺̻‿͜͡ᵊ|‖ ")
_IPA_MAX = max(len(k) for k in _IPA)


def _clean_ipa(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = re.sub("([nlm])\u0329", "ə\\1", s)  # syllabic consonant -> schwa + consonant
    s = unicodedata.normalize("NFC", "".join(ch for ch in s if ch not in _IPA_DROP and not unicodedata.combining(ch)))
    return s


def ipa_to_arpa(ipa: str, unknown: list[str] | None = None) -> list[str]:
    """IPA string (or one recogniser token) -> ARPAbet phones.  Symbols with no mapping are appended to ``unknown``."""
    s = _clean_ipa(ipa)
    out: list[str] = []
    i = 0
    while i < len(s):
        for n in range(min(_IPA_MAX, len(s) - i), 0, -1):
            piece = s[i:i + n]
            if piece in _IPA:
                out.extend(_IPA[piece])
                i += n
                break
        else:
            if unknown is not None:
                unknown.append(s[i])
            i += 1
    return out


_TIMIT = {
    "ix": "ih", "ax": "ah", "ax-h": "ah", "axr": "er", "ux": "uw", "dx": "d", "nx": "n", "eng": "ng", "hv": "hh",
    "em": "m", "en": "n", "el": "l", "zh": "zh", "ch": "ch", "jh": "jh", "ao": "ao", "aa": "aa", "ah": "ah",
}
_TIMIT_DROP = {"h#", "pau", "epi", "q", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl", "sil", "sp", "spn", "<pad>", "|", ""}


def timit_to_arpa(tok: str) -> list[str]:
    tok = tok.lower()
    if tok in _TIMIT_DROP:
        return []
    tok = _TIMIT.get(tok, tok)
    return [tok] if tok in P2I else []
