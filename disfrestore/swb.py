"""Reader for the free MSU/ISIP Switchboard release (OpenSLR resource 5, ``switchboard_word_alignments.tar.gz``).

Layout, one folder per conversation (two speakers, sides ``A`` and ``B``)::

    swb_ms98_transcriptions/20/2001/sw2001A-ms98-a-trans.text   segment level:  id  start  end  text
                                    sw2001A-ms98-a-word.text    word level:     id  start  end  token
    swb_ms98_transcriptions/sw-ms98-dict.text                   pronunciation dictionary (incl. partial words)

The word file lists, per segment id, every word *and* every ``[silence]`` with times, so pauses are known exactly.
Fields are separated by whitespace (some files use tabs), hence ``split()``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .phones import norm_swb_phone

DEFAULT_ROOT = Path("data/MSU Switchboard/swb_ms98_transcriptions")

FILLED_PAUSES = {"uh", "um", "hm", "hum", "ah", "eh", "mm"}
BACKCHANNELS = {"uh-huh", "um-hum", "huh-uh", "hum-um", "uh-hum", "mm-hm", "mhm"}
OTHER_HESITATIONS = {"huh", "ooh", "uh-oh"}
VARIANT = re.compile(r"_\d+$")


def classify(token: str) -> str:
    """Annotation category of one transcript token (same rules as ``switchboard_analysis.ipynb``)."""
    if token == "[silence]":
        return "silence"
    if token == "[noise]":
        return "noise"
    if token == "[vocalized-noise]":
        return "vocalized_noise"
    if token == "[laughter]":
        return "laughter"
    if token.startswith("[laughter-"):
        return "laughter_word"  # [laughter-yeah]: a word spoken while laughing
    if token in ("<b_aside>", "<e_aside>"):
        return "aside_marker"
    if token.startswith("{") and token.endswith("}"):
        return "coinage"
    if token.startswith("[") and token.endswith("]") and "/" in token:
        return "mispronounced"  # [as spoken/intended word]
    if ("[" in token or "]" in token) and (token.endswith("-") or token.startswith("-")):
        return "partial_word"  # y[ou]-   -[o]kay : spoken part outside the brackets
    if token.endswith("-") or token.startswith("-"):
        return "cutoff_word"  # i-
    lower = token.lower()
    if lower in FILLED_PAUSES:
        return "filled_pause"
    if lower in BACKCHANNELS:
        return "backchannel"
    if lower in OTHER_HESITATIONS:
        return "other_hesitation"
    if VARIANT.search(token):
        return "pron_variant"
    return "word"


NON_SPEECH = {"silence", "noise", "vocalized_noise", "laughter", "aside_marker"}


@dataclass(slots=True)
class Tok:
    text: str
    start: float
    end: float
    cat: str = ""

    def __post_init__(self):
        if not self.cat:
            self.cat = classify(self.text)


@dataclass(slots=True)
class Segment:
    id: str
    start: float
    end: float
    text: str  # as in the *-trans.text file
    toks: list[Tok] = field(default_factory=list)  # word-level entries incl. [silence]

    def speech(self) -> list[Tok]:
        return [t for t in self.toks if t.cat != "silence"]


@dataclass(slots=True)
class Side:
    conv: str  # sw2001
    side: str  # A / B
    segments: list[Segment]

    @property
    def name(self) -> str:
        return f"{self.conv}{self.side}"


def conv_dir(root: Path, conv: str) -> Path:
    n = conv[2:]
    return Path(root) / n[:2] / n


def list_conversations(root: Path = DEFAULT_ROOT) -> list[str]:
    return sorted(f"sw{p.name}" for p in Path(root).glob("*/*") if p.is_dir() and p.name.isdigit())


def split_of(conv: str) -> str:
    """Conventional Switchboard disfluency split (Charniak & Johnson 2001 / Zayats et al. 2016): test = sw4004-sw4153,
    dev = sw4519-sw4936, everything else is training."""
    n = int(conv[2:])
    if 4004 <= n <= 4153:
        return "test"
    if 4519 <= n <= 4936:
        return "dev"
    return "train"


def read_side(root: Path, conv: str, side: str) -> Side:
    d = conv_dir(root, conv)
    trans = d / f"{conv}{side}-ms98-a-trans.text"
    words = d / f"{conv}{side}-ms98-a-word.text"
    segs: dict[str, Segment] = {}
    order: list[str] = []
    for line in trans.read_text(errors="replace").splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        sid, a, b = parts[:3]
        text = parts[3] if len(parts) > 3 else ""
        segs[sid] = Segment(sid, float(a), float(b), text)
        order.append(sid)
    for line in words.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        sid, a, b, tok = parts
        seg = segs.get(sid)
        if seg is not None:
            seg.toks.append(Tok(tok, float(a), float(b)))
    return Side(conv, side, [segs[i] for i in order])


# --------------------------------------------------------------------------------------------------------------
# Dictionary
# --------------------------------------------------------------------------------------------------------------
class SwbDict:
    """``sw-ms98-dict.text``: ``token  phone phone ...`` (lines may repeat a token for alternate pronunciations).

    Partial words carry the pronunciation of the *spoken* part only, e.g. ``-[be]cau[se]- k ah``.  Phones are returned
    in the CMU39 inventory (``ax`` -> ``ah``, ``el`` -> ``ah l``, ``en`` -> ``ah n``)."""

    def __init__(self, path: Path):
        self.prons: dict[str, list[tuple[str, ...]]] = {}
        self._lower: dict[str, list[tuple[str, ...]]] = {}
        for line in Path(path).read_text(errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith(("#", "file:")):
                continue
            parts = line.split()
            word, phones = parts[0], parts[1:]
            ph: list[str] = []
            for p in phones:
                ph.extend(norm_swb_phone(p))
            pron = tuple(ph)
            lst = self.prons.setdefault(word, [])
            if pron not in lst:
                lst.append(pron)
            low = self._lower.setdefault(word.lower(), [])
            if pron not in low:
                low.append(pron)

    def __contains__(self, token: str) -> bool:
        return token in self.prons or token.lower() in self._lower

    def lookup(self, token: str) -> list[tuple[str, ...]]:
        return self.prons.get(token) or self._lower.get(token.lower()) or []

    @classmethod
    def load(cls, root: Path = DEFAULT_ROOT) -> "SwbDict":
        return cls(Path(root) / "sw-ms98-dict.text")
