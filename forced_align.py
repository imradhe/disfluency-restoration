#!/usr/bin/env python3
"""Word-level forced alignment for RSML transcripts.

Reads an audio file and an RSML transcript (SRT-like segments with a metadata line) and writes a new
RSML file where every segment is followed by its words, each as a timestamp line, a metadata line and the
word itself, indented with a tab:

    1
    00:00:14,456 --> 00:00:18,685
    primary=1|verified=1|flagged=0|note=
    In this lecture, we will start a new topic that is on stereo geometry.
    <TAB>00:00:14,820 --> 00:00:14,937
    <TAB>confidence=1.000|weakest_letter_confidence=1.000|voiced_fraction=1.00|pause_after_milliseconds=0|
    <TAB>pause_voiced_fraction=0.00|kind=word|script=latin|flagged=0     (one line in the file)
    <TAB>In
    ...

Method: each segment's audio window (segment +/- --pad seconds) is run through a wav2vec2 CTC model and
the transcript is aligned to the frame posteriors with a CTC Viterbi pass. Word edges are then stretched
over the voiced audio next to them (CTC peaks sit inside a sound, which truncates hesitations and prolonged
words); --no-refine turns that off.

Word units are the whitespace-separated tokens of the transcript, except that an annotation such as
`!en[మల్టిపుల్ న్యూక్లియై](multiple nuclei)` stays one unit even though it contains spaces.

Language comes from the RSML itself: a segment's `primary=N` names its speaker, and `[speakers]` gives that
speaker's *spoken* language (`1 = male, te`); --lang overrides it. A segment written entirely in Latin letters under a
non-English speaker is code-mixed speech and uses `[settings] default_code_mixing_language` (e.g. en). English
is aligned with facebook/wav2vec2-base-960h on the letters as written. Every other language is transliterated
into Harvard-Kyoto (HK) and aligned with Meta's multilingual MMS aligner, which works on Latin letters (Devanagari,
Bengali/Assamese, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam scripts). HK writes distinctions in
case (A = long a, T = retroflex t) and the model only has 26 lowercase letters, so HK symbols are then spelled the
way the model's training text spells them (A -> aa, T -> tt, z -> sh, ...). Language-specific rules: Hindi-type
languages (hi, ne, pa, gu) drop the unpronounced word-final "a". Latin words inside an Indian-language transcript
pass through unchanged, so code-mixed Telugu-English text works. LANG_MODELS sets a model per language. The MMS
model is licensed CC-BY-NC-4.0 (non-commercial).

Tags (the lists are in the RSML [tags] block):
  hesitation  @uhh, @umm, ... are sounds the speaker made: aligned acoustically, so they have a real duration
  isolated    @breathe, @laughter, @short-pause, @silence, ... are sounds or silences with no letters: they get the
              pause between the neighbouring words (shared if several); zero-length if the words touch
  span        @repetition-start / -end, @false-start-start / -end, ... mark a stretch of words and have no duration:
              an -end sits where the previous word ends, a -start where the next word starts
Other annotations: #NUM[two](2), [cameya](camera), !en[..](..) -> the bracketed *verbatim* form is aligned, the (normalized
form) is not. Blocks that are not segments (the config trailer) are copied through unchanged.

Word metadata (--word-meta), a `key=value|key=value` line like the segment metadata line:
    confidence                 mean probability the model gives the word's letters on the alignment path (0-1); low =
                               the audio does not match this word. Empty for tags, for hesitations (the model has no
                               sound for "uh", so the score would always be ~0) and for fallback segments.
    weakest_letter_confidence  the same for the single weakest letter (catches one wrong letter in a good word)
    voiced_fraction            fraction of the word's time span that contains speech energy (independent of the model)
    pause_after_milliseconds   time until the next spoken word
    pause_voiced_fraction      fraction of that pause that contains speech, i.e. audio no word accounts for (an omitted
                               word, a breath); catches what confidence cannot: a missing word next to a hesitation
    kind                       word | hesitation | isolated | span | normalized | punctuation | untransliterated
    script                     script of the spoken letters (telugu, latin, ...): shows code-switching
    flagged                    1 if confidence < --flag-confidence (default 0.3), else 0

Requirements: pip install torch transformers numpy   (+ ffmpeg on PATH)
              optional: num2words (spell out digits), uroman (Urdu/Ol Chiki/Meetei Mayek, which HK cannot express)
"""
from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MIN_SAMPLES = 1_600  # wav2vec2 needs a few frames of context; below 0.1 s we fall back to even spreading
HOP = 0.01  # seconds per energy frame used to refine word boundaries
HOP_SAMPLES = int(HOP * SAMPLE_RATE)
INDENT = "\t"
NEG = -1e30

TS_RE = re.compile(r"^\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*$")
META_RE = re.compile(r"^\w+=[^|]*(\|\w+=[^|]*)*$")
# Prefixes seen: #NUM[..](..) #PER #ITEM #DATETIME, $[..](..), !en[..](..) !bio[..](..) !!bio[..](..), or none.
ANNOTATION_RE = re.compile(r"(?:[#$!][#$!A-Za-z]*)?\[(?P<verbatim>[^\]]*)\]\((?P<normalized>[^)]*)\)")
TAG_RE = re.compile(r"(?<![A-Za-z0-9])@[A-Za-z]+(?:-[A-Za-z]+)*")
DEFAULT_HESITATIONS = ("umm", "uhh", "hmm", "ugh", "huh", "tsk", "uh-huh", "ehh", "ooh", "uh-oh")
DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿]")  # ZWJ/ZWNJ etc. sit inside Indic words

# Model per language code. Anything not listed gets the English model for English and the multilingual MMS
# aligner for every other language. Add e.g. "te": "<a Telugu CTC model with a Latin vocabulary>" to override.
LANG_MODELS: dict[str, str] = {}
ENGLISH_MODEL = "facebook/wav2vec2-base-960h"
MMS_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"  # MMS_FA from torchaudio, converted to HF
# ISO 639-1 -> 639-3 for the languages in the RSML [languages] block (other 3-letter codes pass through).
ISO3 = {
    "en": "eng", "as": "asm", "bn": "ben", "brx": "brx", "doi": "doi", "gu": "guj", "hi": "hin", "kn": "kan",
    "kok": "kok", "ks": "kas", "mai": "mai", "ml": "mal", "mni": "mni", "mr": "mar", "ne": "nep", "or": "ory",
    "pa": "pan", "sa": "san", "sat": "sat", "sd": "snd", "ta": "tam", "te": "tel", "ur": "urd",
}
ISO2 = {v: k for k, v in ISO3.items()}

try:
    from num2words import num2words
except ImportError:
    num2words = None


# ---------------------------------------------------------------------------------------------------
# RSML parsing / writing
# ---------------------------------------------------------------------------------------------------

@dataclass
class Segment:
    header: list[str]  # segment id, timestamps and (if present) metadata lines, verbatim
    text_lines: list[str]
    start: float
    end: float


@dataclass
class Token:
    text: str  # verbatim whitespace-separated token from the transcript
    words: list[str]  # normalised words to align; empty for tags / punctuation
    dropped: bool = False  # had letters but none the model can use (a script that could not be transliterated)
    kind: str = ""  # word | hesitation | isolated | span | normalized | punctuation | untransliterated
    script: str = ""  # script of the spoken letters, e.g. telugu / latin


def to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def format_ts(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def read_blocks(path: Path) -> list[list[str]]:
    blocks, current = [], []
    for line in path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").split("\n"):
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_block(block: list[str]) -> Segment | None:
    """Return a Segment for `id / timestamps / [metadata] / transcript` blocks, None for anything else."""
    if len(block) < 3 or not block[0].strip().isdigit():
        return None
    match = TS_RE.match(block[1])
    if not match:
        return None
    # Indented lines are word-level lines from an earlier run; drop them and regenerate.
    body = [line for line in block[2:] if line[:1] not in " \t"]
    header = block[:2]
    if body and META_RE.match(body[0]):
        header.append(body.pop(0))
    return Segment(header, body, to_seconds(*match.groups()[:4]), to_seconds(*match.groups()[4:]))


@dataclass
class Config:
    """What the RSML's own config blocks declare."""
    hesitations: set[str]  # [tags] hesitations = @umm, @uhh, ...
    default_language: str  # [settings] default_language = te (older files; speakers carry the language now)
    code_mixing_language: str  # [settings] default_code_mixing_language = en: what the speaker mixes into the speech
    speakers: dict[str, str]  # [speakers] "1 = male, te" -> {"1": "te"}; a segment's `primary=N` names its speaker


def read_config(config_blocks: list[list[str]]) -> Config:
    cfg, section = Config(set(DEFAULT_HESITATIONS), "", "", {}), ""
    for block in config_blocks:
        for line in (l.strip() for l in block):
            if not line or line.startswith("#"):
                continue
            if header := re.fullmatch(r"\[(\w+)\]", line):
                section = header.group(1)
                continue
            key, sep, value = (x.strip() for x in line.partition("="))
            if not sep:
                continue
            if section == "tags" and key == "hesitations":
                cfg.hesitations = {t.strip().lstrip("@").lower() for t in value.split(",") if t.strip()}
            elif section == "settings" and key == "default_language":
                cfg.default_language = value.lower()
            elif section == "settings" and key == "default_code_mixing_language":
                cfg.code_mixing_language = value.lower()
            elif section == "speakers":
                fields = [f.strip() for f in value.split(",")]  # gender, language
                cfg.speakers[key] = fields[1].lower() if len(fields) > 1 else ""
    return cfg


def script_of(text: str) -> str:
    """"latin" if every letter of the text (ignoring @tags) is Latin script, "none" if it has no letters, else "other"."""
    letters = [c for c in TAG_RE.sub("", text) if c.isalpha()]
    if not letters:
        return "none"
    return "latin" if all(unicodedata.name(c, "").startswith("LATIN") for c in letters) else "other"


def resolve_language(seg: Segment, cfg: Config, override: str, notes: collections.Counter) -> str:
    """A segment's language: --lang if given, else its speaker's spoken language from [speakers] (via the segment's
    `primary=N`), else [settings] default_language, else guessed from the script. A segment written entirely in
    Latin letters under a non-English speaker is code-mixed speech, so it uses default_code_mixing_language when the
    file declares one. Contradictions between the declaration and the text are counted in `notes`."""
    if override != "auto":
        return override.lower()
    meta = seg.header[2] if len(seg.header) > 2 else ""
    speaker = re.search(r"(?:^|\|)primary=(\w+)", meta)
    lang = (cfg.speakers.get(speaker.group(1), "") if speaker else "") or cfg.default_language
    script = script_of(" ".join(seg.text_lines))
    if not lang:
        notes[f"no language declared for speaker {speaker.group(1) if speaker else '?'}: guessed from the script"] += 1
        return "en" if script != "other" else "und"
    english = ISO3.get(lang, lang) == "eng"
    if english and script == "other":
        notes[f"declared '{lang}' but the text has non-Latin letters: used the multilingual path"] += 1
        return "und"
    if not english and script == "latin":
        if cfg.code_mixing_language:
            notes[f"all Latin script under a '{lang}' speaker: aligned as the code-mixing language '{cfg.code_mixing_language}'"] += 1
            return cfg.code_mixing_language
        notes[f"declared '{lang}' but the text is all Latin script (English?): kept '{lang}'"] += 1
    return lang


# ---------------------------------------------------------------------------------------------------
# Transcript -> alignable words
# ---------------------------------------------------------------------------------------------------

# Brahmic scripts -> Harvard-Kyoto (HK). The nine blocks below all follow the same ISCII layout (क / క / க are
# each at offset 0x15 in their block), so one table keyed by that offset serves every script.
BRAHMIC_BLOCKS = {0x0900: "dev", 0x0980: "ben", 0x0A00: "pan", 0x0A80: "guj", 0x0B00: "ori",
                  0x0B80: "tam", 0x0C00: "tel", 0x0C80: "kan", 0x0D00: "mal"}  # Hindi/Marathi/Nepali/..., Bengali/Assamese, ...
HK_VOWELS = {0x05: "a", 0x06: "A", 0x07: "i", 0x08: "I", 0x09: "u", 0x0A: "U", 0x0B: "R", 0x0C: "lR", 0x0D: "e",
             0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "RR", 0x61: "lRR"}
HK_MATRAS = {0x3E: "A", 0x3F: "i", 0x40: "I", 0x41: "u", 0x42: "U", 0x43: "R", 0x44: "RR", 0x45: "e", 0x46: "e",
             0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "lR", 0x63: "lRR"}
HK_CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "G", 0x1A: "c", 0x1B: "ch", 0x1C: "j",
                 0x1D: "jh", 0x1E: "J", 0x1F: "T", 0x20: "Th", 0x21: "D", 0x22: "Dh", 0x23: "N", 0x24: "t",
                 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
                 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "L", 0x34: "L",
                 0x35: "v", 0x36: "z", 0x37: "S", 0x38: "s", 0x39: "h", 0x5C: "D", 0x5D: "Dh", 0x5F: "y"}
HK_SIGNS = {0x01: "M", 0x02: "M", 0x03: "H"}  # candrabindu, anusvara, visarga
# Per-script exceptions to the shared table. HK has no symbol for Tamil/Malayalam zh (ழ ഴ), so that is spelled "zh".
HK_OVERRIDES = {"tam": {0x34: "zh"}, "mal": {0x34: "zh"}, "tel": {0x58: "c", 0x59: "j", 0x5A: "r"},
                "kan": {0x5E: "L"}, "ben": {0x70: "r", 0x71: "v"}, "ori": {0x71: "v"}}
HK_CHILLU = {0x7A: "N", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "L", 0x7F: "k"}  # Malayalam: consonant, no inherent a
HK_NUKTA = {"k": "q", "j": "z", "ph": "f"}  # consonant + nukta; for every other consonant the nukta is dropped
# HK separates letters by case (A long a, T retroflex t, z/S the two sibilants), but the MMS aligner only has the 26
# lowercase letters and was trained on uroman-style spellings, so each HK symbol is rewritten the way that
# convention writes it. Symbols not listed are already plain lowercase letters (k, kh, a, e, ai, ...).
HK_TO_MODEL = {"A": "aa", "I": "ii", "U": "uu", "R": "ri", "RR": "rii", "lR": "li", "lRR": "lii", "M": "m", "H": "h",
               "G": "ng", "J": "ny", "T": "tt", "Th": "tth", "D": "dd", "Dh": "ddh", "N": "nn", "L": "ll",
               "z": "sh", "S": "ss"}


def to_model_alphabet(hk: str) -> str:
    return HK_TO_MODEL.get(hk, hk.lower())


# Languages that do not pronounce the inherent "a" at the end of a word (HK is orthographic and would keep it).
SCHWA_DELETING = {"hin", "nep", "pan", "guj"}


def indic_to_hk(text: str, spell=lambda piece: piece, drop_final_schwa: bool = False) -> str:
    """Harvard-Kyoto transliteration of every Brahmic-script character in `text` (NFC input); anything else,
    e.g. Latin words in code-mixed text, passes through unchanged. `spell` rewrites each HK symbol as it is
    emitted (identity = pure HK, e.g. స్వాగతం -> svAgataM). `drop_final_schwa` omits the unpronounced word-final
    inherent "a" of Hindi-type languages (आज -> Aj instead of Aja)."""
    out, i, geminate, run_start = [], 0, False, 0
    while i < len(text):
        o = ord(text[i])
        script = BRAHMIC_BLOCKS.get(o & ~0x7F)
        if script is None:
            out.append(text[i])
            i += 1
            run_start = i
            continue
        base, off, start = o & ~0x7F, o & 0x7F, i
        i += 1
        if script == "mal" and off in HK_CHILLU:
            out.append(spell(HK_CHILLU[off]))
        elif script == "ben" and off == 0x4E:  # khanda ta ৎ
            out.append(spell("t"))
        elif off in HK_OVERRIDES.get(script, ()) or off in HK_CONSONANTS:
            cons = HK_OVERRIDES.get(script, {}).get(off) or HK_CONSONANTS[off]
            if i < len(text) and ord(text[i]) == base + 0x3C:  # nukta
                cons = HK_NUKTA.get(cons, cons)
                i += 1
            piece = spell(cons) * (2 if geminate else 1)  # Gurmukhi addak doubles the consonant after it
            geminate = False
            nxt = ord(text[i]) - base if i < len(text) and ord(text[i]) & ~0x7F == base else -1
            if nxt == 0x4D:  # virama: no inherent vowel
                out.append(piece)
                i += 1
            elif nxt in HK_MATRAS:
                out.append(piece + spell(HK_MATRAS[nxt]))
                i += 1
            elif drop_final_schwa and nxt == -1 and start > run_start:  # last letter of a multi-letter word
                out.append(piece)
            else:
                out.append(piece + spell("a"))
        elif off in HK_VOWELS:
            out.append(spell(HK_VOWELS[off]))
        elif off in HK_SIGNS or (script == "pan" and off == 0x70):  # tippi
            out.append(spell(HK_SIGNS.get(off, "M")))
        elif script == "pan" and off == 0x71:  # addak
            geminate = True
        elif script == "dev" and off == 0x50:  # om
            out.append(spell("o") + spell("M"))
        # remaining Brahmic characters (stray viramas, nuktas, avagraha, danda, length marks) carry no sound
    return "".join(out)


def strip_marks(text: str) -> str:
    """Drop accents: 'naïve' -> 'naive'. Only safe on Latin text, it also deletes Indic viramas and nuktas."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


class Normaliser:
    """Turns the verbatim part of a transcript token into the letters the model knows.

    English: letters as written. Any other language: every Indic-script character is transliterated into
    Harvard-Kyoto and then spelled in the model's lowercase alphabet (translit="hk"); Latin words in code-mixed
    text pass through. translit="hk-plain" just lowercases the raw HK, translit="uroman" uses uroman instead.
    Scripts HK cannot express (Urdu/Perso-Arabic, Ol Chiki, Meetei Mayek) fall back to uroman if installed."""

    def __init__(self, hesitations: set[str], case, valid: set[str], lang: str = "en", translit: str = "hk"):
        self.hesitations, self.case, self.valid, self.translit = hesitations, case, valid, translit
        self.lcode = ISO3.get(lang.lower(), lang.lower())  # ISO 639-3
        self.lang2 = ISO2.get(self.lcode, self.lcode)  # num2words uses 2-letter codes
        self.english = self.lcode == "eng"
        self._uroman = None
        if translit == "uroman" and not self.english:
            self._load_uroman()

    def _load_uroman(self):
        if self._uroman is None:
            try:
                import uroman
            except ImportError:
                sys.exit("error: this script/option needs uroman: pip install uroman")
            self._uroman = uroman.Uroman()
        return self._uroman

    def _romanize(self, raw: str) -> str:
        if self.translit == "uroman":
            return self._load_uroman().romanize_string(raw, lcode=self.lcode)
        raw = indic_to_hk(raw, to_model_alphabet if self.translit == "hk" else lambda piece: piece,
                          drop_final_schwa=self.lcode in SCHWA_DELETING)
        if any(ord(c) > 127 and c.isalpha() and not unicodedata.combining(c) for c in raw):  # non-Brahmic script left
            try:
                import uroman  # noqa: F401
            except ImportError:
                return raw  # reported as untransliterated by the caller
            raw = self._load_uroman().romanize_string(raw, lcode=self.lcode)
        return raw

    def _tag_sound(self, match: re.Match) -> str:
        tag = match.group()[1:].lower()
        if tag not in self.hesitations:
            return " "
        # "uhh" -> "uh", "umm" -> "um", "hmm" -> "hm": a doubled consonant would need an extra blank frame between
        # the two in CTC. Vowels are left alone, so "ooh" stays "ooh".
        sound = re.sub(r"([^aeiou-])\1+", r"\1", tag).replace("-", " ")
        return f" {sound} "

    def _spell_number(self, match: re.Match) -> str:
        n = int(match.group())  # \d also matches Telugu/Devanagari/... digits, int() reads them all
        if num2words:
            try:
                return f" {num2words(n, lang=self.lang2)} "
            except (NotImplementedError, OverflowError):
                pass
        if self.english:
            return " " + " ".join(DIGIT_WORDS[int(d)] for d in str(n)) + " "
        return " "  # no way to say it in this language: leave it out rather than align English words

    def _prepare(self, raw: str) -> str:
        raw = ZERO_WIDTH_RE.sub("", TAG_RE.sub(self._tag_sound, raw.replace("’", "'")))
        if self.english:
            raw = strip_marks(raw)
        return re.sub(r"\d+", self._spell_number, raw)

    def words(self, raw: str) -> list[str]:
        raw = self._prepare(raw)
        if not self.english:
            raw = strip_marks(self._romanize(unicodedata.normalize("NFC", raw)))
        chars = "".join(c if c in self.valid else " " for c in self.case(raw))
        return [w for w in (w.strip("'") for w in chars.split()) if w]


def tokenize(text: str, norm: Normaliser) -> list[Token]:
    """Split a transcript line into word units. Whitespace separates units, except that an annotation such as
    `!en[మల్టిపుల్ న్యూక్లియై](multiple nuclei)` stays one unit even though it contains spaces."""
    # Mark which characters were said: inside an annotation [verbatim](normalized) only the verbatim part is.
    said, annotations = [True] * len(text), []
    for m in ANNOTATION_RE.finditer(text):
        annotations.append((m.start(), m.end()))
        said[m.start():m.end()] = [False] * (m.end() - m.start())
        said[m.start("verbatim"):m.end("verbatim")] = [True] * (m.end("verbatim") - m.start("verbatim"))
    units, annotation_end = [], -1  # each unit: [start, end, [verbatim text of each whitespace token in it]]
    for m in re.finditer(r"\S+", text):
        start, end = m.span()
        raw = "".join(c for i, c in enumerate(m.group(), start) if said[i])
        if units and start < annotation_end:  # this token continues an annotation begun in an earlier token
            units[-1][1] = end
            units[-1][2].append(raw)
        else:
            units.append([start, end, [raw]])
        annotation_end = max([annotation_end] + [b for a, b in annotations if a < end and b > start])

    tokens = []
    for start, end, raws in units:
        raw = " ".join(raws)
        words = norm.words(raw)
        letters = TAG_RE.sub("", raw)
        tags = TAG_RE.findall(raw)
        dropped = not words and any(c.isalpha() for c in letters)
        if words:
            kind = "word" if any(c.isalpha() for c in letters) else "hesitation"  # a bare @uhh
        elif dropped:
            kind = "untransliterated"
        elif not raw.strip():
            kind = "normalized"  # nothing but the (normalized) half of an annotation
        elif tags:
            kind = "span" if all(re.search(r"-(start|end)$", t) for t in tags) else "isolated"
        else:
            kind = "punctuation"
        scripts = collections.Counter(unicodedata.name(c, "").split(" ")[0].lower() for c in letters if c.isalpha())
        script = scripts.most_common(1)[0][0] if scripts else ("latin" if words else "")
        tokens.append(Token(text[start:end], words, dropped, kind, script))
    return tokens


def build_targets(tokens: list[Token], char_ids: dict[str, int], delimiter: int | None):
    """CTC target ids for the whole segment, plus each token's (first, last) target index (None = no audio).

    `delimiter` is the model's word-break symbol, or None for models (MMS) that have none."""
    sep = [] if delimiter is None else [delimiter]
    targets: list[int] = []
    spans: list[tuple[int, int] | None] = []
    for tok in tokens:
        if not tok.words:
            spans.append(None)
            continue
        if targets:
            targets += sep
        first = len(targets)
        for i, word in enumerate(tok.words):
            if i:
                targets += sep
            targets.extend(char_ids[c] for c in word)
        spans.append((first, len(targets) - 1))
    return targets, spans


# ---------------------------------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------------------------------

def ctc_align(log_probs: np.ndarray, targets: list[int], blank: int):
    """CTC Viterbi forced alignment.

    Returns (first_frame, last_frame) arrays, one entry per target, or None when the audio window has too
    few frames to emit the targets.
    """
    T, L = len(log_probs), len(targets)
    S = 2 * L + 1
    labels = np.full(S, blank)
    labels[1::2] = targets
    # s-2 -> s skips the blank between two labels; not allowed when they are the same label
    can_skip = np.zeros(S, bool)
    can_skip[3::2] = labels[3::2] != labels[1:S - 2:2]
    lp = log_probs[:, labels].astype(np.float64)

    dp = np.full(S, NEG)
    dp[:2] = lp[0, :2]
    back = np.zeros((T, S), np.uint8)
    cand = np.full((3, S), NEG)
    cols = np.arange(S)
    for t in range(1, T):
        cand[0] = dp
        cand[1, 1:] = dp[:-1]
        cand[2, 2:] = np.where(can_skip[2:], dp[:-2], NEG)
        best = cand.argmax(axis=0)
        back[t] = best
        dp = cand[best, cols] + lp[t]

    state = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[state] < NEG / 2:
        return None
    first = np.full(L, -1)
    last = np.full(L, -1)
    for t in range(T - 1, -1, -1):
        if state % 2:
            k = state // 2
            first[k] = t
            if last[k] < 0:
                last[k] = t
        state -= int(back[t, state])
    return first, last


class Aligner:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.model = AutoModelForCTC.from_pretrained(model_name).to(device).eval()
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        except OSError:  # repo ships no preprocessor config: fall back to plain per-utterance normalisation
            self.extractor = None

        vocab = AutoTokenizer.from_pretrained(model_name).get_vocab()
        # English model: blank is <pad> (= config.pad_token_id). MMS aligner: <pad> is a separate symbol, the
        # CTC blank is <blank>.
        self.blank = vocab["<blank>"] if "<blank>" in vocab else self.model.config.pad_token_id
        self.delimiter = vocab.get("|")  # MMS has no word-break symbol
        self.char_ids = {k: v for k, v in vocab.items() if len(k) == 1 and (k.isalpha() or k == "'")}
        self.case = str.upper if "E" in self.char_ids else str.lower

    def log_probs(self, wave: np.ndarray) -> np.ndarray:
        if self.extractor is not None:
            values = self.extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        else:
            values = self.torch.from_numpy(((wave - wave.mean()) / (wave.std() + 1e-7)).astype(np.float32))[None]
        with self.torch.inference_mode():
            logits = self.model(values.to(self.device)).logits
        return self.torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()


def align_segment(aligner: Aligner, audio: np.ndarray, seg: Segment, tokens: list[Token], pad: float):
    """(times, scores) with one entry per token, or None if alignment failed.

    times: (start, end) seconds, None for tokens without audio. scores: (confidence, weakest letter) or None, where confidence is the
    mean over the token's letters of the model posterior of that letter on the alignment path, min the weakest letter."""
    targets, spans = build_targets(tokens, aligner.char_ids, aligner.delimiter)
    if not targets:
        return [None] * len(tokens), [None] * len(tokens)
    win_start = max(0.0, seg.start - pad)
    a = int(win_start * SAMPLE_RATE)
    b = min(len(audio), int((seg.end + pad) * SAMPLE_RATE))
    if b - a < MIN_SAMPLES:
        return None
    log_probs = aligner.log_probs(audio[a:b])
    aligned = ctc_align(log_probs, targets, aligner.blank)
    if aligned is None:
        return None
    first, last = aligned
    sec_per_frame = (b - a) / SAMPLE_RATE / len(log_probs)
    times, scores = [], []
    for s in spans:
        if s is None:
            times.append(None)
            scores.append(None)
            continue
        times.append((win_start + first[s[0]] * sec_per_frame, win_start + (last[s[1]] + 1) * sec_per_frame))
        letters = [k for k in range(s[0], s[1] + 1) if targets[k] != aligner.delimiter]
        per_letter = [float(np.exp(log_probs[first[k]:last[k] + 1, targets[k]]).mean()) for k in letters]
        scores.append((sum(per_letter) / len(per_letter), min(per_letter)))
    return times, scores


def spread_evenly(tokens: list[Token], seg: Segment):
    """Fallback: split the segment across tokens in proportion to their length."""
    weights = [sum(map(len, t.words)) for t in tokens]
    total = sum(weights) or 1
    out, done = [], 0
    for w in weights:
        out.append(None if w == 0 else (
            seg.start + (seg.end - seg.start) * done / total,
            seg.start + (seg.end - seg.start) * (done + w) / total,
        ))
        done += w
    return out


def voiced_frames(audio: np.ndarray) -> np.ndarray:
    """Per-10 ms voiced/unvoiced flags: frame energy above the midpoint (in dB) between this file's quiet
    floor (5th percentile) and its speech level (95th percentile)."""
    n = len(audio) // HOP_SAMPLES
    frames = audio[:n * HOP_SAMPLES].reshape(n, HOP_SAMPLES)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-8)
    low, high = np.percentile(db, [5, 95])
    return db > (low + high) / 2


def extend_into_voiced(spans, seg: Segment, voiced: np.ndarray):
    """CTC pins a word to where its letters peak, so word edges land inside the sound: hesitations and
    prolonged words come out far too short. Stretch every word over the voiced audio touching it; a gap
    that is voiced end to end is split at its midpoint, silence is left as a pause."""
    words = [list(s) for s in spans if s]
    if not words:
        return spans

    def right(t, limit):
        i = int(t / HOP)
        while i < min(int(limit / HOP), len(voiced)) and voiced[i]:
            i += 1
        return max(t, i * HOP)

    def left(t, limit):
        i = int(t / HOP) - 1
        while int(np.ceil(limit / HOP)) <= i < len(voiced) and voiced[i]:
            i -= 1
        return min(t, (i + 1) * HOP)

    words[0][0] = left(words[0][0], seg.start)
    words[-1][1] = right(words[-1][1], seg.end)
    for a, b in zip(words, words[1:]):
        end, start = right(a[1], b[0]), left(b[0], a[1])
        if end >= start:
            end = start = (a[1] + b[0]) / 2
        a[1], b[0] = end, start
    it = iter(words)
    return [next(it) if s else None for s in spans]


WORD_META_KEYS = ("confidence", "weakest_letter_confidence", "voiced_fraction", "pause_after_milliseconds",
                  "pause_voiced_fraction", "kind", "script", "flagged")


def voiced_fraction(voiced: np.ndarray, start: float, end: float) -> float:
    i = int(start / HOP)
    return float(voiced[i:max(int(end / HOP), i + 1)].mean()) if i < len(voiced) else 0.0


def pauses(tokens: list[Token], times, voiced: np.ndarray):
    """Per token: (ms until the next spoken word, fraction of that pause containing speech) or None.

    Speech inside a pause is audio no word accounts for: an omitted word, a breath, noise."""
    spoken = [i for i, (t, sp) in enumerate(zip(tokens, times)) if t.words and sp[1] > sp[0]]
    out = [None] * len(tokens)
    for i, j in zip(spoken, spoken[1:]):
        gap = max(0.0, times[j][0] - times[i][1])
        out[i] = (round(gap * 1000), voiced_fraction(voiced, times[i][1], times[j][0]) if gap > 0 else 0.0)
    return out


def word_meta(keys, tok: Token, score, span, voiced: np.ndarray, pause, flag_confidence: float) -> str:
    """One `key=value|key=value` line per word, in the style of the segment metadata line."""
    # Hesitations get no posterior score: the models were trained on transcripts without "uh", so they give the
    # letters U-H ~0 probability whether or not the alignment is right.
    score = None if tok.kind == "hesitation" else score
    timed = span[1] > span[0] and (bool(tok.words) or tok.kind == "isolated")
    values = {
        "confidence": f"{score[0]:.3f}" if score else "",
        "weakest_letter_confidence": f"{score[1]:.3f}" if score else "",
        "voiced_fraction": f"{voiced_fraction(voiced, *span):.2f}" if timed else "",
        "pause_after_milliseconds": f"{pause[0]}" if pause else "",
        "pause_voiced_fraction": f"{pause[1]:.2f}" if pause else "",
        "kind": tok.kind,
        "script": tok.script,
        "flagged": "1" if score and round(score[0], 3) < flag_confidence else "0",  # judged on the value as written
    }
    return "|".join(f"{k}={values[k]}" for k in keys)


def place_tags(tokens: list[Token], times, seg: Segment):
    """Give every token without letters its time, between the spoken words around it.

    Span tags (@x-start / @x-end), punctuation and the like have no duration: an -end sits where the previous
    word ends, a -start where the next word starts. Isolated tags (@breathe, @short-pause, @laughter, ...) are
    sounds or silences of their own, so they get the gap between the neighbouring words, shared if there are
    several. A gap of zero leaves them zero-length: there is no room to invent."""
    times = [list(t) for t in times]
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i].words:
            i += 1
            continue
        j = i
        while j < n and not tokens[j].words:
            j += 1
        # The gap runs from the end of the previous word (or the segment start) to the start of the next word (or
        # the segment end). Words may overshoot the segment edge, so never push it past the word beside it.
        next_start = times[j][0] if j < n else seg.end
        gap_start = times[i - 1][1] if i > 0 else min(seg.start, next_start)
        gap_end = max(next_start, gap_start)
        isolated = sum(t.kind == "isolated" for t in tokens[i:j])
        cursor, slot = gap_start, (gap_end - gap_start) / isolated if isolated else 0.0
        tail = j  # span-start tags at the end of the run open where the next word begins
        while tail > i and tokens[tail - 1].kind == "span" and tokens[tail - 1].text.rstrip(".,?!;:)").endswith("-start"):
            tail -= 1
        for k in range(i, j):
            if k >= tail:
                times[k] = [gap_end, gap_end]
            elif tokens[k].kind == "isolated":
                times[k] = [cursor, cursor + slot]
                cursor += slot
            else:
                times[k] = [cursor, cursor]
        i = j
    return times


def finalize(spans, seg: Segment, max_gap: float, clamp: bool) -> list[list[float]]:
    """Absorb short silences into words, optionally clamp to the segment, and place zero-length tags."""
    out = [None if s is None else list(s) for s in spans]
    aligned = [w for w in out if w]
    if aligned and max_gap > 0:
        if 0 <= aligned[0][0] - seg.start <= max_gap:
            aligned[0][0] = seg.start
        if 0 <= seg.end - aligned[-1][1] <= max_gap:
            aligned[-1][1] = seg.end
        for cur, nxt in zip(aligned, aligned[1:]):
            if 0 <= nxt[0] - cur[1] <= max_gap:
                cur[1] = nxt[0]
    if clamp:
        for w in aligned:
            w[:] = [min(max(t, seg.start), seg.end) for t in w]
    # tags/punctuation sit at the end of the previous word (or the start of the first word)
    prev_end = None
    first_start = aligned[0][0] if aligned else seg.start
    for i, w in enumerate(out):
        if w:
            prev_end = w[1]
        else:
            t = first_start if prev_end is None else prev_end
            out[i] = [t, t]
    return out


# ---------------------------------------------------------------------------------------------------
# Audio / CLI
# ---------------------------------------------------------------------------------------------------

def load_audio(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("error: ffmpeg not found on PATH (needed to decode the audio)")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: ffmpeg could not read {path}:\n{e.stderr.decode(errors='replace')}")
    return np.frombuffer(raw, dtype=np.float32)


class Pipelines:
    """One acoustic model per model name and one text normaliser per language, each loaded on first use, so a
    file whose speakers use different languages is aligned with the right model and rules for each segment."""

    def __init__(self, cfg: Config, model: str | None, device: str, translit: str):
        self.cfg, self.model, self.device, self.translit = cfg, model, device, translit
        self.aligners: dict[str, Aligner] = {}
        self.by_lang: dict[str, tuple[Aligner, Normaliser]] = {}

    def get(self, lang: str) -> tuple[Aligner, Normaliser]:
        if lang not in self.by_lang:
            english = ISO3.get(lang, lang) == "eng"
            name = self.model or LANG_MODELS.get(lang) or (ENGLISH_MODEL if english else MMS_MODEL)
            if name not in self.aligners:
                print(f"loading model: {name} (for lang={lang})", file=sys.stderr)
                self.aligners[name] = Aligner(name, self.device)
            aligner = self.aligners[name]
            norm = Normaliser(self.cfg.hesitations, aligner.case, set(aligner.char_ids), lang, self.translit)
            self.by_lang[lang] = (aligner, norm)
        return self.by_lang[lang]


def main() -> None:
    p = argparse.ArgumentParser(description="Word-level forced alignment for RSML transcripts.")
    p.add_argument("audio", help="audio (or video) file, any format ffmpeg reads")
    p.add_argument("rsml", type=Path, help="input RSML transcript")
    p.add_argument("-o", "--output", type=Path, help="output path (default: <rsml>.aligned.rsml)")
    p.add_argument("--lang", default="auto",
                   help="language code (te, hi, en, ...). By default each segment uses the language declared in the RSML:"
                        " its speaker's entry in [speakers] (the segment's primary=N), else [settings] default_language."
                        " Passing a code here overrides the declaration for every segment (default: %(default)s)")
    p.add_argument("--translit", choices=["hk", "hk-plain", "uroman"], default="hk",
                   help="how non-English text becomes Latin letters: hk = Harvard-Kyoto, then spelled in the model's"
                        " lowercase alphabet; hk-plain = raw HK, just lowercased; uroman = the uroman romanizer"
                        " (default: %(default)s)")
    p.add_argument("--model",
                   help=f"HF wav2vec2 CTC model with a Latin character vocab, used for every language (default: {ENGLISH_MODEL}"
                        f" for English, {MMS_MODEL} for the rest, which is CC-BY-NC-4.0; per-language overrides: LANG_MODELS)")
    p.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cpu, cuda, mps")
    p.add_argument("--pad", type=float, default=0.2,
                   help="seconds of extra audio on each side of a segment while aligning (default: %(default)s)")
    p.add_argument("--max-gap", type=float, default=0.0,
                   help="make word spans contiguous by absorbing silences up to this many seconds (between words,"
                        " and between a word and the segment edge) into the neighbouring word. Off by default: it"
                        " assigns real pauses to words, which cost ~100 ms of end-time accuracy in tests on"
                        " synthetic speech (default: %(default)s)")
    p.add_argument("--clamp", action="store_true",
                   help="force word times inside their segment. Off by default: a word the annotated segment"
                        " boundary cuts off keeps its real time and may extend up to --pad past the segment edge,"
                        " whereas clamping squeezes it to ~0 s")
    p.add_argument("--no-refine", action="store_true",
                   help="keep raw CTC word edges instead of stretching words over adjacent voiced audio")
    p.add_argument("--word-meta", default=",".join(WORD_META_KEYS),
                   help="metadata line written between each word's timestamps and its text: comma list from"
                        f" {','.join(WORD_META_KEYS)}, or 'none' for the plain two-line format (default: %(default)s)")
    p.add_argument("--flag-confidence", type=float, default=0.3,
                   help="flagged=1 for words whose confidence is below this (default: %(default)s)")
    p.add_argument("--drop-tags", action="store_true",
                   help="omit tag/punctuation-only tokens (@repetition-start, @short-pause, ...) from the word lines")
    args = p.parse_args()
    meta_keys = [] if args.word_meta.strip().lower() == "none" else [k.strip() for k in args.word_meta.split(",") if k.strip()]
    if bad := [k for k in meta_keys if k not in WORD_META_KEYS]:
        p.error(f"--word-meta: unknown field(s) {bad}; choose from {list(WORD_META_KEYS)} or 'none'")
    output = args.output or args.rsml.with_suffix(".aligned.rsml")

    blocks = read_blocks(args.rsml)
    parsed = [(parse_block(b), b) for b in blocks]
    n_segments = sum(seg is not None for seg, _ in parsed)
    if not n_segments:
        sys.exit(f"error: no segments found in {args.rsml}")
    cfg = read_config([b for seg, b in parsed if seg is None])

    print(f"loading audio: {args.audio}", file=sys.stderr)
    audio = load_audio(args.audio)
    pipelines = Pipelines(cfg, args.model, args.device, args.translit)
    voiced = voiced_frames(audio)

    out_blocks, fallbacks, untransliterated, notes, done = [], [], [], collections.Counter(), 0
    for seg, block in parsed:
        if seg is None:
            out_blocks.append(block)
            continue
        lang = resolve_language(seg, cfg, args.lang, notes)
        aligner, norm = pipelines.get(lang)
        tokens = tokenize(" ".join(seg.text_lines), norm)
        untransliterated += [t.text for t in tokens if t.dropped]
        aligned = align_segment(aligner, audio, seg, tokens, args.pad)
        if aligned is None:
            fallbacks.append(seg.header[0].strip())
            print(f"warning: segment {fallbacks[-1]}: audio window too short for the text, "
                  "spreading words evenly", file=sys.stderr)
            spans, scores = spread_evenly(tokens, seg), [None] * len(tokens)
        else:
            spans, scores = aligned
            if not args.no_refine:
                spans = extend_into_voiced(spans, seg, voiced)
        times = place_tags(tokens, finalize(spans, seg, args.max_gap, args.clamp), seg)
        word_lines = []
        gaps = pauses(tokens, times, voiced)
        for tok, span, score, pause in zip(tokens, times, scores, gaps):
            if tok.words or not args.drop_tags:
                word_lines.append(f"{INDENT}{format_ts(span[0])} --> {format_ts(span[1])}")
                if meta_keys:
                    word_lines.append(INDENT + word_meta(meta_keys, tok, score, span, voiced, pause, args.flag_confidence))
                word_lines.append(INDENT + tok.text)
        out_blocks.append(seg.header + seg.text_lines + word_lines)
        done += 1
        if done % 25 == 0 or done == n_segments:
            print(f"aligned {done}/{n_segments} segments", file=sys.stderr)

    for note, count in notes.items():
        print(f"note: {count} segment(s): {note}", file=sys.stderr)
    if untransliterated:
        print(f"warning: {len(untransliterated)} word(s) contain letters the model has no sound for (script not supported"
              f" by Harvard-Kyoto; for Urdu/Ol Chiki/Meetei Mayek pip install uroman) and were left untimed, e.g. {untransliterated[:5]}", file=sys.stderr)
    output.write_text("\n\n".join("\n".join(b) for b in out_blocks) + "\n", encoding="utf-8")
    print(f"wrote {output}" + (f" ({len(fallbacks)} segment(s) used the even-spread fallback)" if fallbacks else ""),
          file=sys.stderr)


if __name__ == "__main__":
    main()
