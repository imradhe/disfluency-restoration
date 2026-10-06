#!/usr/bin/env python3
"""Convert the MSU/ISIP Switchboard transcriptions to RSML: one .rsml file per recording.

The corpus is laid out as  <NN>/<NNNN>/ , a two-digit folder holding four-digit folders, and each four-digit folder is
one recording with two speakers, A and B:

    20/2001/sw2001A-ms98-a-trans.text   speaker A, segment level (utterances with start and end)
            sw2001A-ms98-a-word.text    speaker A, word level (every word with start and end)
            sw2001B-ms98-a-trans.text   speaker B, segment level
            sw2001B-ms98-a-word.text    speaker B, word level

Both speakers' segment and word levels are merged into ONE file per recording, laid out the same way:

    python switchboard_to_rsml.py          # 2,438 recordings -> data/MSU Switchboard/rsml/20/2001/sw2001.rsml, ...
    python switchboard_to_rsml.py --limit 20

    1                                                      segment number, in time order across both speakers
    00:00:00,978 --> 00:00:11,561                          utterance start --> end, as in the corpus
    primary=1|verified=1|flagged=0|note=sw2001A-ms98-a-0002    speaker (A = 1, B = 2) | verified | note = source utterance
    hi @umm yeah i'd like to talk ...                      segment transcript, in RSML
    <TAB>00:00:01,215 --> 00:00:01,725                     word start --> end
    <TAB>confidence=|...|pause_after_milliseconds=549|...|kind=word|script=latin|flagged=0
    <TAB>hi<TAB><TAB>hh ay                                 word, two tabs, phones (from sw-ms98-dict.text)

Segments are ordered by start time, and A comes before B when they start together. The two speakers talk over each
other, so neighbouring segments can overlap in time. `verified=1` because the corpus is hand-corrected. The corpus
files carry no gender, so the roster says `unspecified`. Timestamps are those of the corpus: seconds from the start
of the recording, the same clock for both speakers.

Conversion rules (Switchboard -> RSML):
    uh um hm eh huh uh-huh ooh uh-oh   -> @uhh @umm @hmm @ehh @huh @uh-huh @ooh @uh-oh      (real durations)
    ah                                 -> @uhh
    hum                                -> @hmm                                               (bare "hum" is a hesitation here)
    um-hum / uh-hum / hum-um / huh-uh / mm-hm -> @umm @hmm / @uhh @hmm / @hmm @umm / @huh @uhh / @umm @hmm   (time split by phone count)
    [laughter]                         -> @laughter                                          (isolated tag, real duration)
    [laughter-a] [laughter-b]          -> @laughing-start a b @laughing-end                  (one span per run; no duration)
    y[ou]-   -[o]kay   i-              -> @broken-word-start y @broken-word-end
    [tranged/changed]                  -> [tranged](changed)                                 ([verbatim](normalized))
    [disea[l]-/diesel]                 -> @broken-word-start disea @broken-word-end          (fragment; see --fragment-normalization)
    {alrighty}                         -> !!coinage[alrighty](alrighty)                      (domain `coinage`, declared in the config)
    them_1 because_1 ...               -> them because ...                                   (phones keep the reduced pronunciation)
    [noise]  [silence]                 -> dropped: not speech events (silence only feeds pause_after_milliseconds)
    [vocalized-noise] <b_aside> <e_aside> -> dropped (decisions below)
Segments left with no words after this (silence-only, noise-only) are not written.

Word metadata uses the same fields as the aligner's output. The corpus has no confidence or voicing measurements, so
those are empty; pause_after_milliseconds is the real gap to the next spoken word.

Needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from dataclasses import dataclass
from pathlib import Path

INDENT = "\t"
SPEAKER_IDS = {"A": 1, "B": 2}  # side of the call -> RSML speaker id

# ---------------------------------------------------------------------------------------------------------------
# Decisions. The first group is settled; the second group are my defaults, still open, each easy to change.
# ---------------------------------------------------------------------------------------------------------------
HESITATION_TAGS = {  # one Switchboard word -> one RSML hesitation tag
    "uh": "@uhh", "um": "@umm", "hm": "@hmm", "hum": "@hmm", "eh": "@ehh", "ah": "@uhh", "huh": "@huh", "uh-huh": "@uh-huh",
    "ooh": "@ooh", "uh-oh": "@uh-oh",  # @ooh and @uh-oh are new RSML hesitations
}
SPLIT_HESITATIONS = {  # one Switchboard word -> two tags; the third item is the first component, to split the phones
    "um-hum": ("@umm", "@hmm", "um"), "uh-hum": ("@uhh", "@hmm", "uh"),
    "hum-um": ("@hmm", "@umm", "hum"), "huh-uh": ("@huh", "@uhh", "huh"), "mm-hm": ("@umm", "@hmm", "mm"),
}
NOT_SPEECH = {"[noise]"}  # deliberately untagged in RSML: not a speech event

# Open decisions (defaults; --fragment-normalization and --pronunciation-variants switch two of them):
NOT_SPEECH |= {"[vocalized-noise]"}  # treated like [noise]. RSML does tag specific vocal sounds (@cough ...): to confirm
NOT_SPEECH |= {"<b_aside>", "<e_aside>"}  # RSML has no aside tag
PRONUNCIATION_VARIANTS = "plain"  # --pronunciation-variants verbatim spells them out, using the table below
VARIANT_SPELLING = {  # proposals, from the dictionary pronunciations of the *_1 entries
    "them": "em", "them's": "'ems", "because": "cause", "about": "bout", "okay": "mkay",
    "especially": "specially", "depends": "pends",
}

WORD_META_KEYS = ("confidence", "weakest_letter_confidence", "voiced_fraction", "pause_after_milliseconds",
                  "pause_voiced_fraction", "kind", "script", "flagged")

# The config trailer, in the layout BhashaCheck writes (configText.js). Copied from the project's RSML files, plus
# @ooh / @uh-oh in the hesitations and the `coinage` domain.
VERSIONS = {"bhashacheck": "1.0.0", "rsml": "3.3.4"}
TAGS = {
    "hesitations": "@umm @uhh @hmm @ugh @huh @tsk @uh-huh @ehh @ooh @uh-oh".split(),
    "isolatedParalinguistics": ("@laughter @cry @hum @breathe @sniff @nose-blowing @cough @sneeze @throat-clearing "
                                "@yawn @eating-sounds @snore @groan @sigh").split(),
    "isolatedOther": "@silence @unintelligible @stutter-block @pause @short-pause @long-pause".split(),
    "disfluencySpans": "filler repetition broken-word repair false-start prolongation".split(),
    "paralinguisticSpans": "crying yelling laughing singing humming whistling whispering".split(),
    "prosodySpans": "emphasis falling-pitch raising-pitch fast-speech slow-speech".split(),
}
LEGENDS = {
    "entities": {
        "BRAND": "Brand", "DATETIME": "Date/Time", "EVENT": "Event", "FAC": "Facility", "GPE": "Geo Political Entity",
        "HON": "Honorific", "ID": "Identifier", "ITEM": "Item", "LANG": "Language", "LAW": "Law/Policy",
        "LOC": "Location", "MONEY": "Money", "NUM": "Number", "ORG": "Organization", "PER": "Person",
        "QUANT": "Quantity", "SPORTS": "Sports", "WOA": "Work of Art",
    },
    "languages": {
        "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "en": "English", "gu": "Gujarati",
        "hi": "Hindi", "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam",
        "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit",
        "sat": "Santali", "sd": "Sindhi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    },
    "dialects": {},
    "domains": {"coinage": "Coinage"},
    "accents": {},
}

VARIANT_RE = re.compile(r"_\d+$")
ANOMALOUS_RE = re.compile(r"^\[(.+)/([^/]+)\](-?)$")  # [as said / intended], a trailing dash = then cut off
BRACKETS_RE = re.compile(r"\[[^\]]*\]")
NON_SPEECH_RE = re.compile(r"^(\[noise\]|\[vocalized-noise\]|\[laughter\]|<[be]_aside>|uh|um|hm|ah|eh|huh|oh)$", re.I)


BOUNDS_TOLERANCE = 0.011  # seconds; word times may stick out of the utterance by about a millisecond of rounding


@dataclass
class Unit:
    """One word-level entry of a segment."""
    text: str
    start: float
    end: float
    kind: str  # word | hesitation | isolated | span
    phones: str = ""


# ---------------------------------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------------------------------

def find_corpus(given: str | None) -> Path:
    if given:
        return Path(given)
    for base in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent):
        for folder in (base / "data").glob("*"):
            if folder.name.strip() == "MSU Switchboard" and (folder / "swb_ms98_transcriptions").is_dir():
                return folder / "swb_ms98_transcriptions"
    sys.exit("error: could not find data/MSU Switchboard/swb_ms98_transcriptions; pass its path as the first argument")


def read_dictionary(corpus: Path) -> dict[str, str]:
    """word -> phones. A few entries are listed more than once, mostly with identical phones: keep each distinct one."""
    seen: dict[str, list[str]] = collections.defaultdict(list)
    for line in (corpus / "sw-ms98-dict.text").read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith(" file:"):
            continue
        word, _, phones = line.strip().partition(" ")
        phones = " ".join(phones.split())
        if phones not in seen[word]:
            seen[word].append(phones)
    return {w: " | ".join(p) for w, p in seen.items()}


def read_transcript(trans_path: Path):
    """-> [(utt_id, start, end, [(start, end, token), ...] without [silence])] for one speaker's side."""
    words = collections.defaultdict(list)
    for line in trans_path.with_name(trans_path.name.replace("-trans", "-word")).read_text(errors="replace").splitlines():
        utt, start, end, token = line.split()  # a few files separate the fields with tabs or runs of spaces
        if token != "[silence]":
            words[utt].append((float(start), float(end), token))
    out = []
    for line in trans_path.read_text(errors="replace").splitlines():
        utt, start, end, _ = line.split(None, 3)
        out.append((utt, float(start), float(end), words.get(utt, [])))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Token -> RSML units
# ---------------------------------------------------------------------------------------------------------------

def clean(text: str) -> str:
    """Drop partial-word brackets and dashes: 'y[ou]-' -> 'you', '-[o]kay' -> 'okay'."""
    return text.replace("[", "").replace("]", "").strip("-")


def follows(intended: str, following: list[str], within: int = 3) -> bool:
    """Does the intended word turn up in the next few spoken words?"""
    want = {intended.lower(), intended.lower().replace("-", "")}
    nxt = [t.lower() for t in following if not NON_SPEECH_RE.match(t)][:within]
    return any(t in want or t.replace("-", "") in want for t in nxt)


def fragment(said: str, intended: str, start: float, end: float, phones: str, following: list[str], opt) -> list[Unit]:
    """A cut-off word: the fragment wrapped in a @broken-word span. The span has no duration of its own."""
    text = said
    if opt.fragment_normalization and not follows(intended, following):
        text = f"[{said}]({intended})"  # keep the intended word when the transcript never says it
    return [Unit("@broken-word-start", start, start, "span"), Unit(text, start, end, "word", phones),
            Unit("@broken-word-end", end, end, "span")]


def split_hesitation(token: str, start: float, end: float, phones: str, dictionary: dict) -> list[Unit]:
    """um-hum -> @umm @hmm. The phones and the time are split between the two tags."""
    first, second, first_word = SPLIT_HESITATIONS[token.lower()]
    ph = phones.split(" | ")[0].split()
    first_ph = dictionary.get(first_word, "").split(" | ")[0].split()
    k = len(first_ph) if first_ph and ph[:len(first_ph)] == first_ph and len(first_ph) < len(ph) else (len(ph) + 1) // 2
    middle = start + (end - start) * (k / len(ph) if ph else 0.5)
    return [Unit(first, start, middle, "hesitation", " ".join(ph[:k])), Unit(second, middle, end, "hesitation", " ".join(ph[k:]))]


def convert_token(token: str, start: float, end: float, key: str, following: list[str], dictionary: dict, opt,
                  stats: collections.Counter) -> list[Unit]:
    """One Switchboard token -> its RSML units. `key` is the dictionary entry holding the phones of what was said."""
    phones, low = dictionary.get(key, ""), token.lower()
    if low in HESITATION_TAGS:
        stats["hesitations"] += 1
        return [Unit(HESITATION_TAGS[low], start, end, "hesitation", phones)]
    if low in SPLIT_HESITATIONS:
        stats["backchannels split into two tags"] += 1
        return split_hesitation(token, start, end, phones, dictionary)
    if token == "[laughter]":
        stats["@laughter"] += 1
        return [Unit("@laughter", start, end, "isolated")]
    anomalous = ANOMALOUS_RE.match(token)
    if anomalous:
        said, intended, cut_off = anomalous.groups()
        if "[" in said or cut_off:  # a slip that was also cut off: [disea[l]-/diesel]  or  [tack/talking]-
            stats["slips that were cut off (broken-word)"] += 1
            return fragment(BRACKETS_RE.sub("", said).strip("-"), clean(intended), start, end, phones, following, opt)
        stats["slips ([verbatim](normalized))"] += 1
        return [Unit(f"[{said}]({clean(intended)})", start, end, "word", phones)]
    if ("[" in token or "]" in token) and (token.endswith("-") or token.startswith("-")):  # y[ou]-  -[o]kay
        stats["word fragments (broken-word)"] += 1
        said = BRACKETS_RE.sub("", token).strip("-")
        if not said:  # nothing audible was transcribed: nothing to align. Counted in the report, never silent
            stats["fragments with nothing audible (dropped)"] += 1
            return []
        return fragment(said, clean(token), start, end, phones, following, opt)
    if token.endswith("-"):  # i-
        stats["whole words cut off (broken-word)"] += 1
        return fragment(token.rstrip("-"), token.rstrip("-"), start, end, phones, following, opt)
    if token.startswith("{") and token.endswith("}"):
        stats["coinages (!!coinage)"] += 1
        return [Unit(f"!!coinage[{token[1:-1]}]({token[1:-1]})", start, end, "word", phones)]
    if VARIANT_RE.search(token):
        stats["pronunciation variants"] += 1
        base = VARIANT_RE.sub("", token)
        if opt.pronunciation_variants == "verbatim" and base in VARIANT_SPELLING:
            return [Unit(f"[{VARIANT_SPELLING[base]}]({base})", start, end, "word", phones)]
        return [Unit(base, start, end, "word", phones)]
    return [Unit(token, start, end, "word", phones)]


def convert_utterance(entries, dictionary: dict, opt, stats: collections.Counter) -> list[Unit]:
    units, i = [], 0
    tokens = [t for _, _, t in entries]
    while i < len(entries):
        start, end, token = entries[i]
        if token in NOT_SPEECH:
            stats[f"dropped {token}"] += 1
            i += 1
        elif token.startswith("[laughter-"):
            j = i  # a run of laughed words is one @laughing span
            while j < len(entries) and entries[j][2].startswith("[laughter-"):
                j += 1
            units.append(Unit("@laughing-start", start, start, "span"))
            for k in range(i, j):
                s, e, t = entries[k]
                inner = t[len("[laughter-"):-1]
                units += convert_token(inner, s, e, t, tokens[k + 1:], dictionary, opt, stats)
            units.append(Unit("@laughing-end", entries[j - 1][1], entries[j - 1][1], "span"))
            stats["laughing spans"] += 1
            stats["words spoken while laughing"] += j - i
            i = j
        else:
            units += convert_token(token, start, end, token, tokens[i + 1:], dictionary, opt, stats)
            i += 1
    stats["spoken entries with no dictionary phones"] += sum(1 for u in units if u.kind in ("word", "hesitation") and not u.phones)
    return units


# ---------------------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------------------

def format_ts(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def pauses(units: list[Unit]) -> list[int | None]:
    """Per unit: ms until the next spoken word (None for tags and for the last word)."""
    out: list[int | None] = [None] * len(units)
    spoken = [i for i, u in enumerate(units) if u.kind in ("word", "hesitation")]
    for i, j in zip(spoken, spoken[1:]):
        out[i] = max(0, round((units[j].start - units[i].end) * 1000))
    return out


def word_meta(keys, unit: Unit, pause: int | None) -> str:
    spoken = unit.kind in ("word", "hesitation")
    values = {  # the corpus has no confidence or voicing measurements, so those stay empty
        "confidence": "", "weakest_letter_confidence": "", "voiced_fraction": "",
        "pause_after_milliseconds": "" if pause is None else str(pause), "pause_voiced_fraction": "",
        "kind": unit.kind, "script": "latin" if spoken else "", "flagged": "0",
    }
    return "|".join(f"{k}={values[k]}" for k in keys)


def config_text(speakers: int) -> str:
    kv = lambda k, v: f"{k} = {v}" if v else f"{k} ="
    lines = ["# BhashaCheck config", "[versions]", *(kv(k, v) for k, v in VERSIONS.items()), "",
             "[settings]", kv("default_code_mixing_language", ""), "", "[speakers]",
             *(kv(i, "unspecified, en") for i in range(1, speakers + 1)), "", "[tags]",
             *(kv(k, ", ".join(v)) for k, v in TAGS.items())]
    for name, legend in LEGENDS.items():
        lines += ["", f"[{name}]", *(kv(code, desc) for code, desc in legend.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Convert the MSU/ISIP Switchboard transcriptions to RSML, one file per recording.")
    p.add_argument("corpus", nargs="?", help="swb_ms98_transcriptions folder (default: found under data/)")
    p.add_argument("-o", "--output-dir", type=Path,
                   help="folder for the .rsml files, laid out as NN/NNNN/swNNNN.rsml (default: rsml/ next to the corpus folder)")
    p.add_argument("--limit", type=int, help="convert only the first N recordings (sorted by folder)")
    p.add_argument("--word-meta", default=",".join(WORD_META_KEYS),
                   help=f"word metadata fields, comma list from {','.join(WORD_META_KEYS)}, or 'none' (default: all)")
    p.add_argument("--fragment-normalization", action="store_true",
                   help="when a cut-off word's intended word does not follow within 3 words, keep it: [frag](intended)"
                        " inside the @broken-word span. Off by default, matching RSML's fragment-only spans")
    p.add_argument("--pronunciation-variants", choices=["plain", "verbatim"], default=PRONUNCIATION_VARIANTS,
                   help="them_1 -> 'them' (plain) or [em](them) (verbatim, using VARIANT_SPELLING) (default: %(default)s)")
    opt = p.parse_args()
    keys = [] if opt.word_meta.strip().lower() == "none" else [k.strip() for k in opt.word_meta.split(",") if k.strip()]
    if bad := [k for k in keys if k not in WORD_META_KEYS]:
        p.error(f"--word-meta: unknown field(s) {bad}")

    corpus = find_corpus(opt.corpus)
    out_dir = opt.output_dir or corpus.parent / "rsml"
    dictionary = read_dictionary(corpus)
    recordings = sorted({f.parent for f in corpus.glob("*/*/*-trans.text")})  # the four-digit folders
    if opt.limit:
        recordings = recordings[:opt.limit]
    if not recordings:
        sys.exit(f"error: no *-trans.text files under {corpus}")

    stats, widened = collections.Counter(), []
    for n, folder in enumerate(recordings, start=1):
        segments = []  # (start, end, speaker, utterance id, units), both speakers together
        for path in sorted(folder.glob("*-trans.text")):
            side = path.name.split("-")[0][-1]  # sw2001A -> A
            if side not in SPEAKER_IDS:
                sys.exit(f"error: {path} is not side A or B")
            for utt, start, end, entries in read_transcript(path):
                units = convert_utterance(entries, dictionary, opt, stats)
                if not units:
                    stats["segments not written (silence / noise only)"] += 1
                    continue
                low, high = min(u.start for u in units), max(u.end for u in units)
                if low < start - BOUNDS_TOLERANCE or high > end + BOUNDS_TOLERANCE:
                    # The corpus gives this utterance times that cannot hold its own words (e.g. 25 words in 2.7 s); the
                    # alignment is the plausible one, so the segment is widened to cover its words.
                    stats["segments widened to cover their words (corpus inconsistency)"] += 1
                    widened.append(utt)
                    start, end = min(start, low), max(end, high)
                segments.append((start, end, SPEAKER_IDS[side], utt, units))
        if not segments:
            stats["recordings with no speech (no .rsml written)"] += 1
            continue
        # Time order, A before B when they start together. Sorted on the start as it is written (whole milliseconds),
        # so the file's order never contradicts the timestamps it prints.
        segments.sort(key=lambda seg: (round(seg[0] * 1000), seg[2]))

        blocks, previous_end = [], -1.0
        for number, (start, end, speaker, utt, units) in enumerate(segments, start=1):
            stats["segments written"] += 1
            stats["word-level entries"] += len(units)
            if start < previous_end - 1e-9:
                stats["segments that overlap the previous one (both speakers talking)"] += 1
            previous_end = max(previous_end, end)
            lines = [f"{number}", f"{format_ts(start)} --> {format_ts(end)}",
                     f"primary={speaker}|verified=1|flagged=0|note={utt}", " ".join(u.text for u in units)]
            for unit, pause in zip(units, pauses(units)):
                lines.append(f"{INDENT}{format_ts(unit.start)} --> {format_ts(unit.end)}")
                if keys:
                    lines.append(INDENT + word_meta(keys, unit, pause))
                lines.append(INDENT + unit.text + (f"\t\t{unit.phones}" if unit.phones else ""))
            blocks.append("\n".join(lines) + "\n\n")

        target = out_dir / folder.parent.name / folder.name / f"sw{folder.name}.rsml"  # 20/2001/sw2001.rsml
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(blocks) + config_text(max(SPEAKER_IDS[f.name.split("-")[0][-1]] for f in folder.glob("*-trans.text"))),
                          encoding="utf-8")
        stats["recordings written"] += 1
        if n % 250 == 0 or n == len(recordings):
            print(f"{n:,}/{len(recordings):,} recordings", file=sys.stderr)

    size = sum(f.stat().st_size for f in out_dir.glob("*/*/*.rsml"))
    print(f"\nwrote {stats['recordings written']:,} .rsml files under {out_dir} ({size / 1e6:,.0f} MB)", file=sys.stderr)
    for name, count in sorted(stats.items()):
        print(f"  {count:>10,}  {name}", file=sys.stderr)
    if widened:
        print(f"\nsegments widened: {', '.join(widened)}", file=sys.stderr)


if __name__ == "__main__":
    main()
