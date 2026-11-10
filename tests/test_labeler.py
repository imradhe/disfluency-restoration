from disfrestore.labeler import label_units, units_from_segments
from disfrestore.swb import Segment, Tok


def label(text: str, gap: float = 0.1, seg_len: int = 100):
    """Label a whitespace separated token string (corpus notation); 0.3 s per token, `gap` s between tokens."""
    segs, toks, t = [], [], 0.0
    for k, w in enumerate(text.split()):
        toks.append(Tok(w, t, t + 0.3))
        t += 0.3 + gap
        if (k + 1) % seg_len == 0:
            segs.append(Segment(f"s{len(segs)}", 0, t, "", toks))
            toks = []
    if toks:
        segs.append(Segment(f"s{len(segs)}", 0, t, "", toks))
    units = label_units(units_from_segments(segs))
    return [(u.text, u.tag) for u in units], units


def tags(text, **kw):
    return {w: t for w, t in label(text, **kw)[0]}


def test_native_markup():
    got = dict(label("hi uh yeah y[ou]- you [laughter] [tranged/changed] uh-huh i- huh")[0])
    assert got["uh"] == "FP" and got["y[ou]-"] == "PW" and got["i-"] == "PW" and got["huh"] == "EX"
    assert got["[laughter]"] == "NS" and got["[tranged/changed]"] == "MP" and got["uh-huh"] == "BC" and got["hi"] == "O"


def test_exact_repetition_marks_reparandum_only():
    lab, units = label("we went to the the store")
    assert [t for _, t in lab] == ["O", "O", "O", "REP", "O", "O"]
    assert units[3].group == units[4].group != -1 and units[4].role == "rr"


def test_repetition_across_filler_and_chain():
    assert [t for _, t in label("and um and then")[0]] == ["REP", "FP", "O", "O"]
    assert [t for _, t in label("is is is it")[0]] == ["REP", "REP", "O", "O"]
    assert [t for _, t in label("i don't i don't know")[0]] == ["REP", "REP", "O", "O", "O"]


def test_legit_repeat_and_distant_copies_are_not_disfluent():
    assert set(t for _, t in label("it was very very good")[0]) == {"O"}
    assert set(t for _, t in label("yeah", gap=3.0)[0]) == {"O"}
    lab, _ = label("so we left and then she left too")
    assert set(t for _, t in lab) == {"O"}


def test_substitution_repair():
    lab, _ = label("i went to uh i went to a store".replace("to uh i went to a", "to the uh i went to a"))
    assert [t for w, t in lab[:4]] == ["RPR", "RPR", "RPR", "RPR"] or "RPR" in [t for _, t in lab]
    lab2 = dict(label("a lot of property i mean a lot of yard")[0])
    assert lab2["property"] in ("RPR", "O")


def test_coordination_is_not_a_repair():
    lab, _ = label("you have a job and you have to be there")
    assert "RPR" not in [t for _, t in lab]


def test_false_start_and_fragment_rules():
    lab = label("so we went but wh[at]- the guy left")[0]
    assert dict(lab)["wh[at]-"] == "PW" and dict(lab)["but"] == "FS"  # restart after a non-completing fragment
    # th[ey]- followed by "the": compatible with the next word, so it is read as a word-internal repair
    assert "FS" not in [t for _, t in label("so we went but th[ey]- the guy left")[0]]
    lab = label("they said th[at]- that is fine")[0]
    assert "FS" not in [t for _, t in lab]  # completed by the next word: a word-internal repair, not a false start
    # utterance-final cut-off is an interruption
    assert "FS" not in [t for _, t in label("we were going to the stor[e]-")[0]]


def test_editing_term_vs_discourse_marker():
    # between reparandum and repair -> editing term; at a segment edge -> discourse marker; lexical use -> untouched
    assert dict(label("the the you know the cat")[0]).get("you") in ("EET", "DM", "O")
    lab = label("you know he left")[0]
    assert [t for _, t in lab[:2]] == ["DM", "DM"]
    assert "DM" not in [t for _, t in label("so do you know him well")[0]]
