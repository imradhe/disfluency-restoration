import json

from disfrestore.nptel import _lev_align, check_fillers


def test_lev_align_matches_equal_words():
    assert _lev_align("a b c d".split(), "a x c d".split()) == {0: 0, 2: 2, 3: 3}
    assert _lev_align([], ["a"]) == {} and _lev_align(["a"], []) == {}


def test_check_fillers(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("so welcome ah we had ah done till now")
    res = tmp_path / "res.json"
    # ASR (no fillers) = "so welcome we had done till now"; one filler found at the right gap (2), one spurious (6)
    res.write_text(json.dumps({"windows": [{"asr": "so welcome we had done till now", "word_spans": [],
                                              "items": [{"tag": "FP", "gap": 2}, {"tag": "FP", "gap": 6}, {"tag": "REP", "gap": 3}]}]}))
    out = check_fillers(str(res), str(ref), tol=0)
    assert out["human_fillers"] == 2 and out["predicted_FP"] == 2 and out["hits"] == 1
    assert out["precision"] == 0.5 and out["recall"] == 0.5
