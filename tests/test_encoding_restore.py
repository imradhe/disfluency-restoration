import pytest

from disfrestore.encoding import LABELS, L2I, decode_spans
from disfrestore.lexicon import G2P
from disfrestore.restore import Restorer, render


def ids(*labs):
    return [L2I[l] for l in labs]


def test_decode_spans_bio():
    lab = ids("O", "B-FP", "I-FP", "B-REP", "O", "I-PW", "I-PW")
    assert decode_spans(lab, list(range(len(lab)))) == [(1, 2, "FP"), (3, 3, "REP"), (5, 6, "PW")]
    assert decode_spans(ids("I-FP", "I-REP"), [0, 1]) == [(0, 0, "FP"), (1, 1, "REP")]  # orphan I- starts a span
    assert len(LABELS) == 23


@pytest.fixture(scope="module")
def restorer():
    g2p = G2P()
    counts = {w: 100 - i for i, w in enumerate("the a and to of i you it that in is we uh um so then store went".split())}
    return Restorer(None, g2p, counts, n_lex=50), g2p


def test_filler_choice(restorer):
    r, _ = restorer
    assert r.restore_span("FP", ["ah", "m"], ["x"], 0)["text"] == "um"
    assert r.restore_span("FP", ["ah"], ["x"], 0)["text"] == "uh"
    assert r.restore_span("FP", ["hh", "m"], ["x"], 0)["text"] == "hm"
    assert r.restore_span("BC", ["ah", "hh", "ah"], ["x"], 0)["text"] == "uh-huh"


def test_repetition_copies_neighbour_words(restorer):
    r, _ = restorer
    words = "we went to the store".split()
    it = r.restore_span("REP", ["dh", "ah"], words, 3)  # gap before 'the'
    assert it["text"] == "the" and it["how"].startswith("copy")
    it = r.restore_span("REP", ["t", "uw", "dh", "ah"], words, 4)  # gap after 'the': copy of the previous two words
    assert it["text"] == "to the"


def test_fragment_is_a_prefix_of_the_next_word(restorer):
    r, _ = restorer
    it = r.restore_span("PW", ["s", "t"], "we went to the store".split(), 4)
    assert it["how"] == "fragment" and "store".startswith(it["text"].rstrip("-")) and it["text"].endswith("-")


def test_generic_decoder_and_events(restorer):
    r, _ = restorer
    ws, cost = r.decode(["dh", "ah", "s", "t", "ao", "r"])
    assert ws == ["the", "store"] and cost < 0.3
    assert r.restore_span("NS", ["hh"], ["x"], 0)["text"] == "[noise]"


def test_render(restorer):
    words = "we went to the store".split()
    items = [{"gap": 3, "tag": "FP", "text": "um"}, {"gap": 3, "tag": "REP", "text": "the"}, {"gap": 5, "tag": "NS", "text": "[noise]"}]
    assert render(words, items) == "we went to um the the store"
    assert render(words, items, [(1, 1, "FS")], tagged=True) == "we [FS:went] to {FP:um} {REP:the} the store"


def test_render_leaves_unresolved_items_out_of_the_plain_transcript():
    words = "we went to the store".split()
    items = [{"gap": 3, "tag": "REP", "text": "[d]", "how": "phones"}, {"gap": 3, "tag": "FP", "text": "um", "how": "inventory"}]
    assert render(words, items) == "we went to um the store"
    assert render(words, items, tagged=True) == "we went to {REP:[d]} {FP:um} the store"
