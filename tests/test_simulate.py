import random
from pathlib import Path

import pytest

from disfrestore.encoding import LABELS
from disfrestore.swb import DEFAULT_ROOT

ROOT = Path(__file__).resolve().parents[1] / DEFAULT_ROOT
pytestmark = pytest.mark.skipif(not (ROOT / "sw-ms98-dict.text").exists(), reason="Switchboard corpus not downloaded")


@pytest.fixture(scope="module")
def setup():
    from disfrestore.labeler import chunk_side, label_units, units_from_segments
    from disfrestore.lexicon import G2P
    from disfrestore.swb import SwbDict, read_side

    side = read_side(ROOT, "sw2005", "A")
    chunks = [label_units(units_from_segments(c)) for c in chunk_side(side)]
    return SwbDict.load(ROOT), G2P(neural=False), chunks


def build(setup, i, seed=1, **prof):
    from disfrestore.simulate import Profile, make_example

    swb, g2p, chunks = setup
    p = Profile(**{"verbosity": 0.0, "noise": 0.3, "asr_sub": 0.0, "asr_del": 0.0, "asr_ins": 0.0, **prof})
    return make_example(chunks[i], swb, g2p, random.Random(seed), None, p, f"t:{i}")


def test_invariants(setup):
    n = 0
    for i in range(len(setup[2])):
        ex = build(setup, i)
        if ex is None:
            continue
        n += 1
        W = len(ex["words"])
        assert len(ex["wl"]) == W and len(ex["wf"]) == W and len(ex["gp"]) == W + 1 == len(ex["gl"]) == len(ex["gf"])
        for run, labs, feats in zip(ex["gp"], ex["gl"], ex["gf"]):
            assert len(run) == len(labs) == len(feats)
            assert all(l in LABELS for l in labs)
        assert all(l in LABELS for l in ex["wl"])
        for it in ex["items"]:
            assert 0 <= it["gap"] <= W and it["text"]
    assert n > 10


def test_deterministic(setup):
    assert build(setup, 3, seed=5) == build(setup, 3, seed=5)


def test_clean_asr_drops_most_fillers(setup):
    dropped = kept = 0
    for i in range(len(setup[2])):
        ex = build(setup, i, verbosity=0.0)
        if ex is None:
            continue
        dropped += sum(1 for it in ex["items"] if it["tag"] == "FP")
        kept += sum(1 for l in ex["wl"] if l.endswith("FP") and l.startswith("B"))
    assert dropped > 10 and kept < 0.25 * dropped  # a clean ASR keeps ~5% of the fillers, which then get tagged instead
