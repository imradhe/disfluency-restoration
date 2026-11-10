"""Build the training / dev / test examples from the Switchboard transcripts (``python -m disfrestore prepare``)."""
from __future__ import annotations

import collections
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from .labeler import LEXICAL_CATS, chunk_side, label_units, units_from_segments
from .lexicon import G2P
from .simulate import Neighbors, make_example
from .swb import DEFAULT_ROOT, SwbDict, list_conversations, read_side, split_of

_STATE: dict = {}


def _seed(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _init(root: str, cache: str, counts_path: str):
    swb = SwbDict.load(Path(root))
    g2p = G2P()
    g2p.load_cache(Path(cache))
    counts = json.loads(Path(counts_path).read_text())
    _STATE.update(root=Path(root), swb=swb, g2p=g2p, nb=Neighbors(counts, g2p))


def _build_conv(args) -> list[str]:
    conv, variants, seed, max_words, max_gap, tiny_keep = args
    out: list[str] = []
    split = split_of(conv)
    for side_id in "AB":
        side = read_side(_STATE["root"], conv, side_id)
        for ci, chunk in enumerate(chunk_side(side, max_gap=max_gap, max_words=max_words)):
            units = label_units(units_from_segments(chunk))
            n_tok = sum(1 for u in units if u.tag != "NS")
            for v in range(variants):
                rng = random.Random(_seed(seed, conv, side_id, ci, v))
                if split == "train" and n_tok <= 2 and rng.random() > tiny_keep:
                    continue  # one-word acknowledgements are plentiful; keep a share
                ex = make_example(units, _STATE["swb"], _STATE["g2p"], rng, _STATE["nb"], None,
                                  f"{conv}{side_id}:{ci}:{v}")
                if ex is not None:
                    ex["split"] = split
                    out.append(json.dumps(ex, separators=(",", ":")))
    return out


def prepare(root: Path = DEFAULT_ROOT, out_dir: Path = Path("data/processed"), limit: int | None = None,
            variants: int = 1, workers: int = 4, seed: int = 0, max_words: int = 40, max_gap: float = 1.0,
            tiny_keep: float = 0.3) -> dict[str, int]:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    convs = list_conversations(root)
    if limit:
        # spread the subset over the id range so every split is represented
        step = max(len(convs) // limit, 1)
        convs = convs[::step][:limit]
    # word counts (for ASR-error neighbours) and a warm G2P cache
    counts: collections.Counter = collections.Counter()
    for c in convs:
        for s in "AB":
            for seg in read_side(root, c, s).segments:
                for t in seg.toks:
                    if t.cat in LEXICAL_CATS:
                        from .labeler import surface

                        w = surface(t.text, t.cat)[0]
                        if w:
                            counts[w] += 1
    g2p = G2P()
    cache = out_dir / "g2p_cache.json"
    g2p.load_cache(cache)
    for w in list(counts):
        g2p.word(w)
    g2p.save(cache)
    cpath = out_dir / "word_counts.json"
    cpath.write_text(json.dumps(counts))
    print(f"{len(convs)} conversations, {len(counts)} word types, g2p cache {len(g2p.cache)} ({time.time() - t0:.0f}s)")

    files = {s: open(out_dir / f"{s}.jsonl", "w") for s in ("train", "dev", "test")}
    n = collections.Counter()
    jobs = [(c, variants, seed, max_words, max_gap, tiny_keep) for c in convs]
    with Pool(workers, initializer=_init, initargs=(str(root), str(cache), str(cpath))) as pool:
        for i, lines in enumerate(pool.imap_unordered(_build_conv, jobs, chunksize=4)):
            for line in lines:
                split = line[line.rindex('"split":"') + 9:].split('"', 1)[0]
                files[split].write(line + "\n")
                n[split] += 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} conversations  {dict(n)}  ({time.time() - t0:.0f}s)", flush=True)
    for f in files.values():
        f.close()
    print(f"done {dict(n)} in {time.time() - t0:.0f}s -> {out_dir}")
    return dict(n)


def read_jsonl(path: Path, limit: int | None = None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)
