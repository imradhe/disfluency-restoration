"""Test audio: stitch consecutive NPTEL utterances of one lecture into a continuous wav.

Source: ``speechlab/NPTEL_TECH_ENG_50`` on the Hugging Face Hub (NPTEL technical-English lectures, ~7 s utterances in
lecture order, with a manual transcript that contains no fillers).  The audio stays in ``data/`` (git-ignored)."""
from __future__ import annotations

import collections
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def fetch(args) -> list[Path]:
    import pyarrow.parquet as pq

    shard = Path(args.shard)
    if not shard.exists():
        from huggingface_hub import hf_hub_download

        shard = Path(hf_hub_download("speechlab/NPTEL_TECH_ENG_50", "data/test-00001-of-00005.parquet", repo_type="dataset",
                                     local_dir=str(shard.parents[1])))
    rows = pq.read_table(shard, columns=["id", "audio", "text"]).to_pylist()
    by_lecture: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        m = re.match(r"(.+?_lec\d+)_(\d+)$", r["id"])
        if m:
            by_lecture[m.group(1)].append({"n": int(m.group(2)), **r})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for lecture, utts in sorted(by_lecture.items(), key=lambda kv: -len(kv[1])):
        utts.sort(key=lambda u: u["n"])
        # longest run of consecutive utterance numbers
        run, best = [utts[0]], []
        for u in utts[1:]:
            if u["n"] == run[-1]["n"] + 1:
                run.append(u)
            else:
                best = max(best, run, key=len)
                run = [u]
        best = max(best, run, key=len)[: args.utterances]
        if len(best) < 8:
            continue
        audio = np.concatenate([np.asarray(u["audio"]["array"], dtype=np.float32) for u in best])
        assert all(u["audio"]["sampling_rate"] == 16000 for u in best)
        wav = out / f"{lecture}_{best[0]['n']}-{best[-1]['n']}.wav"
        sf.write(wav, audio, 16000)
        wav.with_suffix(".ref.txt").write_text(" ".join(u["text"].strip() for u in best) + "\n")
        print(f"{wav}  {len(audio) / 16000:.0f}s  {len(best)} utterances")
        written.append(wav)
        if len(written) >= args.clips:
            break
    return written


# --------------------------------------------------------------------------------------------------------------
# Real-audio check against the human annotation: some NPTEL manual transcripts mark fillers ("ah").
# --------------------------------------------------------------------------------------------------------------
HUMAN_FILLERS = {"ah", "uh", "um", "umm", "hmm", "er", "erm", "mm", "aa", "ahh", "uhh"}


def _lev_align(a: list[str], b: list[str]) -> dict[int, int]:
    """Levenshtein alignment of a to b -> {index in a: index in b} for the matched (equal) positions."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (a[i - 1] != b[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def check_fillers(result_json: str, ref_txt: str, tol: int = 1) -> dict:
    """Compare predicted filled pauses (gap items tagged FP, plus FP-tagged ASR words) with the fillers a human wrote
    in the manual transcript.  Positions are gaps between ASR words; a prediction within ``tol`` words counts as a hit."""
    import json

    from .infer import normalise_text

    res = json.loads(Path(result_json).read_text())
    ref = normalise_text(Path(ref_txt).read_text())
    asr, preds, off = [], [], 0
    for w in res["windows"]:
        words = normalise_text(w["asr"])
        for it in w["items"]:
            if it["tag"] == "FP":
                preds.append(off + it["gap"])
        for a, b, t in w["word_spans"]:
            if t == "FP":
                preds.append(off + a)
        asr += words
        off += len(words)
    clean = [w for w in ref if w not in HUMAN_FILLERS]
    m = _lev_align(clean, asr)
    gold = []
    seen = 0
    for w in ref:
        if w in HUMAN_FILLERS:
            nxt = next((m[k] for k in range(seen, len(clean)) if k in m), len(asr))  # ASR index of the next matched word
            gold.append(nxt)
        else:
            seen += 1
    used, hit = set(), 0
    for g in gold:
        for p in preds:
            if p not in used and abs(p - g) <= tol:
                used.add(p)
                hit += 1
                break
    return {"human_fillers": len(gold), "predicted_FP": len(preds), "hits": hit, "precision": hit / max(len(preds), 1),
            "recall": hit / max(len(gold), 1), "asr_words": len(asr), "tolerance_words": tol}
