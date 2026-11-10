"""End-to-end evaluation on simulated Switchboard test data.

Three layers of evidence, from model-only to end-to-end:

1. **tagging**    span P/R/F1 per disfluency type on gap phones and on ASR words (train.span_scores)
2. **restoration items**  every dropped item (tag, gap, words) of the reference vs the restored items:
                  *located* = same tag at the same gap (a repetition may sit at the equivalent neighbouring gap),
                  *text*    = located and the restored words are identical (fragments: prefix within one letter)
3. **verbatim WER**  restored transcript vs the verbatim reference, against the plain ASR transcript as baseline

``--oracle`` replaces the tagger by the gold labels, which isolates the phones->words stage (and shows the ceiling).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from .data import read_jsonl
from .encoding import IGNORE, LABELS, L2I, Encoder, decode_spans
from .labeler import TAGS
from .restore import Restorer, render, spans_from_predictions, word_spans


# which predicted types are actually inserted into the transcript (choose on dev; unreliable types hurt WER when wrong)
POLICIES = {
    "all types": {"FP", "EX", "BC", "PW", "REP", "RPR", "FS", "EET", "DM"},
    "no RPR/FS": {"FP", "EX", "BC", "PW", "REP", "EET", "DM"},
    "no RPR/FS/PW": {"FP", "EX", "BC", "REP", "EET", "DM"},
    "FP+EX+BC+REP": {"FP", "EX", "BC", "REP"},
    "FP+EX+BC": {"FP", "EX", "BC"},
    "FP only": {"FP"},
}


def wer_counts(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(edit errors, ref length)."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j - 1] + (ref[i - 1] != hyp[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[m], n


def match_items(gold: list[dict], pred: list[dict]):
    """Greedy one-to-one matching on (tag, gap); repetitions may also match at a neighbouring gap if the words agree."""
    used = set()
    out = []
    for gi, g in enumerate(gold):
        hit = None
        for pi, p in enumerate(pred):
            if pi in used or p["tag"] != g["tag"]:
                continue
            if p["gap"] == g["gap"]:
                hit = pi
                break
        if hit is None and g["tag"] in ("REP", "RPR"):
            n = max(len(g["text"].split()), 1)
            for pi, p in enumerate(pred):
                if pi not in used and p["tag"] == g["tag"] and abs(p["gap"] - g["gap"]) <= n and p["text"] == g["text"]:
                    hit = pi
                    break
        if hit is not None:
            used.add(hit)
        out.append((gi, hit))
    return out, used


def text_match(tag: str, pred: str, gold: str) -> bool:
    """Exact words, except fragments: ``th-`` and ``the-`` both count for a gold ``th-`` when one is a prefix of the other
    and they differ by at most one letter (the spelled length of a spoken fragment is a guess from the phones)."""
    if tag == "PW" and pred.endswith("-") and gold.endswith("-"):
        a, b = pred[:-1], gold[:-1]
        return (a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 1
    return pred == gold


def gold_label_ids(ex: dict, e) -> list[int]:
    return [l if l != IGNORE else 0 for l in e.labels]


def evaluate(examples: list[dict], enc: Encoder, restorer: Restorer, model=None, pad_id: int = 0,
             oracle: bool = False, show: int = 8, bias=None) -> dict:
    from .train import predict, span_scores

    encoded = [enc.encode(e) for e in examples]
    if oracle:
        preds = [gold_label_ids(e, x) for e, x in zip(examples, encoded)]
    else:
        preds, _ = predict(model, encoded, pad_id=pad_id, bias=bias)
    tagging = span_scores(encoded, preds)

    tp, fp, fn, exact = (collections.Counter() for _ in range(4))
    errs_base = errs_rest = errs_rest_notag = n_ref = 0
    pol_err = collections.Counter()
    shown = []
    for ex, e, pr in zip(examples, encoded, preds):
        spans = spans_from_predictions(e, pr, ex["gp"])
        items = [restorer.restore_span(s["tag"], s["phones"], ex["words"], s["gap"]) for s in spans]
        wspans = word_spans(e, pr)
        gold = [it for it in ex["items"] if it["tag"] != "NS"]
        pred_items = [it for it in items if it["tag"] != "NS"]
        matches, used = match_items(gold, pred_items)
        for gi, hit in matches:
            g = gold[gi]
            if hit is None:
                fn[g["tag"]] += 1
            else:
                tp[g["tag"]] += 1
                if text_match(g["tag"], pred_items[hit]["text"], g["text"]):
                    exact[g["tag"]] += 1
        for pi, p in enumerate(pred_items):
            if pi not in used:
                fp[p["tag"]] += 1
        restored = render(ex["words"], pred_items).split()
        a, n = wer_counts(ex["ref"], ex["words"])
        b, _ = wer_counts(ex["ref"], restored)
        for name, tagset in POLICIES.items():
            pol_err[name] += wer_counts(ex["ref"], render(ex["words"], [it for it in pred_items if it["tag"] in tagset]).split())[0]
        errs_base += a
        errs_rest += b
        n_ref += n
        if len(shown) < show and gold and (a != b):
            shown.append({"ref": " ".join(ex["ref"]), "asr": " ".join(ex["words"]),
                          "restored": render(ex["words"], pred_items, word_spans(e, pr), tagged=True),
                          "gold": [(g["gap"], g["tag"], g["text"]) for g in gold],
                          "pred": [(p["gap"], p["tag"], p["text"], p["how"]) for p in pred_items]})

    rows = {}
    for t in TAGS:
        if t == "NS":
            continue
        P = tp[t] / max(tp[t] + fp[t], 1)
        R = tp[t] / max(tp[t] + fn[t], 1)
        rows[t] = {"located_P": P, "located_R": R, "located_F1": 2 * P * R / max(P + R, 1e-9),
                   "exact_given_located": exact[t] / max(tp[t], 1), "gold": tp[t] + fn[t], "pred": tp[t] + fp[t]}
    T, F, N, X = sum(tp.values()), sum(fp.values()), sum(fn.values()), sum(exact.values())
    P, R = T / max(T + F, 1), T / max(T + N, 1)
    rows["micro"] = {"located_P": P, "located_R": R, "located_F1": 2 * P * R / max(P + R, 1e-9),
                     "exact_given_located": X / max(T, 1), "gold": T + N, "pred": T + F}
    return {"tagging": tagging, "items": rows,
            "wer": {"asr_baseline": errs_base / n_ref, "restored": errs_rest / n_ref, "ref_words": n_ref},
            "wer_policies": {k: v / n_ref for k, v in pol_err.items()},
            "examples": shown, "oracle": oracle, "n": len(examples)}


def format_report(rep: dict) -> str:
    from .train import format_scores

    lines = [f"examples: {rep['n']}   tagger: {'ORACLE (gold labels)' if rep['oracle'] else 'model'}"
             f"{' + prior calibration' if rep.get('calibrated') else ''}", "",
             format_scores(rep["tagging"]), "", "restoration of dropped items (NS events excluded)",
             f"  {'tag':<6}{'locP':>7}{'locR':>7}{'locF1':>7}{'text|loc':>11}{'gold':>7}{'pred':>7}"]
    for t, r in rep["items"].items():
        if r["gold"] or r["pred"]:
            lines.append(f"  {t:<6}{r['located_P']:7.3f}{r['located_R']:7.3f}{r['located_F1']:7.3f}"
                         f"{r['exact_given_located']:11.3f}{r['gold']:7d}{r['pred']:7d}")
    w = rep["wer"]
    lines += ["", f"verbatim WER vs reference ({w['ref_words']} words): plain ASR {100 * w['asr_baseline']:.2f}%  ->  "
                  f"restored {100 * w['restored']:.2f}%"]
    if "wer_policies" in rep:
        lines += ["", "WER by restoration policy (which predicted types are inserted):"]
        lines += [f"  {k:<16}{100 * v:6.2f}%" for k, v in rep["wer_policies"].items()]
    return "\n".join(lines)


def run(args) -> dict:
    import torch

    from .lexicon import G2P
    from .model import Tagger
    from .swb import SwbDict

    data = Path(args.data)
    examples = list(read_jsonl(data / f"{args.split}.jsonl", args.max_examples))
    counts = json.loads((data / "word_counts.json").read_text())
    g2p = G2P()
    g2p.load_cache(data / "g2p_cache.json")
    restorer = Restorer(SwbDict.load(Path(args.root)), g2p, counts)
    if args.oracle:
        from .encoding import load_tokenizer

        tok = load_tokenizer(args.encoder)
        model = None
    else:
        torch.set_num_threads(args.threads)
        model, tok, _ = Tagger.load(args.model)
    from .train import load_bias

    bias = None if args.oracle or args.no_calibration else load_bias(args.model)
    rep = evaluate(examples, Encoder(tok), restorer, model, tok.pad_token_id, oracle=args.oracle, bias=bias)
    rep["calibrated"] = bias is not None
    print(format_report(rep))
    for s in rep["examples"][:args.show]:
        print("\nREF     :", s["ref"], "\nASR     :", s["asr"], "\nRESTORED:", s["restored"],
              "\n  gold:", s["gold"], "\n  pred:", s["pred"])
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1, default=float))
    return rep
