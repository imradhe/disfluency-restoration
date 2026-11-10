"""Training and span-level evaluation of the tagger."""
from __future__ import annotations

import collections
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import read_jsonl
from .encoding import IGNORE, LABELS, Encoder, collate, decode_spans, load_tokenizer
from .labeler import TAGS
from .model import Tagger


def make_batches(encoded, batch_size: int, rng: random.Random, shuffle: bool = True):
    """Length-bucketed batches (sort inside windows of 50 batches so padding stays small)."""
    idx = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(idx)
    window = batch_size * 50
    batches = []
    for s in range(0, len(idx), window):
        part = sorted(idx[s:s + window], key=lambda i: len(encoded[i].ids))
        batches += [part[b:b + batch_size] for b in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def class_weights(encoded, power: float = 0.5, cap: float = 20.0) -> torch.Tensor:
    cnt = collections.Counter(l for e in encoded for l in e.labels if l != IGNORE)
    n_o = cnt[0] or 1
    w = torch.ones(len(LABELS))
    for i in range(1, len(LABELS)):
        w[i] = min((n_o / max(cnt[i], 1)) ** power, cap)
    return w / w[1:].mean().clamp(min=1.0) if power else torch.ones(len(LABELS))


@torch.no_grad()
def logits_for(model, encoded, batch_size: int = 128, pad_id: int = 0):
    """Per-example logits (n_tokens x n_labels), in the order of ``encoded``."""
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].ids))
    out = [None] * len(encoded)
    for s in range(0, len(order), batch_size):
        ids = order[s:s + batch_size]
        batch = collate([encoded[i] for i in ids], pad_id)
        logits = model(batch["ids"], batch["attn"], batch["kind"], batch["feats"])
        for r, i in enumerate(ids):
            out[i] = logits[r, :len(encoded[i].ids)].numpy()
    return out


def class_bias(weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """Undo (alpha=1) or soften the over-prediction that class-weighted cross-entropy builds in: logits - alpha*log(w)."""
    return -alpha * torch.log(weights)


def decode_logits(logits, bias=None):
    """-> (label ids per example, probabilities per example)"""
    preds, probs = [], []
    b = None if bias is None else np.asarray(bias, dtype=np.float32)
    for lg in logits:
        z = lg if b is None else lg + b
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(-1, keepdims=True)
        preds.append(p.argmax(-1).tolist())
        probs.append(p)
    return preds, probs


def predict(model, encoded, batch_size: int = 128, pad_id: int = 0, bias=None):
    """-> list of predicted label-id lists (one per example, aligned with Encoded.ids) and the probabilities."""
    return decode_logits(logits_for(model, encoded, batch_size, pad_id), bias)


def load_bias(model_dir) -> list[float] | None:
    f = Path(model_dir) / "calibration.json"
    return json.loads(f.read_text())["bias"] if f.exists() else None


def calibrate(args) -> dict:
    """Pick alpha on dev: tagger logits - alpha*log(class weights).  Writes <model>/calibration.json."""
    torch.set_num_threads(args.threads)
    model, tok, _ = Tagger.load(args.model)
    enc = Encoder(tok)
    tr = [enc.encode(e) for e in read_jsonl(Path(args.data) / "train.jsonl", args.max_train)]
    dv = [enc.encode(e) for e in read_jsonl(Path(args.data) / "dev.jsonl", args.max_dev)]
    cw = class_weights(tr, 0.5)
    lg = logits_for(model, dv, pad_id=tok.pad_token_id)
    best, rows = None, []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        bias = class_bias(cw, alpha).tolist()
        preds, _ = decode_logits(lg, bias)
        sc = span_scores(dv, preds)
        gap, word = sc["gap"]["micro"], sc["word"]["micro"]
        score = 0.5 * gap["f1"] + 0.5 * word["f1"]
        rows.append({"alpha": alpha, "gap_P": gap["p"], "gap_R": gap["r"], "gap_f1": gap["f1"], "word_f1": word["f1"],
                     "score": score})
        print(f"alpha {alpha:4.2f}  gap P {gap['p']:.3f} R {gap['r']:.3f} F1 {gap['f1']:.3f}   word F1 {word['f1']:.3f}   "
              f"mean {score:.3f}", flush=True)
        if best is None or score > best[0]:
            best = (score, alpha, bias)
    out = {"alpha": best[1], "bias": best[2], "dev_score": best[0], "grid": rows}
    (Path(args.model) / "calibration.json").write_text(json.dumps(out, indent=1))
    print(f"chosen alpha {best[1]} (dev mean span-F1 {best[0]:.3f}) -> {args.model}/calibration.json")
    return out


def span_scores(encoded, preds, group_tags=None) -> dict:
    """Span-level P/R/F1 per tag, separately for words and for gap phones, plus overall. Exact span match."""
    out = {}
    for part in ("word", "gap"):
        tp, fp, fn = (collections.Counter() for _ in range(3))
        for e, pr in zip(encoded, preds):
            runs = [e.word_pos] if part == "word" else e.gap_pos
            for g, pos in enumerate(runs):
                if not pos:
                    continue
                gold_ids = [l if l != IGNORE else 0 for l in e.labels]
                gs = {(g,) + s for s in decode_spans(gold_ids, pos)}
                ps = {(g,) + s for s in decode_spans(pr, pos)}
                for s in gs & ps:
                    tp[s[3]] += 1
                for s in ps - gs:
                    fp[s[3]] += 1
                for s in gs - ps:
                    fn[s[3]] += 1
        rows = {}
        for t in TAGS:
            p = tp[t] / max(tp[t] + fp[t], 1)
            r = tp[t] / max(tp[t] + fn[t], 1)
            rows[t] = {"p": p, "r": r, "f1": 2 * p * r / max(p + r, 1e-9), "gold": tp[t] + fn[t], "pred": tp[t] + fp[t]}
        T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
        p, r = T / max(T + F, 1), T / max(T + N, 1)
        rows["micro"] = {"p": p, "r": r, "f1": 2 * p * r / max(p + r, 1e-9), "gold": T + N, "pred": T + F}
        out[part] = rows
    return out


def format_scores(sc: dict) -> str:
    lines = []
    for part in ("gap", "word"):
        lines.append(f"[{part} spans]   {'tag':<6}{'P':>7}{'R':>7}{'F1':>7}{'gold':>8}{'pred':>8}")
        for t in TAGS + ["micro"]:
            r = sc[part][t]
            if r["gold"] or r["pred"]:
                lines.append(f"               {t:<6}{r['p']:7.3f}{r['r']:7.3f}{r['f1']:7.3f}{r['gold']:8d}{r['pred']:8d}")
    return "\n".join(lines)


def train(args) -> Path:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.encoder)
    enc = Encoder(tok)
    t0 = time.time()
    tr_ex = list(read_jsonl(Path(args.data) / "train.jsonl", args.max_train))
    dv_ex = list(read_jsonl(Path(args.data) / "dev.jsonl", args.max_dev))
    tr = [enc.encode(e) for e in tr_ex]
    dv = [enc.encode(e) for e in dv_ex]
    print(f"{len(tr)} train / {len(dv)} dev examples encoded in {time.time() - t0:.0f}s", flush=True)

    model = Tagger(args.encoder, tok)
    print(f"model {args.encoder}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    cw = class_weights(tr, args.class_weight_power)
    steps_per_epoch = math.ceil(len(tr) / args.batch_size)
    total = args.max_steps or steps_per_epoch * args.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    # the new modules (kind/feature MLP/head) learn faster than the pretrained encoder
    for g in opt.param_groups:
        g["initial_lr"] = args.lr
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(args.warmup, 1), max(0.0, (total - s) / max(total - args.warmup, 1))))

    pad = tok.pad_token_id
    step, best, log = 0, -1.0, []
    done = False
    t0 = time.time()
    while not done:
        for b in make_batches(tr, args.batch_size, rng):
            model.train()
            batch = collate([tr[i] for i in b], pad)
            loss, _ = model.loss(batch, cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{total} epoch {step / steps_per_epoch:.2f} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 or step == total:
                preds, _ = predict(model, dv, pad_id=pad)
                sc = span_scores(dv, preds)
                f1 = sc["gap"]["micro"]["f1"] * 0.5 + sc["word"]["micro"]["f1"] * 0.5
                print(f"== step {step}: dev gap micro-F1 {sc['gap']['micro']['f1']:.3f}  word micro-F1 "
                      f"{sc['word']['micro']['f1']:.3f}", flush=True)
                log.append({"step": step, "gap_f1": sc["gap"]["micro"]["f1"], "word_f1": sc["word"]["micro"]["f1"]})
                if f1 > best:
                    best = f1
                    model.save(out, tok, {"step": step, "dev_f1": f1})
            if step >= total:
                done = True
                break
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"training done in {time.time() - t0:.0f}s; best dev mean F1 {best:.3f} -> {out}", flush=True)
    return out
