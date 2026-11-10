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
def predict(model, encoded, batch_size: int = 128, pad_id: int = 0):
    """-> list of predicted label-id lists (one per example, aligned with Encoded.ids) and the logits' max prob."""
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].ids))
    preds = [None] * len(encoded)
    probs = [None] * len(encoded)
    for s in range(0, len(order), batch_size):
        ids = order[s:s + batch_size]
        batch = collate([encoded[i] for i in ids], pad_id)
        logits = model(batch["ids"], batch["attn"], batch["kind"], batch["feats"])
        p = torch.softmax(logits, -1)
        pr, lab = p.max(-1)
        for r, i in enumerate(ids):
            n = len(encoded[i].ids)
            preds[i] = lab[r, :n].tolist()
            probs[i] = p[r, :n].numpy()
    return preds, probs


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
