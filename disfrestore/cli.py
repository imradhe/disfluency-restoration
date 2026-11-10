"""Command line: ``python -m disfrestore <prepare|stats|train|evaluate|infer> ...``"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disfrestore", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build simulated training examples from the Switchboard transcripts")
    p.add_argument("--root", default="data/MSU Switchboard/swb_ms98_transcriptions")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--limit", type=int, default=None, help="use only N conversations (spread over the id range)")
    p.add_argument("--variants", type=int, default=1, help="independent ASR/recogniser simulations per chunk")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-words", type=int, default=40)

    t = sub.add_parser("train", help="train the tagger on data/processed")
    t.add_argument("--data", default="data/processed")
    t.add_argument("--out", default="runs/tagger")
    t.add_argument("--encoder", default="google/electra-small-discriminator",
                   help="any HF encoder: bert-base-uncased, roberta-base, microsoft/deberta-v3-base ...")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--class-weight-power", type=float, default=0.5)
    t.add_argument("--max-train", type=int, default=None)
    t.add_argument("--max-dev", type=int, default=3000)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        from .data import prepare

        prepare(Path(args.root), Path(args.out), args.limit, args.variants, args.workers, args.seed, args.max_words)

    if args.cmd == "train":
        from .train import train

        train(args)


if __name__ == "__main__":
    main()
