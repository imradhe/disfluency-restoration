"""Command line: ``python -m disfrestore <prepare|stats|train|evaluate|infer> ...``"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disfrestore", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download + verify the free MSU/ISIP Switchboard transcripts (OpenSLR 5)")
    d.add_argument("--dest", default="data/MSU Switchboard")
    d.add_argument("--force", action="store_true")

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

    e = sub.add_parser("evaluate", help="tagging, restoration and WER on a simulated split")
    e.add_argument("--data", default="data/processed")
    e.add_argument("--root", default="data/MSU Switchboard/swb_ms98_transcriptions")
    e.add_argument("--model", default="runs/electra_small")
    e.add_argument("--split", default="test")
    e.add_argument("--max-examples", type=int, default=None)
    e.add_argument("--oracle", action="store_true", help="use gold tags instead of the model (phones->words ceiling)")
    e.add_argument("--encoder", default="google/electra-small-discriminator", help="tokenizer source for --oracle")
    e.add_argument("--show", type=int, default=6)
    e.add_argument("--out", default=None, help="write the report as JSON")
    e.add_argument("--threads", type=int, default=2)

    i = sub.add_parser("infer", help="audio -> phones + ASR -> restored, typed transcript")
    i.add_argument("--audio", required=True)
    i.add_argument("--model", default="runs/electra_small")
    i.add_argument("--out", default="outputs/result", help="output stem; writes <stem>.json and <stem>.txt")
    i.add_argument("--data", default="data/processed")
    i.add_argument("--root", default="data/MSU Switchboard/swb_ms98_transcriptions")
    i.add_argument("--phone-model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    i.add_argument("--phone-set", default="ipa", choices=["ipa", "timit"])
    i.add_argument("--asr-model", default="openai/whisper-small.en")
    i.add_argument("--min-window", type=float, default=4.0)
    i.add_argument("--max-window", type=float, default=14.0)
    i.add_argument("--min-prob", type=float, default=0.5, help="drop predicted spans below this mean probability")
    i.add_argument("--threads", type=int, default=2)

    n = sub.add_parser("fetch-nptel", help="stitch consecutive NPTEL utterances of one lecture into a wav (test audio)")
    n.add_argument("--shard", default="data/nptel_raw/data/test-00001-of-00005.parquet")
    n.add_argument("--out", default="data/nptel_audio")
    n.add_argument("--utterances", type=int, default=40, help="consecutive utterances per clip (about 7 s each)")
    n.add_argument("--clips", type=int, default=2)
    n.add_argument("--lecture", default=None, help="only lectures whose id contains this text, e.g. 108105103")
    args = ap.parse_args(argv)

    if args.cmd == "download":
        from .download import download

        download(Path(args.dest), args.force)
    if args.cmd == "prepare":
        from .data import prepare

        prepare(Path(args.root), Path(args.out), args.limit, args.variants, args.workers, args.seed, args.max_words)

    if args.cmd == "infer":
        from .infer import run

        run(args)
    if args.cmd == "fetch-nptel":
        from .nptel import fetch

        fetch(args)
    if args.cmd == "evaluate":
        from .evaluate import run

        run(args)
    if args.cmd == "train":
        from .train import train

        train(args)


if __name__ == "__main__":
    main()
