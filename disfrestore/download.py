"""Download the free MSU/ISIP Switchboard transcripts (OpenSLR resource 5): segment- and word-level alignments, no audio.

    python -m disfrestore download            # -> data/MSU Switchboard/swb_ms98_transcriptions/

The archive (49.6 MB, 2,438 conversations x 2 sides x {trans, word} + the dictionary) is fetched from the first OpenSLR
mirror that answers, resumed if interrupted, checked against the expected size and SHA-256, and extracted with a
whitelist: only regular files and directories whose path stays inside the target folder.
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://openslr.trmal.net/resources/5/switchboard_word_alignments.tar.gz",
    "https://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz",
    "https://us.openslr.org/resources/5/switchboard_word_alignments.tar.gz",
]
NAME = "switchboard_word_alignments.tar.gz"
SIZE = 49_651_293
SHA256 = "7bbc4dd5edc82b000078a028cc0a7f705e764da17cdf5027fe7888907578b106"  # recorded from the 2017-10-03 release
EXPECT = {"conversations": 2438, "trans": 4876, "word": 4876}


def _fetch(url: str, dest: Path) -> None:
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start again
            have = 0
        with open(dest, "ab" if have else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(tar_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if not (m.isreg() or m.isdir()) or not str(target).startswith(str(dest)):
                raise RuntimeError(f"refusing unsafe archive member: {m.name!r}")
        tar.extractall(dest, members=members, filter="data")


def verify(root: Path) -> dict[str, int]:
    got = {"conversations": len(list(root.glob("*/*/"))), "trans": len(list(root.glob("*/*/*-trans.text"))),
           "word": len(list(root.glob("*/*/*-word.text")))}
    if got != EXPECT or not (root / "sw-ms98-dict.text").exists():
        raise RuntimeError(f"unexpected corpus contents {got}, expected {EXPECT} plus sw-ms98-dict.text")
    return got


def download(dest: Path = Path("data/MSU Switchboard"), force: bool = False) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest / "swb_ms98_transcriptions"
    if root.exists() and not force:
        print(f"already extracted: {verify(root)}")
        return root
    tar_path = dest / NAME
    if not (tar_path.exists() and tar_path.stat().st_size == SIZE):
        err = None
        for url in MIRRORS:
            try:
                print(f"downloading {url}", flush=True)
                _fetch(url, tar_path)
                break
            except Exception as e:  # try the next mirror
                err = e
                print(f"  failed: {e}", flush=True)
        else:
            raise RuntimeError(f"no OpenSLR mirror answered (last error: {err})")
    if tar_path.stat().st_size != SIZE or sha256(tar_path) != SHA256:
        raise RuntimeError(f"{tar_path} does not match the expected size/SHA-256; delete it and retry")
    safe_extract(tar_path, dest)
    print("extracted:", verify(root))
    return root
