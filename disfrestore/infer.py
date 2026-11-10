"""Audio -> phones + ASR transcript -> G2P -> alignment -> tagger -> restored, typed transcript.

    python -m disfrestore infer --audio lecture.wav --model runs/electra_small --out outputs/lecture

Stages for every window of 5-14 s (cut at pauses so that both recognisers see the same audio):

1. phone recogniser      wav2vec2 CTC phoneme model (default facebook/wav2vec2-xlsr-53-espeak-cv-ft, IPA) -> ARPAbet phones
                         with frame times; greedy CTC decoding with the model's own vocabulary
2. ASR                   Whisper (default openai/whisper-small.en) -> normalised words (it drops most fillers/repeats)
3. G2P + alignment       phones of the ASR words vs the recognised phones -> misalignment features + unexplained gap phones
4. tagger                the transformer over words + gap phones -> typed BIO spans
5. restoration           phones -> words (fillers, repetitions, fragments, other) using the transcript as context
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .align import align_phones
from .encoding import Encoder
from .features import RecPhone, compute_features
from .phones import ipa_to_arpa, timit_to_arpa
from .restore import Restorer, render, spans_from_predictions, word_spans

SR = 16000
FRAME = 0.02  # wav2vec2 frame stride


def load_audio(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("error: ffmpeg not found on PATH (needed to decode the audio)")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: ffmpeg could not read {path}:\n{e.stderr.decode(errors='replace')}")
    return np.frombuffer(raw, dtype=np.float32)


def split_windows(audio: np.ndarray, min_len: float = 4.0, max_len: float = 14.0, min_pause: float = 0.2):
    """Cut the audio at pauses into windows of about min_len..max_len seconds -> [(start_s, end_s)]."""
    hop = int(0.01 * SR)
    n = len(audio) // hop
    if n == 0:
        return []
    frames = audio[: n * hop].reshape(n, hop)
    db = 10 * np.log10((frames.astype(np.float64) ** 2).mean(1) + 1e-10)
    floor = np.percentile(db, 10)
    quiet = db < floor + 7.0
    runs, i = [], 0  # pause runs [a, b) in frames
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= min_pause:
                runs.append((i, j))
            i = j
        else:
            i += 1
    total = n * 0.01
    wins, start = [], 0.0
    while total - start > max_len:
        cands = [(b - a, (a + b) / 2 * 0.01) for a, b in runs if start + min_len <= (a + b) / 2 * 0.01 <= start + max_len]
        if cands:
            cut = max(cands, key=lambda c: (round(c[0] / 5), c[1]))[1]  # longest pause (coarsely), then the latest
        else:
            lo, hi = int((start + max_len - 2) / 0.01), int((start + max_len) / 0.01)
            cut = (lo + int(np.argmin(db[lo:hi]))) * 0.01
        wins.append((start, cut))
        start = cut
    if total - start > 0.3:
        if wins and total - start < min_len / 2:
            wins[-1] = (wins[-1][0], total)
        else:
            wins.append((start, total))
    return wins


class PhoneRecognizer:
    """wav2vec2 CTC phoneme recogniser with greedy decoding."""

    def __init__(self, name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft", phone_set: str = "ipa", threads: int = 4):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = Wav2Vec2ForCTC.from_pretrained(name).eval()
        vocab = json.loads(Path(hf_hub_download(name, "vocab.json")).read_text())
        self.id2tok = {i: t for t, i in vocab.items()}
        self.blank = vocab.get("<pad>", 0)
        self.skip = {self.blank} | {vocab[t] for t in ("<s>", "</s>", "<unk>", "|", " ") if t in vocab}
        self.phone_set = phone_set
        self.unknown: dict[str, int] = {}

    def to_arpa(self, tok: str) -> list[str]:
        if self.phone_set == "timit":
            return timit_to_arpa(tok)
        unk: list[str] = []
        out = ipa_to_arpa(tok, unk)
        for u in unk:
            self.unknown[u] = self.unknown.get(u, 0) + 1
        return out

    def __call__(self, audio: np.ndarray) -> list[RecPhone]:
        torch = self.torch
        if len(audio) < 1600:
            return []
        x = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x.astype(np.float32))[None]).logits[0]
        probs = torch.softmax(logits, -1)
        conf, ids = probs.max(-1)
        ids, conf = ids.numpy(), conf.numpy()
        out: list[RecPhone] = []
        t, T = 0, len(ids)
        while t < T:
            u = t
            while u + 1 < T and ids[u + 1] == ids[t]:
                u += 1
            if ids[t] not in self.skip:
                arpa = self.to_arpa(self.id2tok[int(ids[t])])
                start, end = t * FRAME, (u + 1) * FRAME
                for k, ph in enumerate(arpa):
                    s = start + k * (end - start) / len(arpa)
                    out.append(RecPhone(ph, s, s + (end - start) / len(arpa), float(conf[t:u + 1].max())))
            t = u + 1
        return out


class ASR:
    """Whisper (or any HF ASR checkpoint) through the transformers pipeline."""

    def __init__(self, name: str = "openai/whisper-small.en", threads: int = 4):
        import torch
        from transformers import pipeline

        torch.set_num_threads(threads)
        self.pipe = pipeline("automatic-speech-recognition", model=name)

    def __call__(self, audios: list[np.ndarray], batch_size: int = 4) -> list[str]:
        outs = self.pipe([{"raw": a, "sampling_rate": SR} for a in audios], batch_size=batch_size)
        return [o["text"] for o in outs]


_NUM = re.compile(r"\d")


def normalise_text(text: str) -> list[str]:
    """Whisper-style text -> lower-case words without punctuation (apostrophes and inner hyphens kept)."""
    t = text.lower().replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"[^\w'\- ]", " ", t)
    t = re.sub(r"(?<!\w)-+|-+(?!\w)", " ", t)
    return [w for w in t.split() if w.strip("'-")]


def restored_phones(words: list[str], ref: list[list[str]], items: list[dict]) -> str:
    """Stage 1 of the restoration: the ASR transcript at phone level with the *restored sounds* spliced into the gaps,
    e.g. ``so[s ow] {FP: ah m} today[t ah d ey]``.  Stage 2 (phones -> words) is ``items[*].text``."""
    by_gap: dict[int, list[dict]] = {}
    for it in items:
        by_gap.setdefault(it["gap"], []).append(it)
    out = []
    for g in range(len(words) + 1):
        for it in by_gap.get(g, []):
            out.append("{%s: %s}" % (it["tag"], " ".join(it["phones"])))
        if g < len(words):
            out.append("%s[%s]" % (words[g], " ".join(ref[g])))
    return " ".join(out)


class Restoration:
    """Tagger + restorer bundle that turns (ASR words, recognised phones) into typed, restored output."""

    def __init__(self, model_dir: str, data_dir: str = "data/processed", root: str = "data/MSU Switchboard/swb_ms98_transcriptions",
                 threads: int = 4):
        import torch

        from .lexicon import G2P
        from .model import Tagger
        from .swb import SwbDict

        torch.set_num_threads(threads)
        self.torch = torch
        self.model, self.tok, _ = Tagger.load(model_dir)
        self.enc = Encoder(self.tok, max_len=500)
        self.g2p = G2P()
        self.g2p.load_cache(Path(data_dir) / "g2p_cache.json")
        counts = json.loads((Path(data_dir) / "word_counts.json").read_text())
        swb = SwbDict.load(Path(root)) if (Path(root) / "sw-ms98-dict.text").exists() else None
        self.restorer = Restorer(swb, self.g2p, counts)

    def analyse(self, words: list[str], rec: list[RecPhone], min_prob: float = 0.0) -> dict:
        ref = self.g2p.words(words)
        al = align_phones(ref, [p.ph for p in rec])
        wf, gf = compute_features(rec, al)
        ex = {"words": words, "wf": wf, "gp": [[rec[j].ph for j in run] for run in al.gaps], "gf": gf}
        e = self.enc.encode(ex, with_labels=False)
        from .encoding import collate

        batch = collate([e], self.tok.pad_token_id)
        with self.torch.no_grad():
            logits = self.model(batch["ids"], batch["attn"], batch["kind"], batch["feats"])[0]
        prob = self.torch.softmax(logits, -1)
        conf, lab = prob.max(-1)
        lab, conf = lab.tolist(), conf.tolist()
        spans = spans_from_predictions(e, lab, ex["gp"])
        items = []
        for s in spans:
            a, b = s["idx"]
            pos = e.gap_pos[s["gap"]][a:b + 1]
            p = float(np.mean([conf[q] for q in pos]))
            if p < min_prob:
                continue
            it = self.restorer.restore_span(s["tag"], s["phones"], words, s["gap"])
            it["prob"] = round(p, 3)
            items.append(it)
        ws = [(a, b, t) for a, b, t in word_spans(e, lab)]
        return {"words": words, "items": items, "word_spans": ws, "align_cost": al.cost,
                "n_gap_phones": sum(len(g) for g in al.gaps), "restored_phones": restored_phones(words, ref, items)}


def run(args) -> dict:
    import time

    audio = load_audio(args.audio)
    wins = split_windows(audio, args.min_window, args.max_window)
    print(f"{len(audio) / SR:.1f}s of audio -> {len(wins)} windows", flush=True)
    t0 = time.time()
    pr = PhoneRecognizer(args.phone_model, args.phone_set, args.threads)
    asr = ASR(args.asr_model, args.threads)
    res = Restoration(args.model, args.data, args.root, args.threads)
    print(f"models loaded in {time.time() - t0:.0f}s", flush=True)

    clips = [audio[int(a * SR):int(b * SR)] for a, b in wins]
    texts = asr(clips)
    out_windows, counts, n_words = [], {}, 0
    for (a, b), clip, text in zip(wins, clips, texts):
        words = normalise_text(text)
        rec = pr(clip)
        if not words:
            continue
        r = res.analyse(words, rec, args.min_prob)
        n_words += len(words)
        for it in r["items"]:
            counts[it["tag"]] = counts.get(it["tag"], 0) + 1
        for _, _, t in r["word_spans"]:
            counts["word:" + t] = counts.get("word:" + t, 0) + 1
        out_windows.append({
            "start": round(a, 2), "end": round(b, 2), "asr": text.strip(), "recognised_phones": " ".join(p.ph for p in rec),
            "items": r["items"], "word_spans": r["word_spans"], "restored_phones": r["restored_phones"],
            "restored": render(words, r["items"]),
            "tagged": render(words, r["items"], r["word_spans"], tagged=True),
        })
        print(f"[{a:6.1f}-{b:6.1f}] {out_windows[-1]['tagged']}", flush=True)
    summary = {"audio": str(args.audio), "duration_s": len(audio) / SR, "asr_words": n_words, "counts": counts,
               "per_1000_words": {k: round(1000 * v / max(n_words, 1), 1) for k, v in counts.items()},
               "unknown_ipa": pr.unknown}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "windows": out_windows}, indent=1))
    with open(out.with_suffix(".txt"), "w") as f:
        for w in out_windows:
            f.write(f"[{w['start']:.1f}-{w['end']:.1f}]\nASR     : {w['asr']}\nPHONES  : {w['restored_phones']}\n"
                    f"RESTORED: {w['restored']}\nTAGGED  : {w['tagged']}\n\n")
    print(json.dumps(summary, indent=1))
    return summary
