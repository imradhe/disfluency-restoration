# Disfluency restoration from phones, G2P and an ASR transcript

Modern ASR (Whisper, wav2vec2 + LM) writes *fluent* text: fillers, repeated words, restarts and word fragments are
dropped. This repo restores them. It compares what a **phone recogniser heard** with what the **G2P of the ASR
transcript predicts**, finds the sounds the transcript cannot explain, **types** them with a BERT-style tagger using
Switchboard's disfluency categories, and converts those sounds back into **words** with the transcript as context.

```
audio ──► phone recogniser (wav2vec2 CTC, IPA→ARPAbet) ──► recognised phones ─┐
  └────► ASR (Whisper) ──► transcript ──► G2P (CMUdict + g2p_en) ─────────────┤
                                                                              ▼
                 phone alignment (feature-weighted) ──► per-word misalignment + unexplained "gap" phones
                                                                              ▼
        [CLS] <gap0 phones> w1 <gap1 phones> w2 … wn <gap_n phones> [SEP]   (+ alignment features per token)
                                                                              ▼
                 transformer tagger (ELECTRA / BERT / RoBERTa…) ──► BIO tag per word and per gap phone
                                                                              ▼
   stage 1  restored SOUNDS : so[s ow] {FP: ah m} today[t ah d ey] …          (typed phone spans in their gaps)
   stage 2  restored WORDS  : so {FP:um} today …                              (phones → words, using the transcript)
```

## 1. Data: the free Switchboard release (transcripts only, no audio)

```bash
python -m disfrestore download          # OpenSLR resource 5 → data/MSU Switchboard/swb_ms98_transcriptions/
```

The MSU/ISIP release (`switchboard_word_alignments.tar.gz`, 49.6 MB, SHA-256 pinned in `disfrestore/download.py`) has, for
**2,438 conversations × 2 sides**, a *segment-level* file (`*-trans.text`: id, start, end, text) and a *word-level* file
(`*-word.text`: every word and every `[silence]` with times), plus `sw-ms98-dict.text`, a pronunciation dictionary that also
covers partial words (the pronunciation of the part actually spoken), laughter words, coinages and reduced variants.
All 4,876 + 4,876 files were checked; 100 % of lexical tokens have a dictionary entry. The corpus has 391,593 segments,
2.93 M words, 518 h of conversation.

## 2. Disfluency types

| tag | type | source |
|---|---|---|
| `FP` | filled pause — *uh um hm hum ah eh* | native MS98 markup |
| `EX` | other hesitation — *huh ooh uh-oh* | native |
| `BC` | backchannel — *uh-huh um-hum huh-uh* (conversational, not a disfluency proper) | native |
| `PW` | partial / cut-off word — `y[ou]-`, `-[o]kay`, `i-` | native |
| `MP` | mispronounced word — `[tranged/changed]` (the ASR writes the intended word) | native |
| `NS` | non-speech event — laughter, noise, vocalised noise (must **not** be restored as words) | native |
| `REP` | repetition (the reparandum) — *the the*, *i don't i don't* | **rule-derived** |
| `RPR` | repair / substitution (reparandum) — *to the, to a store* | **rule-derived** |
| `FS` | false start — abandoned words before a fragment that the next word does not complete | **rule-derived** |
| `EET` | explicit editing term between reparandum and repair — *i mean*, *you know*, *sorry* | **rule-derived** |
| `DM` | discourse marker at a clause edge or next to a cue — *you know*, *i mean*, *you see* | **rule-derived** |

> **Important caveat.** The free release only marks the *native* types. The Penn Treebank-3 `EDITED` layer
> (reparandum/interregnum/repair structure) is LDC-licensed and not in it, so `REP RPR FS EET DM` are **silver labels** from
> explicit, tested rules in `disfrestore/labeler.py` (reparandum = the part a clean-up would delete; the repair keeps `O` but
> shares a group id). Repetition rates agree with the independent counts in `switchboard_analysis.ipynb`
> (25.0 repeated n-grams per 1,000 words vs ≈ 23). Treat the derived types as a principled approximation, not gold.
> Single-word discourse markers (*well*, *like*, *so*) are deliberately not labelled — too ambiguous without a POS/syntax layer.

## 3. Training data is simulated (there is no audio)

Both pipeline inputs are generated from the verbatim transcript (`disfrestore/simulate.py`):

* **ASR transcript** — the verbatim words with disfluent ones dropped the way a normalising ASR drops them (per-tag drop
  probability scaled by a per-example *verbosity*, 0 = Whisper-like … 0.9 = near-verbatim, in which case the disfluent words stay
  in the text and are *tagged* instead of restored), plus near-homophone substitutions, deletions and insertions.
* **Recognised phones** — the dictionary pronunciation of every verbatim unit (reduced variants and spoken-part-only fragments
  included), laid over the real time span from the word alignments, then corrupted by a recogniser model: feature-aware
  substitutions, deletions biased to weak phones, insertions, junk on laughter/noise, and an Indian-English accent layer
  (θ/ð→t/d, v/w, z/s, vowel shifts).

Every simulated phone remembers its source word, so the gold labels (type + BIO span of each unexplained phone, and the
list of dropped items with their words) are exact. Splits follow the usual Switchboard disfluency convention
(test = sw4004–4153, dev = sw4519–4936, train = the rest): **126,077 / 8,420 / 7,420** chunks.

## 4. Misalignment, model, restoration

* `align.py` — global alignment of the recognised phones against the concatenated G2P phones of the ASR words. Substitution cost
  is articulatory distance (a voicing flip is the cheapest consonant confusion); an extra recognised phone is cheaper *between*
  words than inside one, so dropped material surfaces as a **gap run**; ties place it leftmost (reparandum before repair).
* `features.py` / `encoding.py` — per word: how many of its G2P phones were matched / substituted / deleted, pauses, speaking rate;
  per gap phone: duration, pauses, position in the run. The model reads words **and** gap phones in one sequence, with new `[ph_xx]`
  vocabulary entries, a token-kind embedding and an MLP over the alignment features added to the pretrained embeddings.
* `model.py` / `train.py` — any HuggingFace encoder (`--encoder google/electra-small-discriminator`, `bert-base-uncased`,
  `roberta-base`, `microsoft/deberta-v3-base` …) + linear head over 23 BIO labels, class-weighted cross-entropy.
* `restore.py` — phones → words: fillers → nearest filler pronunciation; repetitions → copy of the neighbouring ASR words whose
  G2P best matches; fragments → the prefix of the next word whose phone prefix matches (`th-`); other types → a lexicon decoder.

## 5. Usage

```bash
pip install -r requirements.txt pyarrow g2p_en cmudict nltk soundfile            # CPU torch: --index-url https://download.pytorch.org/whl/cpu
python -m disfrestore download
python -m disfrestore prepare                       # ~75 s, 4 workers → data/processed/{train,dev,test}.jsonl
python -m disfrestore train --epochs 2              # ELECTRA-small, ~40 min on 4 CPU cores
python -m disfrestore evaluate --split test         # tagging F1, restored items, verbatim WER   (--oracle = gold tags)
python -m disfrestore fetch-nptel --lecture 108105103 --utterances 100 --clips 1   # Indian-English test audio
python -m disfrestore infer --audio data/nptel_audio/108105103_lec021_1-100.wav --out outputs/lec021
python -m pytest -q tests
```

`infer` writes `<stem>.json` and `<stem>.txt` with, per window, the ASR text, the **restored phone sequence**, the **restored words** and
a **tagged** transcript: `{TAG:text}` = restored item, `[TAG:words]` = disfluent word already in the ASR text.

Some of the network hosts used (OpenSLR, huggingface.co and its CDN hosts, download.pytorch.org, NPTEL) may need to be reachable.
