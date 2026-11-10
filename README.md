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
pip install -r requirements.txt        # CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m disfrestore download
python -m disfrestore prepare                       # ~75 s, 4 workers → data/processed/{train,dev,test}.jsonl
python -m disfrestore train --epochs 2              # ELECTRA-small, ~40 min on 4 CPU cores
python -m disfrestore calibrate                     # picks the prior-correction strength on dev → runs/electra_small/calibration.json
python -m disfrestore evaluate --split test         # tagging F1, restored items, verbatim WER   (--oracle = gold tags)
python -m disfrestore fetch-nptel --lecture 108105103 --utterances 100 --clips 1   # Indian-English test audio
python -m disfrestore infer --audio data/nptel_audio/108105103_lec021_1-100.wav --out outputs/lec021
python -m pytest -q tests
```

`infer` writes `<stem>.json` and `<stem>.txt` with, per window, the ASR text, the **restored phone sequence**, the **restored words** and
a **tagged** transcript: `{TAG:text}` = restored item, `[TAG:words]` = disfluent word already in the ASR text.

Setup notes: `g2p_en` fetches NLTK data (`cmudict`, `averaged_perceptron_tagger[_eng]`) the first time it is imported; NLTK ≥ 3.10 refuses
downloads through an HTTP proxy, in which case unzip those packages from `nltk_data` into `~/nltk_data/{corpora,taggers}` yourself. Hosts
used: OpenSLR (corpus), huggingface.co (+ its CDN hosts; models), download.pytorch.org, nptel.ac.in. Model weights and `data/` are not
committed (`.gitignore`); retraining takes the commands above.

## 6. Results (one run, one seed, `google/electra-small-discriminator`, 13.5 M parameters)

### 6.1 Simulated Switchboard test split — 7,420 held-out chunks, 138,549 reference words

Verbatim word error rate of the restored transcript against the verbatim reference (the ASR baseline already contains the simulated
ASR word errors, which restoration does not touch):

| transcript | WER |
|---|---|
| plain ASR (disfluencies dropped, simulated ASR errors) | 11.64 % |
| restored, tagger without calibration | 9.75 % |
| **restored, tagger + calibration (default)** | **9.20 %** |
| restored, oracle (gold tags) — ceiling of the phones→words stage | 8.62 % |

Span F1 per type (exact span match; `results/test_calibrated.txt` has precision/recall and the oracle/uncalibrated runs):

| tag | gap-phone spans (restored sounds) | restored items: located F1 | … words correct given located | disfluent words already in the ASR text |
|---|---|---|---|---|
| FP filled pause | **0.85** | 0.73 | 0.94 | 0.999 † |
| BC backchannel | 0.77 | 0.79 | 0.86 | 0.996 † |
| EX hesitation | 0.54 | 0.56 | 1.00 | 0.95 † |
| NS non-speech (not restored) | 0.77 | – | – | – |
| REP repetition | 0.48 | 0.48 | 0.63 | 0.82 |
| DM discourse marker | 0.58 | 0.66 | 0.69 | 0.73 |
| EET editing term | 0.44 | 0.46 | 0.60 | 0.58 |
| RPR repair | 0.35 | 0.35 | 0.22 | 0.27 |
| FS false start | 0.26 | 0.43 | 0.18 | 0.07 |
| PW partial word | 0.21 | 0.26 | 0.37 ‡ | 0.99 † |
| MP mispronounced | – | – | – | 0.00 (19 instances; too rare) |
| **micro** | **0.62** | **0.56** | **0.77** | **0.82** |

† these words (`uh`, `uh-huh`, `th-`) identify themselves, so such scores say nothing about the phone evidence.
‡ a fragment counts as right if its spelling is a prefix of the gold one within one letter.
"Located" = right type in the right gap (a repetition may sit at the equivalent neighbouring gap). The oracle-tag run reaches a located
micro-F1 of 0.79 — not 1.0 — because 26 % of dropped items are not found in a gap (located recall 0.74): the simulated recogniser deleted a one-phone
`uh`, or the aligner absorbed it into a neighbour. Calibration (logits − α·log class-weight, α = 1.0 chosen on dev) is worth 0.55 WER
points; restricting which types are inserted does not help (all types: 9.20 %, no RPR/FS/PW: 9.36 %, filled pauses only: 10.19 %).

### 6.2 Real Indian-English audio — NPTEL, no retuning

Phones from `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, transcript from `openai/whisper-small.en`, windows of 4–14 s
(outputs in `results/nptel_summary.json`; the audio is not redistributed).

| clip | audio | ASR words | predicted FP / 1,000 words | human-marked fillers |
|---|---|---|---|---|
| `108105103_lec021` utt. 1–100 | 486 s | 1,599 | 18.8 | 48 (all `ah`) |
| `106106049_lec042` utt. 112–151 (control) | 265 s | 551 | 1.8 | 0 |

Whisper wrote none of the 48 `ah`s (0 filler tokens in its 1,567). Against the human markers in lecture 021, the pipeline's 30 predicted filled pauses hit **12 of 48**
(precision 0.40, recall 0.25, exact position); the lecture whose manual transcript has no fillers gets 1 prediction in 551 words.
Examples: `…multilayer perceptrons as {FP:uh} for classification purpose` (the human wrote `as ah for classification`); the output also contains
spurious `{REP:…}` items.

**This is much weaker than the simulated numbers**, and the cause looks like a simulation-to-real gap rather than a bug. Breakdown of the 48
human fillers by what the pipeline saw at their position: 11 found (12 in the filler check above, which also counts window edges),
**23 leave no unexplained phone** (the recogniser emitted nothing for the filler, or the aligner absorbed it into a neighbouring word),
12 leave a gap phone — mostly a single `aa` ([ɑː], the usual Indian-English "ah") — that the tagger left as `O`, and 2 got another tag.
Training fillers are overwhelmingly schwa-like `uh` (69,814 `uh` vs 353 `ah` in Switchboard).
Precision is a lower bound: a human may not have marked every filler.

## 7. Limitations

* **Training and the headline numbers are on simulated inputs**: both the ASR transcript and the phones are generated from the Switchboard
  transcripts. §6.1 is an in-distribution figure; §6.2 shows the real-audio level (filler precision 0.40 / recall 0.25 on one lecture).
* **Derived types are silver labels** (`REP RPR FS EET DM`), from rules, not Treebank-3 annotation. The model partly learns the rules.
  `MP` is too rare (341 in the corpus) to learn.
* **Weak types**: `PW`, `FS`, `RPR` — a 1–2-phone fragment is hard to tell from recogniser noise, and the words of an abandoned clause are
  rarely recovered right by the lexicon decoder (unigram prior, no language model).
* Phones of a word the ASR *deleted* are labelled `O`, not restored.
* One encoder (ELECTRA-small) and one seed, picked for CPU speed; `--encoder bert-base-uncased` etc. are supported but were not run here.
  No confidence intervals.
* Whisper's behaviour with fillers varies by model and version; the pipeline assumes the transcript is mostly fluent.

## 8. Next steps (not done)

1. **Indian-English fillers in the simulator**: realise `uh/ah` as [ɑː]/[aː] and long [ə] with accent-dependent probability, then re-train and
   evaluate on a lecture *not* used above.
2. **Evidence for fillers with no gap phone**: add an (empty-)gap token per word boundary plus CTC posterior mass / pause duration, so a
   filler the greedy decoder missed can still be flagged.
3. Calibrate the recogniser noise model from real phone output; fine-tune on real audio with weak labels from NPTEL manual transcripts.
4. A bigger encoder on a GPU; a phones→words decoder with a Switchboard language model; plug in Treebank-3/NXT labels for the derived types
   (the labeller is the only place that needs to change).

## 9. Repository layout

`disfrestore/` — `swb.py` corpus reader · `labeler.py` tags · `simulate.py` ASR/recogniser simulation · `lexicon.py` G2P · `align.py` ·
`features.py` · `encoding.py` · `model.py` · `train.py` (+ calibration) · `restore.py` · `evaluate.py` · `infer.py` · `nptel.py` · `download.py` ·
`data.py` · `cli.py`. `docs/METHODOLOGY.md` — the exact methodology and architecture. `tests/` — 30 unit/integration tests. `results/` — evaluation reports, training log, calibration, NPTEL summary.
Existing scripts of the repo (`forced_align.py`, `switchboard_to_rsml.py`, notebooks) are untouched.
