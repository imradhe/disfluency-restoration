# Methodology and architecture (exact)

Everything below is what the code in `disfrestore/` does, with the constants it uses. File names in brackets point to the
implementation. "As run" values are those of the reported experiment (`results/`).

## 0. The task, precisely

**Input** for one utterance window: (a) an ASR transcript `W = w_1 … w_n` (lower-case words, normally *fluent*), (b) the phone
sequence `P = p_1 … p_m` recognised from the same audio, with a start time per phone.
**Output**: for each of the `n+1` *gaps* (the slot before `w_1`, between words, after `w_n`) a possibly empty list of typed
spans of *sounds the transcript does not explain*, each with a Switchboard disfluency type; for each ASR word a tag if the word
is itself disfluent; and, for every typed span, the restored **word string**.

Two stages, as requested:

1. **Restore the missing sounds** — find the recognised phones that the G2P of `W` cannot explain and type them
   (`{FP: ah m}`), giving the transcript at phone level: `so[s ow] {FP: ah m} today[t ah d ey]`.
2. **Convert them into words, with respect to the transcript** — `{FP: ah m}` → `um`; `{REP: dh ah}` before *the* → `the`;
   `{PW: s t}` before *store* → `st-`.

## 1. Data  [`download.py`, `swb.py`]

* Corpus: MSU/ISIP Switchboard transcripts (OpenSLR 5, `switchboard_word_alignments.tar.gz`, 49,651,293 bytes, SHA-256
  `7bbc4dd5…b106`), no audio. 2,438 conversations × sides A/B = 4,876 `*-ms98-a-trans.text` (segment level: `id start end text`) and
  4,876 `*-ms98-a-word.text` (word level: every word and `[silence]` with start/end), plus `sw-ms98-dict.text` (38,888 entries,
  alternative pronunciations, partial-word pronunciations of the *spoken part only*, laughter words, coinages, `_1` reduced variants).
  Fields are split with `split()` (some files use tabs).
* Token categories (`swb.classify`): silence, noise, vocalized_noise, laughter, laughter_word, aside_marker, coinage, mispronounced
  `[x/y]`, partial_word `y[ou]-`/`-[o]kay` (spoken part outside the brackets), cutoff_word `i-`, filled_pause {uh um hm hum ah eh mm},
  backchannel {uh-huh um-hum huh-uh hum-um uh-hum mm-hm mhm}, other_hesitation {huh ooh uh-oh}, pron_variant `them_1`, word.
* Splits (conversation level): test = sw4004–sw4153 (149), dev = sw4519–sw4936 (173), train = the rest (2,116).
* Chunks (`labeler.chunk_side`): consecutive speech-bearing segments of one side are merged while the pause between them is ≤ 1.0 s
  and the chunk has ≤ 40 tokens. One-word acknowledgements (≤ 2 tokens) are kept with probability 0.3 in train only.
  Result: 126,077 train / 8,420 dev / 7,420 test examples, one simulation per chunk (`--variants 1`).

## 2. Phone inventory and phonetic distance  [`phones.py`]

* 39 CMU/ARPAbet phones, lower case, no stress. Switchboard dictionary phones are mapped `ax→ah`, `el→ah l`, `en→ah n`.
* Recogniser IPA (espeak style) → ARPAbet by greedy longest match over a table (affricates `tʃ dʒ`, diphthongs `aɪ aʊ eɪ oʊ ɔɪ`,
  retroflex `ʈ ɖ ɳ ʂ ɽ → t d n sh r`, flap `ɾ→d`, `ʔ` dropped, syllabic `n̩ l̩ m̩ → ah n/l/m`); stress, length, aspiration and other
  diacritics are removed; unmapped symbols are counted and dropped.
* Articulatory distance `d(a,b)∈[0,1]`: vowels from (height, backness, rounding) — `min(0.9, sqrt(Δh²+Δb²+(0.5Δr)²)/1.1 + 0.05)`
  (diphthongs = mean of their two targets); consonants from (manner 0–5, place 0–7, voicing) —
  `min(1, 0.55·|Δplace|/7 + 0.33·min(1,|Δmanner|/4) + 0.12·|Δvoice| + 0.05)`; vowel/consonant pairs 1.0 except glides and
  liquids close to vowels (y–iy/ih 0.5, w–uw/uh 0.5, r–er 0.45, l–uh 0.8, l/n/m–ah 0.85, hh–ah 0.8).
* Substitution cost `SUB(a,b) = 0` if `a=b`, else `0.25 + 0.95·d(a,b)` (range 0.41 for t/d … 1.2). Voicing flips are therefore the cheapest
  consonant confusion.

## 3. Disfluency taxonomy and labelling  [`labeler.py`]

Tags (23 BIO labels = `O` + `B-/I-` × 11): `FP EX BC PW MP NS` (native MS98 markup) and `REP RPR FS EET DM` (rule-derived, **silver**).
A *unit* is one non-silence token; `word` = what an ASR would write (variant suffix stripped, `[laughter-yeah]→yeah`,
`{x}→x`, `[spoken/intended]→intended`), `spoken` = what was said (fragments keep only the spoken part).
Rules run in this order on the lexical stream (fillers, fragments, noise skipped; closed-list editing-term phrases removed from the
stream so they are transparent):

1. **REP** — longest-first (n = 5…1) exact n-gram repetition `A A`; the first copy is the reparandum (`rm`), the second the repair
   (`rr`); chained (`the the the` → first two copies tagged); skipped if the copies are > 1.5 s apart, if the n-gram is in
   `{bye, very, tut, hee, ha, ho}`, or if the first copy is already tagged.
2. **RPR** — a repair that restarts the reparandum with changes: reparandum length 2…6, repair starts with the same word; `lead` =
   number of leading words shared; accepted if (`lead ≥ 2` and `n − lead ≤ 2`) or (`lead = 1`, a cue between the two — filler,
   fragment, mispronunciation, a pause ≥ 0.25 s or a segment break — and `n ≤ 2`); `lead = n` is a REP, not an RPR.
3. **FS** — for each fragment `PW` that is followed, in the same segment, by a lexical word that does not start with the fragment's first
   two letters, is not itself a fragment, and is in a restart-starter list or repeats one of the preceding words: up to 4 preceding
   untagged lexical words of the segment, stopping after a clause starter {and but so because or then that if when while}, are `FS`.
   Utterance-final cut-offs and stutters on the same word are *not* false starts.
4. **EET / DM** — closed list {you know, i mean, you see, or rather, excuse me, sorry}: `EET` if the phrase lies between a reparandum and
   its repair; else `DM` for {you know, i mean, you see} at a segment edge, next to a filler/fragment, or beside a pause ≥ 0.25 s;
   else `EET` for {or rather, excuse me, sorry} next to a filler/fragment or pause. Single-word markers (*well, like, so*) are not labelled.

Native tags: `FP` (filled_pause), `EX` (other_hesitation), `BC` (backchannel), `PW` (partial and cut-off words), `MP`
(mispronounced), `NS` (noise, vocalized noise, laughter). Over the corpus: REP 25.0, RPR 3.9, FS 1.2 instances per 1,000 words.

## 4. Simulating the pipeline's two inputs  [`simulate.py`, `data.py`]

No audio exists, so each chunk is turned into `(W, P)` by two seeded random processes (seed = md5 of seed|conversation|side|chunk|variant).
Per example a **profile** is drawn: `verbosity` = 0 (p 0.50), U(0.1,0.5) (p 0.35), U(0.5,0.9) (p 0.15); `noise ~ U(0,1)`;
`asr_sub ~ U(0,0.08)`, `asr_del ~ U(0,0.03)`, `asr_ins ~ U(0,0.01)`; `accent = 0` (p 0.4) else `U(0.3,1)`.

**ASR transcript `W`.** `NS` units never appear. A unit with tag t is dropped with probability `BASE_DROP[t]·(1 − verbosity)`,
`BASE_DROP = {FP .95, EX .90, BC .60, PW .90, REP .75, RPR .85, FS .90, DM .05, EET .10}`. Retained fragments are written with a
trailing hyphen (`th-`); `MP` words are written as the intended word. Units tagged `O` are deleted w.p. `asr_del`, or replaced w.p.
`asr_sub` by a frequency-weighted near-homophone (G2P edit distance 1, from the 20,000 most frequent words, found by deletion/
substitution hashing), and a random function word (12 candidates) is inserted after a word w.p. `asr_ins`.

**Recognised phones `P`.** For every unit (including dropped ones) the dictionary pronunciation (variant uniform; G2P if absent) is
generated, then per phone: accent replacement (`th→t .6, dh→d .6, v→w .35, w→v .2, z→s .3, ae→eh .2, ao→aa .25, ey→eh .2, ow→ao .2,
ih→iy .15, uh→uw .2`, each × accent strength), deletion w.p. `p_del` (×1.5 for ah ih t d hh dh), substitution w.p. `p_sub` to another phone
drawn ∝ `exp(−d/0.18)`, insertion w.p. `p_ins` (a frequent phone or a duplicate), with
`p_sub = 0.05+0.22·noise`, `p_del = 0.02+0.10·noise`, `p_ins = 0.005+0.04·noise`. `NS` units give 1–4 junk phones (from 12) w.p. 0.7.
The phones of a unit share its real `[start,end]` from the word alignments (≥ 0.02 s per phone). Every phone remembers its source unit.

**Gold labels from provenance.** After alignment (§6) each unexplained phone gets a label from its source unit: a dropped unit →
its item's tag; a retained unit that belongs to a REP/RPR group → that group's tag (which of the two equal copies is left in the gap is
a tie, so reparandum and repair copy share one span key); otherwise `O` (phone noise, or the phones of a word the ASR deleted).
Consecutive phones with the same span key are `B-`/`I-` tagged. ASR words get BIO labels from their own unit's tag (retained
disfluent words). **Items** = dropped units with a restorable tag, in reference order, with their gap index and word string
(consecutive dropped units of one REP/RPR/FS group merge). `covered` = ≥ 60 % of the item's phones were left in a gap.
The verbatim reference `ref` = surface words incl. fragments, without `NS`.

## 5. G2P  [`lexicon.py`]

Special table (uh, um, hm, uh-huh, …) → CMUdict first pronunciation → hyphen split → digits via g2p_en's normaliser → possessives of
known words → g2p_en neural model for OOV; stress digits removed, only the 39 phones kept; cached and persisted so workers do not
re-run the network.

## 6. Phone alignment — the misalignment signal  [`align.py`]

Reference string `g` = concatenated G2P phones of `W` (length N, word ends known); recognised string `r` (length M).
Row `i` = number of reference phones consumed; rows 0 and every word end are *boundary rows*.

```
D[0][j]   = j · C_GAP
D[i][j]   = min( D[i-1][j-1] + SUB(g_i, r_j),      # match / substitution
                 D[i-1][j]   + C_DEL,              # reference phone not recognised
                 D[i][j-1]   + c_i )               # recognised phone not explained
c_i = C_GAP = 0.45 on boundary rows, C_INT = 0.90 inside a word;   C_DEL = 0.75
```

Solved row-wise with `np.minimum.accumulate(cand − c·j) + c·j` for the insertion closure. Backtrace prefers diagonal, then deletion,
then insertion, which places ambiguous insertions as **far left as possible** (an extra copy of a word appears *before* the retained one =
reparandum before repair). Output per word: `match, sub, dele, ins` (inside), `sub_cost`, first/last aligned recognised phone; per gap: the
recognised-phone indices inserted at that boundary row (the *unexplained* phones). A 50-word chunk aligns in ≈ 1.7 ms.

## 7. Features  [`features.py`]

Recognised-phone duration `dur_j = min(start_{j+1} − start_j, 0.4)`, `pause_after_j = max(start_{j+1} − start_j − 0.4, 0)`.
`chunk_cost = min(1, D[N][M]/N)`.
**Word (FW = 11):** `n_ref/8` (≤ 1.5), match, sub, del, ins (each ÷ n_ref), mean sub cost/1.2, pause before/2, pause after/2 (clipped 1),
speaking rate `span/(0.08·n_ref)` clipped 3 then /3, `absent` flag, `chunk_cost`.
**Gap phone (FG = 9):** `dur/0.4`, `pause_after/2`, `pause_before/2`, position in run, first-in-run, last-in-run, run length/10 (≤ 1.5),
run duration/2 (≤ 1), `chunk_cost`. Only relative quantities are used so that simulated phones (word alignments) and CTC phones
(real audio) are comparable.

## 8. Sequence and model  [`encoding.py`, `model.py`]

**Token sequence** (max 320 tokens in training, 500 at inference):

```
[CLS]  g0-phones  w1-pieces  g1-phones  w2-pieces … wn-pieces  gn-phones  [SEP]
```

Word pieces come from the encoder's WordPiece tokenizer; each of the 39 phones is a new special vocabulary entry `[ph_xx]`
(vocabulary 30,522 + 39 = 30,561). Per token: `kind ∈ {special 0, first word-piece 1, later word-piece 2, gap phone 3}`; features
vector of size 11 (word tokens: the word's FW features repeated on every piece; gap phones: FG features, zero-padded); label id
(word label on its first piece only; gap-phone label per phone; others −100).

**Network** (`Tagger`, 13.51 M parameters, as run):

```
e_t = E_tok[id_t]                         # pretrained ELECTRA-small embedding (128-d); new phone rows = mean of the
      + E_kind[kind_t]                    #   embeddings of the word-pieces of the phone's name; E_kind: 4×128, N(0,0.02)
      + MLP(feat_t)                       # Linear(11→128) – GELU – Linear(128→128); last layer zero-initialised
h = ELECTRA(inputs_embeds = e, attention_mask)   # 12 layers, hidden 256, 4 heads, FFN 1024, learned positions (≤512);
                                                 # the encoder adds position/type embeddings, LayerNorm and 128→256 projection
logits = Linear(256→23)(Dropout(0.1)(h))         # one BIO label per token
```

Zero-initialising the feature MLP's last layer makes the model start as the plain pretrained encoder and learn to use the alignment
evidence. Any HF encoder can replace ELECTRA-small (`--encoder`); checkpoints are self-contained (weights, config, tokenizer).

## 9. Training  [`train.py`]

Loss: token-level cross-entropy with class weights `w_k = min((n_O/n_k)^0.5, 20)` (`n_k` = label counts in train), then divided by the mean
of the non-`O` weights (clamped ≥ 1); −100 ignored. AdamW (β default), lr 2e-4, weight decay 0.01 (none on bias/LayerNorm), batch 64,
2 epochs = 3,940 steps, 300 warm-up steps then linear decay to 0, gradient clipping 1.0, length-bucketed batches (sort inside windows of
50 batches), seed 0, 4 CPU threads, ≈ 2,400 s. Every 500 steps: dev span-F1 on 3,000 dev examples; the checkpoint with the best mean of
(gap micro-F1, word micro-F1) is kept (last step: gap 0.565, word 0.729 uncalibrated).

## 10. Calibration  [`train.py: calibrate`]

Class-weighted CE over-predicts rare tags. Logits are corrected as `z' = z − α·log w` and α ∈ {0, .25, .5, .75, 1, 1.25} is chosen to
maximise the mean of gap and word micro span-F1 on the whole dev split (class counts from the first 40,000 training examples):
α = 1.0 (dev mean F1 0.651 → 0.717). The bias vector is stored in `calibration.json` and applied by `evaluate` and `infer`.

## 11. Span decoding  [`encoding.decode_spans`]

`argmax` per token, then BIO decoding per run (each gap, and the word sequence): `B-X` starts a span, `I-X` continues an `X` span, an orphan
`I-X` starts one. A span's confidence (inference) = mean of the maximum softmax probability over its tokens; spans below 0.5 are dropped.

## 12. Restoring words from typed phone spans  [`restore.py`]

Let `q` be the span's phones, `g` its gap index. `edit_cost` = the feature-weighted Levenshtein distance (indel 0.75, `SUB`),
`norm_cost = edit_cost / (0.75·max(len))`.

* **FP / EX / BC** — nearest entry of the inventory (`FP`: uh=ah, um=ah m, hm=hh m, ah=aa, eh=eh; `EX`: huh, ooh, uh-oh; `BC`: uh-huh, um-hum,
  huh-uh), cost = `edit_cost` + 0.04·(−ln relative Switchboard frequency).
* **REP** — candidates = the 1…5 ASR words after the gap or before it; pick min `norm_cost` against the concatenated G2P; accept if ≤ 0.55
  (else fall through to the lexicon decoder).
* **PW** — for the next word and the previous word, the phone prefix `G2P(w)[:m]` (all `m`) with minimal normalised cost; accept if ≤ 0.55;
  spelled prefix length `round(len(w)·m/len(G2P(w)))` clipped to [1, len−1] → `st-`; otherwise `[ph ph]-`.
* **RPR / FS / EET / DM / other** — lexicon decoder: the 6,000 most frequent Switchboard words (pronunciation from the dictionary), dynamic
  programming over segmentations of `q`; a span `q[s:e]` costs `edit_cost(q[s:e], pron(w)) + 0.55 (word penalty) + 0.10·(−ln p(w))`, evaluated
  for all words at once with a vectorised Levenshtein; accepted if the cost per phone ≤ 0.75, else the span stays `[ph ph]`.
* **NS** — kept as an event, never inserted.

Plain restored transcript = ASR words with the restored items inserted in their gaps. `NS` events and unresolved items that stayed
`[ph ph]` are left out of it (a bracketed phone string is never the right word) but remain in the tagged transcript and the JSON;
the one exception is an unresolved partial word, which stays in the plain transcript as a `[ph ph]-` fragment marker.

## 13. Evaluation protocol  [`evaluate.py`]

* **Span F1** (exact span match, per tag, micro over tags), separately for gap-phone spans and for ASR-word spans.
* **Restored items**: gold items vs restored items (NS excluded), matched greedily one-to-one on (tag, gap); a REP/RPR also matches at a
  gap ≤ (its length in words) away if the words are identical. *Located* P/R/F1 count matches; *text|located* = restored words identical
  (partial words: one is a prefix of the other, length difference ≤ 1).
* **Verbatim WER**: Levenshtein WER of the restored transcript against the verbatim reference, vs plain ASR as baseline; **oracle** = same with
  gold tags; policy table = WER when only some types are inserted.

## 14. Inference on audio  [`infer.py`, `nptel.py`]

1. Decode to 16 kHz mono (ffmpeg). **Windows**: 10 ms frame energy in dB; pause = run of frames < (10th percentile + 7 dB) lasting ≥ 0.2 s;
   from the current start cut at the longest pause (coarsely, then the latest) whose centre lies 4–14 s ahead, otherwise at the quietest frame
   in the last 2 s before 14 s. Both recognisers see the same windows.
2. **Phone recogniser**: `facebook/wav2vec2-xlsr-53-espeak-cv-ft`; waveform standardised (zero mean, unit variance); greedy CTC (argmax per 20 ms
   frame, collapse repeats, drop `<pad>`/`<s>`/`</s>`/`<unk>`/`|`); each token → ARPAbet (§2); start time = first frame of the run × 0.02 s.
3. **ASR**: `openai/whisper-small.en` through the transformers pipeline; text lower-cased, punctuation removed (apostrophes and inner hyphens
   kept) → `W`.
4. G2P of `W` → alignment with the recognised phones (§6) → features (§7) → tagger + calibration (§8–11) → restoration (§12).
5. Output: JSON/TXT with ASR text, **restored phone sequence**, restored words, tagged transcript (`{TAG:text}` restored item,
   `[TAG:words]` disfluent word already in the ASR text), per-1,000-word rates.
   Tail rule: a remaining tail shorter than 2 s is merged into the previous window, a longer one becomes its own window.
6. Test audio: `speechlab/NPTEL_TECH_ENG_50` test shard 1; consecutive ~7 s utterances of one lecture are concatenated (`fetch-nptel`).
   `check_fillers` compares predicted filled pauses with the fillers a human wrote in the manual transcript (clean reference aligned to
   the ASR words by Levenshtein; ±1-word tolerance).

## 15. What is heuristic or simulated (read before citing numbers)

* `REP RPR FS EET DM` are rule-derived silver labels; `MP` has 341 corpus instances.
* Both inputs of the training data are simulated; the recogniser noise levels were set from one measured comparison (real vs simulated gap rate and
  alignment cost) and from literature-style assumptions, not fitted to the target recogniser.
* All reported simulated-test numbers are in-distribution; the real-audio check covers one annotated lecture.
* One seed, one encoder; no confidence intervals.
