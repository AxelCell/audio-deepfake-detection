# ROADMAP.md — phases 2 to 8

Technical detail for everything after phase 1. Phase 1 is in `BUILD.md`. The
process — kickoff, handoffs, reports, closing a phase — is in `PROJECT.md`.

**How to use this file:** we point you at one section at a time. "Read
ROADMAP.md section P3.2 and build it" works. Pasting seventeen weeks of
instructions at once produces worse code than pasting one section.

Section ids are `P<phase>.<task>`.

---

# Phase 2 — front-ends and caching (weeks 4–5)

Delivers: all three front-ends implemented, features cached to disk, a working
PyTorch DataLoader.

## P2.1 — LFCC front-end
**Owner: Tania**

`src/features.py` gains `lfcc(wav, config)`.

Linear Frequency Cepstral Coefficients. Same pipeline as MFCC but with a
**linear** filterbank instead of a mel one, so high frequencies keep their
resolution. That matters because vocoder artefacts concentrate up there — the
mel scale's whole purpose is to discard detail we can't hear, which is exactly
the detail we want.

Pipeline: pre-emphasis → frame → window → power spectrum → linear filterbank →
log → DCT → keep coefficients 0–19 → append delta and delta-delta, giving 60
dimensions per frame.

```yaml
features:
  lfcc:
    n_filters: 20
    n_ceps: 20
    with_deltas: true
    pre_emphasis: 0.97
```

Use `scipy.fftpack.dct` with `type=2, norm='ortho'`. Deltas via
`librosa.feature.delta`.

Add a comment: this is a Python reimplementation, not the official MATLAB
baseline, so absolute numbers may differ slightly from published LFCC results.

## P2.2 — CQCC front-end
**Owner: Tania**

`cqcc(wav, config)`.

Constant-Q Cepstral Coefficients, from Todisco et al. 2017 (reference [4]). The
constant-Q transform uses a variable time-frequency resolution — fine frequency
resolution at low frequencies, fine time resolution at high ones — which matches
how speech actually behaves better than a fixed FFT does.

Pipeline: `librosa.cqt` → power → log → **uniform resampling** of the log-spaced
CQT bins onto a linear scale → DCT → keep 0–19 → deltas.

That resampling step is what makes CQCC different from a naive log-CQT, and it
is the step people get wrong. Explain it before implementing it.

```yaml
features:
  cqcc:
    bins_per_octave: 96
    f_min: 15
    n_bins: 96
    resample_factor: 16
    n_ceps: 20
    with_deltas: true
```

CQCC is slow — expect it to dominate feature extraction time. Same
reimplementation caveat in the comments.

## P2.3 — Feature caching
**Owner: Prateek**

`scripts/extract_features.py`.

Extracting features on every training run wastes hours. Extract once, save to
disk, load from then on.

- Writes to `data/features/<front_end>/<partition>/<utt_id>.npy`
- Skips any file that already exists, so an interrupted run resumes
- `tqdm` progress bar
- Parallelise with `multiprocessing.Pool` over CPU cores
- Flags: `--front-end logmel|lfcc|cqcc`, `--partition train|dev|eval`
- Writes `manifest.json` per front-end recording the exact config used

The manifest matters: if someone changes `n_mels` and reruns, the cache is stale
and silently wrong. The loader must compare the manifest against the current
config and refuse to run on a mismatch. Silent staleness is the worst kind of
bug in this project because nothing errors and every number is wrong.

**Handoff:** running this takes hours of CPU across three front-ends and three
partitions. Expect tens of GB on disk. Flag it and suggest running log-mel first
so phase 3 isn't blocked on CQCC finishing.

## P2.4 — PyTorch Dataset
**Owner: Pranjal**

`src/dataset.py`.

```python
class ASVspoofDataset(torch.utils.data.Dataset):
    def __init__(self, partition, config, augment=False)
    def __len__(self)
    def __getitem__(self, idx) -> (tensor, label)

def make_loaders(config) -> (train_loader, dev_loader, eval_loader)
```

Reads the protocol, loads the cached `.npy`, returns a float32 tensor shaped
`(1, n_features, n_frames)` — the leading 1 is the channel dimension the CNN
expects — plus the label as a float.

Shuffle train only. Set `num_workers` from config; on Windows it must be 0 or
you get spawn errors.

The `augment` flag does nothing yet — it's wired up in phase 4.

## Phase 2 checklist

- [ ] All three front-ends implemented, with reimplementation noted in comments
- [ ] Features cached for all three partitions, at least for log-mel
- [ ] Manifest written, and a stale cache is detected and refused
- [ ] DataLoader yields correctly shaped batches
- [ ] Disk usage checked and reported

---

# Phase 3 — baseline model and first number (weeks 6–8)

The milestone. Everything before this is plumbing, everything after is
variations.

## P3.1 — The model
**Owner: Pranjal**

`src/model.py`.

```python
class BaselineCNN(nn.Module):
    def __init__(self, config)
    def forward(self, x) -> logit
```

Four convolutional blocks, each `Conv2d(3x3) → BatchNorm2d → activation →
MaxPool2d(2x2)`. Channels 32 → 64 → 128 → 128. Then adaptive average pooling
across both dimensions, dropout, and a linear layer to **one** output.

One output, not two. This is binary classification and we want a single score
per utterance, which is what EER is computed over.

Adaptive pooling is what lets the model accept variable input sizes — needed for
experiment E4 in phase 5.

```yaml
model:
  channels: [32, 64, 128, 128]
  activation: relu     # 'mfm' used in E3
  dropout: 0.3
```

Also implement Max Feature Map activation for later use in E3: split the channel
dimension in half and take the elementwise max. It comes from the LCNN work
(reference [5]) and acts as a learned feature selector.

Print the parameter count. The baseline should be small — a few hundred
thousand parameters, not millions.

## P3.2 — Training loop
**Owner: Aishwary**

`src/train.py`.

```
python -m src.train --config config.yaml --name baseline_logmel
```

- Adam, initial LR from config, `ReduceLROnPlateau` on dev EER
- `BCEWithLogitsLoss` — numerically stabler than sigmoid plus BCE separately
- After each epoch: score the whole dev set, compute dev EER
- **Early stopping on dev EER, not on loss.** Loss can improve while
  discrimination gets worse.
- Checkpoint every epoch to `checkpoints/<name>/` so a dropped session doesn't
  cost the run; keep the best-dev-EER checkpoint separately
- Log to `results/<name>/train_log.csv`: epoch, train loss, dev EER, LR, time
- Write `results/<name>/config_snapshot.yaml` — the exact config used, so a
  result can always be traced back
- `--resume` flag to continue from the last checkpoint
- Set the seed from config at the start and record it

```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 10
  num_workers: 0
```

**Handoff:** every training run is hours on a GPU. This is the dominant handoff
for the rest of the project.

## P3.3 — EER and min t-DCF
**Owner: Prateek**

`src/metrics.py`. Get this right; every number in the project flows through it.

```python
compute_eer(bonafide_scores, spoof_scores) -> (eer, threshold)
compute_det_curve(bonafide_scores, spoof_scores) -> (frr, far, thresholds)
```

EER: sort all scores, sweep the threshold, compute false rejection rate and
false acceptance rate at each point, find where they cross. Interpolate rather
than taking the nearest point.

**Sanity-check it before trusting it, as tests:** random scores must give
roughly 50% EER; perfectly separated scores must give 0%. If this function is
subtly wrong, every number in the project is wrong and nothing will tell you.

For **min t-DCF**, use the official ASVspoof 2019 Python script
(`evaluate_tDCF_asvspoof19.py`) unmodified, so results stay comparable with
published work. It needs ASV scores, which ship in
`data/LA/ASVspoof2019_LA_asv_scores/`. Vendor it into `src/tdcf/` with its
licence header intact and note in the README that it is unmodified.

## P3.4 — Evaluation
**Owner: Tania**

`src/evaluate.py`.

```
python -m src.evaluate --checkpoint checkpoints/baseline_logmel/best.pt
```

Scores a partition, writes `results/<name>/scores_<partition>.txt` in ASVspoof
format (`utt_id score`), computes EER and min t-DCF, and produces a per-attack
breakdown — EER for each of A07 through A19 separately.

The per-attack breakdown is a promised deliverable. A good average EER can hide
total failure on one synthesiser, and that failure is usually the most
interesting thing in the results.

## P3.5 — The baseline result
**Owner: Aishwary**

Run it. Log-mel, 4-second input, plain BCE. Report the number **in full,
whatever it is**, in the README results table before touching anything.

A baseline that gets tuned before it's reported makes every later comparison
meaningless, and it cannot be fixed retroactively. This is the single most
important discipline in the project.

Expect somewhere in the 5–15% EER range for a small CNN on log-mel. Published
state of the art is under 1%, but those systems use large self-supervised
front-ends. **If you get under 3%, check for a bug** — most commonly a label
inversion or a train/eval leak.

## Phase 3 checklist

- [ ] Model built, parameter count reported
- [ ] Training loop with checkpointing and early stopping on dev EER
- [ ] EER sanity-checked against random and perfect scores, as tests
- [ ] Official t-DCF script vendored unmodified
- [ ] **Baseline reported in full before any tuning**
- [ ] Per-attack breakdown produced
- [ ] One filled row in the README results table

---

# Phase 4 — E1 and E2 (weeks 9–11)

Each experiment changes **exactly one variable** against the baseline.
Everything else — partitions, optimiser, schedule, seed — stays fixed.

## P4.0 — Experiment infrastructure
**Owner: Aishwary**

`configs/experiments/` holds one YAML per run, each inheriting from
`config.yaml` and overriding one block. `scripts/run_experiment.py` takes a
config path and does train → evaluate → append a row to the results table
automatically.

Run each configuration with **3 seeds** where time allows and report mean and
spread. A single lucky run is not evidence, and distinguishing a real effect
from run-to-run variance is part of what this project is meant to demonstrate.

## P4.1 — E1, front-end comparison
**Owner: Aishwary**

Three runs: log-mel, LFCC, CQCC. Identical everything else.

Answers: which representation exposes synthesis artefacts best? The literature
says the front-end has often mattered more than the classifier — this tests that
on our own data.

## P4.2 — E2, data augmentation
**Owner: Tania**

Against the winning front-end from E1. Four runs: none (the E1 result),
SpecAugment, signal-level distortion, both.

SpecAugment (reference [11]): mask random bands of time and frequency in the
spectrogram. Roughly 2 frequency masks up to 20 bands, 2 time masks up to 40
frames. Cheap, about 15 lines.

Signal-level: additive noise at random SNR, random band-pass filtering, and
codec simulation. This is the RawBoost idea (reference [10]) in simplified form.

Augmentation applies to **training only**, never dev or eval. Wire it to the
`augment` flag in `ASVspoofDataset`.

This is the experiment most likely to matter for the cross-dataset test in phase
6, since it targets channel robustness directly.

## Phase 4 checklist

- [ ] Experiment runner works and appends results automatically
- [ ] E1 complete, 3 front-ends, seeds where possible
- [ ] E2 complete, 4 conditions
- [ ] Augmentation verified to apply to training only
- [ ] Results table updated with spread

---

# Phase 5 — E3, E4, E5 (weeks 12–14)

## P5.1 — E3, model capacity
**Owner: Pranjal**

Three runs: the shallow baseline, a deeper residual CNN, and the baseline with
Max Feature Map activations instead of ReLU.

Answers: does capacity help, or does the model just memorise the six training
attacks harder? Watch the gap between dev and eval EER, not just eval EER — a
widening gap is the signature of overfitting to known attacks.

## P5.2 — E4, input duration
**Owner: Prateek**

Three runs: 2 s, 4 s, full utterance (padded to the batch maximum).

Answers: how much audio does a reliable decision actually need? This is the most
practically useful result in the set — it directly determines latency in any
real deployment.

## P5.3 — E5, loss weighting
**Owner: Aishwary**

Three runs: plain BCE, class-weighted BCE (`pos_weight` set to the
bona-fide:spoof ratio), focal loss (γ=2).

Answers: how should the 1:9 imbalance be handled? Note that EER is already
threshold-free, so the effect may be smaller than intuition suggests — a
negative result here is a perfectly good finding and must be reported.

## Phase 5 checklist

- [ ] E3, E4, E5 complete with seeds where possible
- [ ] Dev-vs-eval gap reported for E3
- [ ] Results table updated
- [ ] Any negative results recorded, not quietly dropped

---

# Phase 6 — E6 and cross-dataset (weeks 15–16)

## P6.1 — E6, score fusion
**Owner: Aishwary**

Take the two strongest systems that make *different* kinds of errors — check the
correlation between their score vectors first; fusing two near-identical systems
gains nothing, and that check is itself a reportable result.

Fuse at score level: fit logistic regression on **dev** scores, apply to eval.
Never fit on eval.

Answers: are the errors complementary?

## P6.2 — Cross-dataset test
**Owner: Prateek**

Take the single best configuration. Score ASVspoof 2021 **with no retraining and
no adaptation**.

Expect substantial degradation. The point is to quantify it, not avoid it. The
gap between the 2019 number and this one is a headline result in its own right —
it says how much of our within-dataset performance is genuine detection versus
familiarity with six specific synthesisers.

ASVspoof 2021 eval data is on Zenodo. If we have lab GPU access and time, the
ASVspoof 5 evaluation set (Zenodo, 142 GB total — the eval tarballs alone are
enough) is a harder and more current alternative. **Ask Dr. Kumar before
substituting it**, since the synopsis names 2021.

## Phase 6 checklist

- [ ] Score correlation between candidate systems checked and reported
- [ ] E6 fusion fitted on dev only
- [ ] Cross-dataset test run with no retraining
- [ ] 2019 vs cross-dataset gap reported as a headline number

---

# Phase 7 — analysis and figures (weeks 17–18)

## P7.1 — Plots
**Owner: Prateek (curves), Tania (training plots and polish)**

`scripts/make_plots.py`:

- DET curve overlaying all main configurations, log-scale axes, EER diagonal
  marked
- Bar chart of per-attack EER for the best system, sorted worst first
- Score distribution histograms, bona fide vs spoof
- Training curves: dev EER against epoch for each experiment
- Bar chart of 2019 vs cross-dataset EER

All to `figures/`, committed, 150 dpi, readable in greyscale print.

## P7.2 — Error analysis
**Owner: Pranjal, assembled by Aishwary**

`results/analysis.md`, written by us rather than generated:

- Which attacks remain hardest, and a hypothesis about why. Look at what
  generated A17, A18 and A19 — the hardest in the literature.
- **Listen to several false accepts and false rejects and describe what you
  hear.** This is a handoff; you can't do it.
- What the cross-dataset gap tells us
- **Which techniques produced no improvement.** An explicit deliverable in the
  synopsis. Examiners notice when negative results are quietly dropped.

## Phase 7 checklist

- [ ] All five figure types produced and committed
- [ ] Per-attack analysis written with hypotheses
- [ ] Error cases listened to and described
- [ ] Negative results section written

---

# Phase 8 — prototype, docs, report (weeks 19–20)

## P8.1 — Streamlit prototype
**Owner: Tania**

`app.py`. Upload an audio file → score, verdict against the dev-selected
threshold, and a display of the spectrogram the model actually processed.

Must import from `src/` and call the same preprocessing and feature functions as
training. If the app computes features differently from training, it will fail
for reasons that look like model failure.

Show the score, not just the verdict, and state the threshold. Add a line saying
this is a research prototype, not a forensic tool.

## P8.2 — Repository finalisation
**Owner: Pranjal (walkthrough), Prateek (reproducibility check)**

- README results table complete, with a reproduction command per row
- `docs/WALKTHROUGH.md` explaining every module in plain language
- All 13 literature notes present
- `requirements.txt` frozen with exact versions
- A fresh clone runs end to end — actually test this on another machine

## P8.3 — Report
**Owner: Aishwary, everyone contributes**

The synopsis maps onto report chapters. Sections 1–3 are written; section 4
needs updating to describe what we built rather than what we planned; sections
5–7 need the real numbers.

**Do not leave the report to week 20.** Write each experiment up as it finishes,
while the reasons for each configuration are still fresh.

## Phase 8 checklist

- [ ] Streamlit app shares training's code path
- [ ] Walkthrough written
- [ ] Fresh clone reproduces every number
- [ ] All figures and tables in the report
- [ ] Report submitted

---

# The three ways this goes wrong

**Tuning the baseline before reporting it.** Every comparison afterwards becomes
meaningless and it cannot be fixed retroactively. Report the ugly number first.

**Touching the eval partition more than once.** Every peek is a tuning decision.
Dev is for tuning; eval is scored at the end.

**Leaving the report to week 20.** Write each experiment up as it finishes.
