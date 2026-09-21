# LEARN.md — what to study, and when

Learn each topic just before you need it. Don't read ahead; it won't stick.

Roughly one topic every two or three days, studied alongside the matching phase.
If you can explain a topic to a teammate without notes, you know it.

This file is mainly for us, not for Claude — though Claude can use it to pitch
explanations at the right level.

---

## Phase 1, weeks 1–2 — how audio becomes numbers

Needed for: `BUILD.md` steps 6 and 8.

**Waveforms and sampling.** A sound is air pressure over time. A recording is
that pressure measured 16,000 times a second. Understand what sample rate means,
why 16 kHz is enough for speech (speech energy mostly sits below 8 kHz), and
what the Nyquist limit is.

**Frames, windows, hop.** We don't analyse a whole recording at once — we chop
it into overlapping 25 ms slices, stepping forward 10 ms each time. Understand
why: speech changes too fast to treat as one block, but is roughly stable inside
25 ms.

**The Fourier transform, intuitively.** Any signal can be described as a sum of
pure tones. The FFT tells you how much of each frequency is present. You don't
need the mathematics — you need to know what the output means.

**Spectrograms.** Run an FFT on every frame and stack the results: a picture
with time across and frequency up, brightness showing energy. This is the image
our CNN will look at.

**The mel scale.** Human hearing resolves low frequencies much more finely than
high ones. The mel scale warps the frequency axis to match. Also learn the
counter-argument, because it's in our own literature review: the mel scale
throws away high-frequency detail, and that's exactly where AI vocoder traces
tend to live. This is why experiment E1 compares mel against LFCC and CQCC
instead of assuming mel is best.

**How to actually learn it:** plot things. Load one utterance. Plot the
waveform. Plot the raw spectrogram. Plot the mel spectrogram. Look at what
changed at each step. Then listen to a real and a fake clip back to back and
notice you can't tell them apart — that's the whole motivation for the project.

---

## Phase 1, weeks 2–3 — the problem and the metric

Needed for: understanding what we're measuring.

**Read reference [1]** — Todisco et al., *ASVspoof 2019: Future Horizons*. Short
paper, gives you the vocabulary: bona fide, spoof, countermeasure, logical
access, known versus unknown attacks.

**Equal Error Rate.** The model outputs a score, not a verdict. Slide a
threshold from one end to the other: at one extreme you accept everything, at
the other you reject everything. EER is the point where the fraction of real
clips wrongly rejected equals the fraction of fakes wrongly accepted. One
number, no threshold needed. Lower is better; published systems get below 1% on
this dataset.

**Why accuracy is useless here.** Our data is about 1:9 real to fake. A model
that outputs "fake" for every input scores 90% accuracy while being completely
worthless. Make sure this is genuinely obvious to you — it justifies half the
design decisions in the project.

**min t-DCF, lightly.** It scores our detector in combination with a speaker
verification system, reflecting how a countermeasure gets deployed in reality.
Know what it's for; the details can wait for phase 3.

---

## Phase 2, weeks 4–5 — cepstral features

**What a cepstrum is.** Take the log of a spectrum, then transform it again with
a DCT. This separates the slowly-varying envelope (the vocal tract shape) from
the fast-varying fine structure (the pitch and excitation). Vocoder artefacts
live in the parts of that separation a mel spectrogram blurs together.

**Why LFCC and CQCC exist.** LFCC keeps high-frequency resolution that mel
discards. CQCC uses a variable time-frequency resolution that matches speech
better than a fixed FFT. Read reference [4] for the CQCC argument.

**Deltas.** First and second differences across time, capturing how features
change rather than their instantaneous value.

---

## Phase 3, weeks 5–8 — PyTorch and CNNs

Start this early. It's the biggest gap for most people and phase 3 is where the
project stalls if you're learning PyTorch and debugging a model at the same
time.

**PyTorch basics.** Tensors and how they differ from numpy arrays. `nn.Module`
and how a model is just a class with a `forward` method. Autograd — you write
the forward pass and the gradients come free. The shape of a training loop:
batch in, forward, loss, `backward()`, `optimizer.step()`, repeat. `Dataset` and
`DataLoader`.

**How to learn it:** the official PyTorch 60-minute blitz, then build one small
CNN that classifies MNIST digits end to end yourself. That single exercise
covers almost everything we need.

**CNNs on spectrograms.** Why a 2D convolution over a time-frequency picture
makes sense: a filter learns a small local pattern and slides it everywhere, so
a telltale artefact gets found wherever it occurs. What pooling does. What
global pooling across time buys us — variable-length input and one decision per
utterance. Batch normalisation, at the level of what it's for.

**Read reference [5]** — Lavrentyeva et al., the light CNN system. Close to what
our baseline is.

---

## Phases 4–6 — one paper each

Split across the team, one short note per paper in `docs/literature/`: what
problem, what method, what dataset, what EER, what we take from it, what we
deliberately don't.

- [4] Todisco et al. — CQCC features (feeds P2.2)
- [6] RawNet2 — learning the filterbank from the raw waveform
- [7] AASIST — graph attention, current state of the art
- [8] wav2vec 2.0 — self-supervised speech representations
- [9] Tak et al. — wav2vec 2.0 as an anti-spoofing front-end
- [10] RawBoost and [11] SpecAugment — augmentation, feeding E2 directly
- [12], [13] — ASVspoof 2021, for the cross-dataset test in P6.2

Also worth understanding before phase 6: **score-level fusion**, and why
logistic regression fitted on dev scores is the standard approach.

---

## Phase 7 — reading results

**DET curves.** A plot of false rejection against false acceptance on warped
axes, showing performance across every threshold rather than one point. EER is
where the curve crosses the diagonal.

**Why per-attack breakdown matters.** A 5% average EER could be 1% on twelve
attacks and 50% on one. The one is the interesting finding.

**Variance.** Two runs of identical code with different seeds give different
numbers. An improvement smaller than that spread isn't an improvement.

---

## Deliberately skip

Transformers and self-supervised models — reference [9] is a stretch goal after
phase 8, not a plan. How TTS and voice conversion work internally: we're
building a detector, not a generator. GANs. Replay attacks.

Curiosity is fine, but these aren't on the critical path and the schedule is
tighter than it looks.
