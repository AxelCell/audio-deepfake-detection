# Who owns what, and what you need to be able to explain

Our work-split plan says who builds which file. This file is the other
half: what you should understand about your own part well enough to explain it
without notes, and the questions you are most likely to be asked about it.

Read your own section properly. Skim the other three so you know who to ask.

A rule that applies to everyone: if you cannot explain why a choice was made,
you do not own that file yet. "It was already there" is not an answer in a
viva.

---

## Aishwary Bisht — repo, config, utilities, reports

**Phase 1 files:** `.gitignore`, `.gitattributes`, `requirements.txt`,
`config.yaml`, `src/utils.py`, `README.md`, `PROGRESS.md`, the phase reports.

**Later:** `src/train.py` and the training loop (phase 3), the experiment
runner (phase 4), final report assembly (phase 8).

### What your part does

`config.yaml` is the single place every setting lives — sample rate, window
size, mel bands, seed, learning rate. `src/utils.py` loads it, makes the paths
absolute so scripts run from any folder, seeds the random number generators,
and sets up logging.

### What you must be able to explain

- **Why settings live in the config and not in the code.** If a number that
  changes a result is typed into a `.py` file, you can no longer say which
  number produced which result. Every experiment in phases 4–6 works by
  changing one config value and nothing else.
- **Why window and hop are stored in milliseconds, not samples.** 25 ms is a
  claim about speech — the vocal tract cannot reshape itself much faster, so
  the signal is roughly steady inside one window. "400 samples" only means the
  same thing at 16 kHz.
- **The YAML number trap.** PyYAML follows YAML 1.1, which needs both a
  decimal point and a signed exponent. `1.0e-6` is a float; `1e-6` is a
  *string*. A string `log_offset` breaks far from the cause, inside `np.log`.
- **Why `set_seed` imports torch inside the function.** So the tests run on a
  machine where torch is not installed yet.
- **Why `.gitattributes` marks PNG as binary.** Git normalising line endings
  inside a PNG would corrupt it, and silently break the requirement that the
  figure is byte-identical on re-run.

### Likely questions

- "What happens if two people run this on different machines?"
- "Where would I change the input length to 2 seconds?" (`config.yaml`,
  `audio.duration_seconds` — and that is experiment E4)
- "Why is `results/` gitignored but `figures/` is not?"

---

## Pranjal — data pipeline, protocol, model, tests

**Phase 1 files:** `src/protocol.py`, `src/audio.py`, `tests/test_basics.py`,
`conftest.py`.

**Later:** `src/dataset.py` and the DataLoader (phase 2), `src/model.py` — the
baseline CNN (phase 3), error-correlation analysis (phase 6).

### What your part does

`protocol.py` reads the ASVspoof label files into a pandas table. `audio.py`
loads a waveform and gets every clip to the same length. The tests prove both
work without the dataset present.

### What you must be able to explain

- **The label convention: 0 = real, 1 = fake.** It is fixed in `LABEL_MAP` and
  must never change. Every EER in the project assumes this orientation, so
  flipping it later would invert every result with no error message.
- **Why every clip has to be the same length.** The boring reason is that a
  batch is a rectangular tensor. The real reason is that TTS and
  voice-conversion systems produce clips whose lengths are distributed
  differently from real recordings — so a model free to notice that learns
  "long = real, short = fake", scores well on dev, and collapses on eval.
- **Why repeat-padding, not zero-padding.** Padding a 1-second clip out to 4
  seconds with silence hands the model three seconds of digital silence, which
  is itself a giveaway unrelated to how the audio was made.
- **Why long clips are trimmed from the front, not randomly cropped.** Random
  cropping is the usual training-time augmentation and may come back in phase
  3, but a deterministic trim is what makes the figure reproducible.
- **Why `protocol.py` imports only pandas and the standard library.** So the
  tests run before anyone has downloaded 7 GB or installed torch.
- **Why file existence is not checked in `protocol.py`.** That is `verify.py`'s
  job. Keeping it out is exactly what makes this file testable without data.

### Likely questions

- "What does the third column of the label file mean?" (Nothing — always `-`,
  discarded.)
- "Why is `A01` in train but `A11` only in eval?" (The eval attacks are
  generators the model never sees, so the number measures generalisation.)
- "How do you know your padding works?" (Point at
  `test_fix_length_repeat_pads_100_to_250` and walk through it.)

---

## Prateek Kukreti — scripts, dataset acquisition, metrics

**Phase 1 files:** `scripts/download.py`, `scripts/verify.py`.

**Later:** feature extraction and caching (phase 2), `src/metrics.py` — EER and
t-DCF (phase 3), the cross-dataset test (phase 6), DET curves (phase 7).

### What your part does

`download.py` fetches 7.12 GB and survives being interrupted. `verify.py` is
the gate that decides whether the dataset can be trusted at all.

### What you must be able to explain

- **How HTTP resume works, and the trap in it.** Three responses need
  different handling. `206` means resume worked, so append. `200` means the
  server ignored the `Range` header and is re-sending from the start — so the
  partial file must be thrown away, or you get a corrupt concatenation. `416`
  means you already have the whole file. Getting the `200` case wrong gives a
  12 GB file that only fails when the zip is opened.
- **Why the exact byte count matters.** 7,640,952,520 exactly, read from the
  server's `Content-Range`. An exact figure makes truncation detectable; a
  rounded one does not.
- **Why `verify.py` warns instead of crashing on a count mismatch.** So an
  incomplete download shows every symptom at once, rather than one per re-run.
- **Why a `HEAD` request on the dataset URL is misleading.** DSpace serves an
  HTML page for `HEAD` but the real file for `GET`. If you trust `HEAD` you
  conclude the link is dead and go looking for a mirror that does not exist.
- **Why the class balance check exists at all.** It prints roughly 1 real to 9
  fake, which is the whole reason the project reports EER and not accuracy.

### Coming in phase 3, and it is the most important thing you will write

`compute_eer` — every number in the project flows through it. If it is subtly
wrong, every result is wrong and nothing will tell you. It gets sanity-checked
as a test: random scores must give about 50% EER, perfectly separated scores
must give 0%.

### Likely questions

- "What is EER, in one sentence?" (Slide the threshold until the fraction of
  real clips wrongly rejected equals the fraction of fakes wrongly accepted.)
- "Why not just report accuracy?"
- "What would you do if the counts came back short?"

---

## Tania Sharma — features, visualisation, the app

**Phase 1 files:** `src/features.py`, `src/plots.py`, `scripts/inspect.py`.

**Later:** LFCC and CQCC (phase 2), `src/evaluate.py` and the per-attack
breakdown (phase 3), augmentation E2 (phase 4), the Streamlit app (phase 8).

### What your part does

`features.py` turns a waveform into a log-mel spectrogram — the picture the
CNN actually sees. `plots.py` and `inspect.py` produce the phase 1 deliverable.

### What you must be able to explain

- **What a mel spectrogram is.** Cut the audio into 25 ms windows stepping
  10 ms, run an FFT on each, stack them: time across, frequency up, brightness
  is energy. Then warp the frequency axis to the mel scale, compressing 257
  FFT bins into 80 bands.
- **Why the windows overlap.** Hop 10 ms is less than the 25 ms window, so
  consecutive windows overlap by 60%. Without that, an event landing on a
  window boundary gets split and smeared across two frames.
- **The argument *against* the mel scale** — and you need this one, because it
  is the reason experiment E1 exists. Mel matches human hearing, which
  resolves low frequencies far more finely than high ones. But we are not
  building for human ears, and vocoder artefacts tend to live in exactly the
  high-frequency detail that mel throws away. So mel is the baseline because
  it is standard and comparable, not because it is right.
- **Why a plain `np.log`, not `librosa.power_to_db`.** `power_to_db` clips
  relative to the loudest point *within each clip*, so the result depends on
  that one recording. A plain log with an explicit offset keeps every clip on
  the same scale, and the offset is something we set and record rather than
  something hidden in a library.
- **Why both spectrograms share one colour scale.** If each auto-scaled to its
  own range, the brightness difference between the two panels would reflect
  the colour mapping rather than the audio, and the figure would be lying.
- **Why the shape is (80, 401).** 4.0 s × 16 kHz = 64000 samples; hop 10 ms =
  160 samples; 64000/160 = 400, plus 1 because librosa centres frames and pads
  the edges. 80 is the mel band count. We predicted this before running it.

### Likely questions

- "Can you hear the difference between the real and fake clip?" (No. That is
  the entire motivation for the project.)
- "What is the offset in `log(S + 1e-6)` for?" (`log(0)` is `-inf`. Silence
  would produce `-inf` everywhere and poison the batch.)
- "Why does the figure have to be byte-identical on re-run?"

---

## How to use this before a review or viva

Each person: open your own files, read them top to bottom, and try saying the
bullets above out loud without looking. Where you stall, that is the bit to
study — the study plan says what to read for each topic.

Then swap: ask someone else's questions to them. The gaps show up much faster
when somebody else is asking.
