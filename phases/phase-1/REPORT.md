# Phase 1 — Repository, environment, dataset verification, first spectrogram

**Weeks:** 1–3
**Status:** complete
**Tag:** `phase-1-complete`
**Dates:** started 2026-09-21 … finished 2026-09-21

All ten steps of the phase 1 plan are built, run and verified against the
real dataset.
`verify.py` exits 0, and `figures/first_look.png` exists. Section 5 lists what
was produced. Nothing here describes a run that did not happen.

## 1. What this phase was for

Before anything can be learned from speech recordings, the recordings have to
be on disk, correctly labelled, and turned into something a network can read.
Phase 1 builds that path end to end: download ASVspoof 2019 LA, check it is
complete, parse the label files into a table, load a waveform, and convert it
to a log-mel spectrogram.

It ends with one figure comparing a real clip to a fake one. The figure itself
is not the point — being able to regenerate it identically from one command on
any of our machines is. No model is trained here.

## 2. What we built

- [x] Step 1 — skeleton: `.gitignore`, `requirements.txt`, folders, README
- [x] Step 2 — `config.yaml`, `src/utils.py`
- [x] Step 3 — `src/protocol.py`
- [x] Step 4 — `tests/test_basics.py`
- [x] Step 5 — `scripts/download.py` *(7.12 GB downloaded and extracted)*
- [x] Step 6 — `src/audio.py`
- [x] Step 7 — `scripts/verify.py` *(exits 0 on the real dataset)*
- [x] Step 8 — log-mel in `src/features.py`
- [x] Step 9 — `src/plots.py`, `scripts/inspect.py` *(figure produced and frozen)*
- [x] Step 10 — `README.md`, `PROGRESS.md`, this report, `HOW_TO_RUN.md`

### Step 1 — Skeleton (Aishwary)

Choices we made, and what we rejected:

- **PyTorch stays in `requirements.txt`** with install-order notes at the top,
  rather than being documented separately. The plan's done-when is that
  `pip install -r requirements.txt` works on a fresh machine, and a file
  missing torch gives an incomplete setup without saying so.
- **Version floors (`>=`), not exact pins.** Pins are more reproducible, but
  the lab GPU machine may need a different CUDA build and we have not seen it
  yet. We will capture exact versions with `pip freeze` at phase close.
- **`results/` stays gitignored** even though the scripts write there. It is
  regenerated every run; what matters gets copied into `artifacts/`.

We also added `!phases/**/artifacts/**` to `.gitignore`, because the global
`*.npy` and `*.pt` rules would otherwise swallow result files copied into a
phase folder.

`.gitattributes` was added later, when git was initialised: `* text=auto
eol=lf` so Windows and WSL machines stop rewriting each other's line endings,
and `*.png binary` so git never touches the figure. Without the second rule,
line-ending normalisation would corrupt `first_look.png` and quietly break the
byte-identical requirement.

### Step 2 — `config.yaml` and `src/utils.py` (Aishwary)

Window and hop are stored in **milliseconds, not samples**. A 25 ms window is
a claim about speech: the vocal tract cannot reshape itself much faster, so
the signal is roughly stable inside one window. "400 samples" only means the
same thing at 16 kHz — store 400, change the sample rate later, and the window
silently becomes a different duration with no error. Papers state settings in
milliseconds too, which keeps our config comparable.

`load_config` resolves paths against the repo root, not the working directory,
so scripts run from anywhere. It returns strings rather than `Path` objects
because `inspect.py` writes part of the config into a JSON file.

`set_seed` imports torch inside the function so the tests run without it.

A YAML trap worth knowing: PyYAML follows YAML 1.1, which needs both a decimal
point and a signed exponent. `1.0e-6` parses as a float; `1e-6` and `1.0e6`
parse as **strings**. A string `log_offset` fails far from the cause, inside
`np.log`. There is a comment in `config.yaml` warning not to tidy it.

### Step 3 — `src/protocol.py` (Pranjal)

`LABEL_MAP = {"bonafide": 0, "spoof": 1}` is fixed now. Every EER we report
assumes this orientation, so changing it later would invert all our results
without any error. An unexpected value in a label file raises instead of
producing a silent `NaN`.

Imports pandas and the standard library only, which is what lets the tests run
before the dataset exists. File existence is not checked here — that belongs
to `verify.py`, and leaving it out is what keeps this file testable.

### Step 4 — `tests/test_basics.py` (Pranjal)

19 tests against a six-line protocol file and synthetic audio in a temp
folder. All of them run without the dataset.

Problem found: pytest's default import mode puts only `tests/` on `sys.path`,
so `from src.protocol import ...` failed. Fixed with an empty `conftest.py` at
the repo root.

### Step 5 — `scripts/download.py` (Prateek)

The interesting part is the resume, not the download. Three server responses
need different handling: `206` means resume worked, so append; `200` means the
server ignored our `Range` header and is re-sending from the start, so the
partial file must be thrown away or we end up with a corrupt concatenation;
`416` means we already have the whole thing. Getting the `200` case wrong
gives a 12 GB file that only fails when the zip is opened.

Checked against the live server: a ranged `GET` returns `206`,
`Accept-Ranges: bytes`, ZIP magic `PK\x03\x04`, and a `Content-Range` giving
the exact total of **7,640,952,520 bytes**. That exact figure replaced an
earlier ±1 GB tolerance, because an exact number makes truncation detectable.

Throughput was poor and erratic — 40 to 130 kB/s, averaging around 6 MB/min,
and the whole thing took about 4.5 hours. Three parallel range requests pulled
~530 KB/s between them against ~100 KB/s for one stream, so the server
throttles per connection. We kept the single stream anyway: the plan calls
for one resumable stream, and one stream is much harder to corrupt than
several writing at offsets.

**The first run died.** At 1.21 GB the connection dropped and the script
raised `ChunkedEncodingError` and stopped. The `Range` resume worked, but only
if a human noticed and re-ran it — which over 7.6 GB on this connection would
have meant restarting it by hand five or six times. `download()` now retries
by itself: each attempt sends a fresh `Range` header for the bytes not yet on
disk, so a drop costs seconds rather than the run. It gives up after three
attempts in a row that move no bytes, so a dead network fails loudly instead
of spinning. The retry and the give-up path were both tested with a simulated
flaky connection before being trusted with the remaining 6.4 GB. The second
run completed to the exact expected byte count.

### Step 6 — `src/audio.py` (Pranjal)

Why every clip has to be the same length: the boring reason is that a batch is
a rectangular tensor. The reason that matters is that duration would otherwise
become something the model can cheat with. TTS and voice-conversion systems
produce output whose lengths are distributed differently from real recordings,
so a model free to notice that will learn "long = real, short = fake", do well
on dev, and collapse on eval where the attacks are different systems. Fixing
every clip at 4 seconds removes the shortcut; E4 varies it deliberately later.

Clips that are too long are trimmed from the front rather than randomly
cropped. Random cropping is the usual training-time augmentation and may come
back in phase 3, but a deterministic trim is what makes the figure repeatable.

Repeat-padding rather than zero-padding follows the ASVspoof convention.
Zero-padding a 1-second clip to 4 seconds gives the model three seconds of
silence, which is a giveaway unrelated to how the audio was made.

### Step 7 — `scripts/verify.py` (Prateek)

All eight checks from the plan. Count mismatches warn and continue instead of
crashing, so an incomplete download shows every symptom at once.

Added beyond spec: a `--config` flag. Without it the script can only point at
the real `data/`, and could not be tested until after the download finished.
With it, all eight checks were run against a synthetic dataset with wrong
counts and a deleted file. Every check behaved: folders passed, counts
reported `MISMATCH` without crashing, missing attacks were listed, and the
report was written.

### Step 8 — log-mel in `src/features.py` (Tania)

Window and hop in samples are derived from the millisecond values in the
config rather than hardcoded as 400 and 160.

Uses a plain `np.log(S + log_offset)`, not `librosa.power_to_db`.
`power_to_db` applies `10*log10` and then clips relative to the maximum value
within each clip, so the result depends on the loudest part of that particular
recording. A plain log with an explicit offset keeps every clip on the same
scale and leaves the offset as something we set and record.

We predicted the shape before running it: 4.0 s × 16 kHz = 64000 samples, hop
10 ms = 160 samples, librosa centres frames and pads the edges, so
64000/160 + 1 = 401 frames with 80 mel bands, giving **(80, 401)**. The actual
output is `(80, 401)`. Silence returns `log(1e-6) = -13.8155` everywhere
instead of `-inf`, which is what the offset is for.

Note: librosa 1.0.0 was installed, not the 0.10.x our requirements floor
assumes. The `melspectrogram` API we use is unchanged between them.

### Step 9 — `src/plots.py` and `scripts/inspect.py` (Tania)

Both spectrograms share one `vmin`/`vmax` computed across the pair. Without
that each panel auto-scales to its own range and brightness differences
between them mean nothing.

Bug found: `scripts/inspect.py` shadows the standard library. Python has a
module called `inspect`, and running `python scripts/inspect.py` puts
`scripts/` first on `sys.path`, so every internal `import inspect` in numpy,
librosa and matplotlib found our file and re-executed it mid-import. The error
blamed a circular import in `src/audio.py`, which was innocent. Fixed by
replacing `sys.path[0]` — the script's own directory — with the repo root.
The plan mandates the command `python scripts/inspect.py`, so renaming the
file was not an option.

Bug found: the waveform's "time (s)" label collided with the spectrogram title
below it. `bbox_inches="tight"` trims the outer margin but does not fix
collisions between panels; `constrained_layout=True` does. We then dropped
`bbox_inches`, since it fights constrained layout and can crop unpredictably.

Bug found: `choose_utterances` only validated the ids from `config.yaml` when
both were set. Since the documented workflow is "run once, paste the ids in",
a typo in one of them fell through to `.iloc[0]` and raised a bare
`IndexError`. It now names the field and says how to recover.

Determinism was measured, not assumed. matplotlib writes a `Software:
Matplotlib vX.Y.Z` tag into PNG metadata by default, which on its own would
make two runs differ. With it suppressed, two consecutive runs gave identical
SHA-256 hashes.

## 3. Files

| File | New/changed | Purpose | Owner |
|---|---|---|---|
| `.gitignore` | new | Keeps audio and working output out of git | Aishwary |
| `.gitattributes` | new | LF in the repo; PNG never rewritten | Aishwary |
| `requirements.txt` | new | Dependency floors, torch install order | Aishwary |
| `config.yaml` | new | Every setting that affects a result | Aishwary |
| `src/utils.py` | new | Config loading, seeding, logging | Aishwary |
| `src/protocol.py` | new | Label files to pandas table; fixes label 0/1 | Pranjal |
| `src/audio.py` | new | Load, normalise, fix length | Pranjal |
| `src/features.py` | new | Waveform to log-mel spectrogram | Tania |
| `src/plots.py` | new | Waveform and spectrogram drawing | Tania |
| `scripts/download.py` | new | Resumable 7.12 GB fetch and extract | Prateek |
| `scripts/verify.py` | new | The dataset gate, 8 checks | Prateek |
| `scripts/inspect.py` | new | The phase deliverable figure | Tania |
| `tests/test_basics.py` | new | 19 tests, no dataset needed | Pranjal |
| `conftest.py` | new | Puts the repo root on `sys.path` for pytest | Pranjal |
| `README.md` | new | Description, quickstart, empty results table | Aishwary |
| `PROGRESS.md` | new | Phase status table | Aishwary |
| `phases/phase-1/REPORT.md` | new | This document | Aishwary |
| `phases/phase-1/HOW_TO_RUN.md` | new | Exact commands and failure modes | Aishwary |

## 4. How to run it

See `HOW_TO_RUN.md` in this folder. The command that matters:

```bash
python scripts/verify.py       # must exit 0 before anything else is trusted
```

## 5. What it produced

**`results/verification.txt`** — `verify.py` exits 0. Every partition matches
the published ASVspoof 2019 LA figures exactly:

| Partition | Real | Fake | Total | Expected |
|---|---|---|---|---|
| train | 2,580 | 22,800 | 25,380 | matches |
| dev | 2,548 | 22,296 | 24,844 | matches |
| eval | 7,355 | 63,882 | 71,237 | matches |

Attacks are as documented: A01–A06 in train and dev, all thirteen of A07–A19
in eval only. All 121,461 files are present on disk. Of the 300 files sampled,
every one is 16 kHz mono; durations run 0.88 s to 7.91 s with a median near
3.0 s, which is why the 4-second window mostly pads rather than trims.

Measured class balance is **1 real : 8.8 fake**, 89.7% spoof. That is the
number behind rule 4 — a model answering "fake" every time scores about 90%.

**`figures/first_look.png`** — `LA_T_8970804` (bona fide, 3.02 s) against
`LA_T_3653342` (spoof, attack A06, 5.66 s). Both ids are frozen in
`config.yaml`. Three consecutive runs gave SHA-256 `9402b4e8…` every time.

Two differences are visible in the figure without any model. The real clip has
a noise floor during silence — a haze across all frequencies before speech
starts — where the A06 fake is nearly black; synthesis produces unnaturally
clean silence. And the fake's harmonics are more regular, with visible banding
around 2–4 kHz. Worth remembering in phase 3: the silence difference is a real
cue, but it is also a shortcut a model can learn instead of the speech.

**`results/inspect.json`** — records both ids, labels, durations before and
after length fixing, the spectrogram shape, the feature settings and the seed.
Confirms `log_offset` parsed as the float `1e-06` rather than a string.

**Other:**

- `pytest -q` → 19 passed, ~3 seconds, no dataset needed
- `src/features.py` returns (80, 401), the shape predicted in advance
- Git repository, 7 commits, no audio ever committed
- All three artifacts copied into `phases/phase-1/artifacts/`

## 6. What we learned

**Sampling.** A recording is air pressure measured 16,000 times a second.
16 kHz is enough for speech because nearly all speech energy is below 8 kHz,
and Nyquist says a 16 kHz rate captures frequencies up to 8 kHz.

**Frames, windows, hop.** Speech changes too fast to analyse in one block but
is roughly stable within 25 ms, so we cut it into 25 ms windows stepping
forward 10 ms. Since 10 < 25 the windows overlap by 60%; without the overlap
an event landing on a boundary would be split across two frames.

**Spectrograms.** An FFT on every frame, stacked: time across, frequency up,
brightness showing energy. That picture is what the CNN sees.

**The mel scale, and the argument against it.** Human hearing resolves low
frequencies much more finely than high ones, and the mel scale warps the axis
to match, compressing 257 FFT bins into 80 bands. But we are not building for
human ears, and vocoder artefacts tend to live in exactly the high-frequency
detail mel throws away. That is why E1 exists: mel is the baseline because it
is standard and comparable, then we test it against LFCC and CQCC rather than
assume it is right.

**Why accuracy is useless here.** Roughly 1:9 real to fake, so answering
"fake" every time scores about 90% while being worthless. Hence EER and t-DCF.

**Why the eval attacks differ from the training attacks.** A01–A06 in train
and dev, A07–A19 in eval only. The test set uses generators the model has
never seen, so the headline number measures generalisation.

## 7. Problems and how we solved them

| Problem | Cause | Resolution |
|---|---|---|
| `from src.protocol import ...` failed under pytest | Default import mode puts only `tests/` on `sys.path` | Empty `conftest.py` at the repo root |
| `inspect.py` died with a circular-import error blaming `src/audio.py` | The filename shadows the stdlib `inspect`, which numpy and librosa import | Replace `sys.path[0]` with the repo root |
| Waveform label overlapped the spectrogram title | `bbox_inches="tight"` trims the outer margin only | `constrained_layout=True`, drop `bbox_inches` |
| PNG would not have been byte-identical across runs | matplotlib stamps a `Software:` tag into PNG metadata | `metadata={"Software": None}`, verified by hashing two runs |
| A typo'd utterance id in `config.yaml` gave a bare `IndexError` | Validation only ran when both ids were set | Validate each id independently, name the field in the error |
| `HEAD` on the dataset URL returned `text/html`, suggesting a dead link | DSpace serves an HTML page for `HEAD` but the real file for `GET` | Confirmed with a ranged `GET` |
| Creating `scripts/inspect.py` silently broke `download.py` and `verify.py`, which had passed earlier | The shadowing hazard is the *folder*, not the file | Same fix in all three scripts, plus a final pass that re-runs everything |
| The download died at 1.21 GB and stayed dead | `ChunkedEncodingError` on a dropped connection was unhandled; resume needed a human to re-run | `download()` retries itself, resuming by `Range`; gives up after three attempts that move no bytes |
| A background task reported "exit code 0" on that failed download | The command was piped through `tail`, and a pipeline reports the *last* command's status | Check the output and the file size, not the exit code |

Two lessons worth recording. First, fixing `inspect.py` alone was not enough,
because the hazard belongs to the folder rather than the file — two scripts
that had passed their own done-when checks were broken and nothing re-ran them
until a final sweep. Second, the `HEAD` check was actively misleading; had we
trusted it we would have gone looking for a mirror that does not exist.
Verify with the method the script actually uses.

## 8. Who did what

The repository was initialised after the code was written, so the first commit
is a single import under one name. The commit message says so. Per-member
attribution starts from the next commits, and the Owner column in section 3
records whose area each file belongs to.

| Member | Contribution | Commits |
|---|---|---|
| Aishwary Bisht | Repo skeleton, config, utils, README, reports | initial import, then docs and the download retry |
| Pranjal | `protocol.py`, `audio.py`, tests, `conftest.py` | — |
| Prateek Kukreti | `download.py`, `verify.py` | — |
| Tania Sharma | `features.py`, `plots.py`, `inspect.py` | — |

## 9. Checklist

- [x] Fresh clone + `pip install -r requirements.txt` gives a working setup —
      everything except torch/torchaudio, which phase 3 needs and nothing
      before it imports
- [x] `pytest -q` passes without the dataset — 19 passed
- [x] `python scripts/verify.py` shows the expected counts and exits 0
- [x] `python scripts/inspect.py` writes `figures/first_look.png`
- [x] Running it twice gives an identical file — SHA-256 `9402b4e8…` across
      three runs, with the ids frozen in `config.yaml`
- [ ] All four members have done the above on their own machines — **one has**
- [ ] 13 literature notes in `docs/literature/` — template and index in place,
      notes not written, references [2] and [3] not chosen
- [x] `git log --stat` shows no audio file was ever committed
- [x] `phases/phase-1/REPORT.md` and `HOW_TO_RUN.md` complete
- [x] Tagged `phase-1-complete`

Two items remain open and are carried into phase 2 rather than blocking the
tag: both are team tasks rather than code, and neither changes what the
pipeline does. They are listed in `PROGRESS.md`.

## 10. What's next

Carried over from this phase:

1. All four members run `HOW_TO_RUN.md` on their own machines. Each will need
   the 7.12 GB download, so start it early.
2. The 13 literature notes, and choose references [2] and [3].
3. Push to GitHub, then branch protection and per-member identities, so
   `git log` becomes the record of who did what.

Phase 2 proper: LFCC and CQCC front-ends, the dataset class and DataLoader,
and feature caching across all 121,461 files.

What phase 2 needs from this phase: a verified dataset, `load_protocol`
returning a stable table with `label` fixed at 0/1, `fix_length` guaranteeing
uniform input, and `logmel` as the reference front-end that LFCC and CQCC will
be written to match in shape and interface.
