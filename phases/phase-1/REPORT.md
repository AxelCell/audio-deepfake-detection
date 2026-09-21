# Phase 1 — Repository, environment, dataset verification, first spectrogram

**Weeks:** 1–3
**Status:** in progress — code complete, deliverable blocked on the dataset
**Tag:** phase-1-complete *(not yet tagged — see section 9)*
**Dates:** started 2026-09-21 … finished —

> **Read this first.** All ten BUILD.md steps have been written. The phase is
> **not** complete: the ASVspoof dataset has not been downloaded, so
> `figures/first_look.png` — the actual deliverable — does not exist yet.
> Section 5 says exactly what does and does not exist. Nothing in this report
> describes a run that did not happen.

## 1. What this phase was for

Before anything can be learned from speech recordings, the recordings have to
be on disk, correctly labelled, and turned into something a neural network can
read. Phase 1 builds that path end to end and proves it works: download the
ASVspoof 2019 LA dataset, confirm it is complete, parse its label files into a
table, load a waveform, and convert it into a log-mel spectrogram — the
time-frequency picture every later phase operates on.

It ends with one figure comparing a real recording to a fake one. That figure
is not the point in itself; being able to regenerate it identically, from one
command, on any team member's machine, is the point. No model is trained here.

## 2. What we built

- [x] Step 1 — skeleton: `.gitignore`, `requirements.txt`, folders, starter README
- [x] Step 2 — `config.yaml`, `src/utils.py`
- [x] Step 3 — `src/protocol.py`
- [x] Step 4 — `tests/test_basics.py`
- [x] Step 5 — `scripts/download.py` *(written; the 7.12 GB download not yet run)*
- [x] Step 6 — `src/audio.py`
- [x] Step 7 — `scripts/verify.py` *(written; never run on real data)*
- [x] Step 8 — log-mel in `src/features.py`
- [x] Step 9 — `src/plots.py`, `scripts/inspect.py` *(written; figure not yet produced)*
- [x] Step 10 — `README.md`, `PROGRESS.md`, this report, `HOW_TO_RUN.md`

### Step 1 — Repository skeleton (2026-09-21, Aishwary)

**Decisions, with the alternative that lost:**

- *PyTorch is listed in `requirements.txt`, with install-order instructions at
  the top.* The alternative was omitting torch and documenting it separately.
  Rejected because BUILD.md's done-when is "`pip install -r requirements.txt`
  works on a fresh machine", and a file missing torch silently produces an
  incomplete setup. Install torch from the pytorch.org selector first; pip then
  finds the constraint satisfied and leaves the build alone. A plain
  `pip install torch` on Windows can yield a CPU-only build, which would not be
  noticed until phase 3.
- *Version floors (`>=`), not exact pins (`==`).* Exact pins are more
  reproducible, which is the project's stated deliverable. Rejected for now
  because the lab GPU machine may need a different CUDA-matched build and we
  have not seen that machine yet. Floors are set at the first release carrying
  Python 3.12 wheels. Exact versions will be captured with `pip freeze` at
  phase close.
- *`results/` stays gitignored although `verify.py` and `inspect.py` write into
  it.* The alternative was committing it. Rejected because it is regenerated
  every run; what matters is copied into `artifacts/` at close. `results/` is
  scratch, `artifacts/` is the record.
- *`streamlit` left commented out.* Phase 8 only; installing it now would be
  working seven phases ahead.

**Additions beyond the BUILD.md list:** `venv/` and `.pytest_cache/` added to
`.gitignore`; the negation `!phases/**/artifacts/**` added because the global
`*.npy` and `*.pt` rules would otherwise swallow result files copied into a
phase artifacts folder.

**Verified, not assumed.** The ignore rules were tested in a throwaway git repo
with dummy files. 13 of 13 paths resolved correctly: a `.flac` under `data/`,
`LA.zip`, `results/*`, `__pycache__/`, `.venv/` and `checkpoints/best.pt` are
ignored; `figures/first_look.png`, `src/utils.py` and all three file types
under `phases/phase-1/artifacts/` (including `scores.npy`) remain tracked.

### Step 2 — `config.yaml` and `src/utils.py` (Aishwary)

Window and hop lengths are stored in **milliseconds, not samples**. A 25 ms
window is a claim about speech — the vocal tract cannot reshape itself much
faster than that, so the signal is roughly stable inside one window. "400
samples" only means the same thing if the audio is 16 kHz. Store 400, switch
sample rate later, and the window silently becomes a different duration while
nothing errors. Milliseconds also match how every paper states its settings,
which keeps our configuration comparable with published work.

`load_config` resolves paths against the repo root rather than the working
directory, so scripts run from anywhere. It returns paths as **strings, not
`Path` objects**, because `inspect.py` dumps parts of the config into
`results/inspect.json` and `Path` is not JSON-serialisable.

`set_seed` imports torch *inside* the function so that the tests run on a
machine without torch installed.

**A YAML trap worth knowing.** PyYAML follows YAML 1.1, which requires both a
decimal point and a signed exponent to parse a number in exponent form.
Verified: `1.0e-6` → float, `1e-6` → **string**, `1.0e6` → **string**. A
string `log_offset` would fail far away from the cause, inside `np.log`. A
comment in `config.yaml` now says not to "tidy" the spelling.

### Step 3 — `src/protocol.py` (Pranjal)

`LABEL_MAP = {"bonafide": 0, "spoof": 1}` is a module-level constant. Every EER
and t-DCF number in the project assumes this orientation, so flipping it later
would silently invert every result we have reported. An unexpected value in a
label file raises rather than producing a silent `NaN`.

Imports pandas and stdlib only — no torch, no librosa — which is what lets the
tests run before the dataset or a deep-learning stack exist. File existence is
deliberately **not** checked here; that is `verify.py`'s job, and keeping it
out is what makes this file testable without data.

### Step 4 — `tests/test_basics.py` (Pranjal)

Seven tests initially, built against a miniature six-line protocol file and
empty `.flac` files in a temp folder.

**Problem found and fixed:** pytest's default import mode puts only `tests/` on
`sys.path`, so `from src.protocol import ...` failed. Fixed with an empty
`conftest.py` at the repo root, which makes pytest prepend the root instead.
This was flagged as a risk during step 1 and resolved here rather than
pre-emptively, so the file has a visible reason to exist.

### Step 5 — `scripts/download.py` (Prateek)

The interesting part is the **resume**, not the download. Three server
responses need different handling: `206` means resume worked, append; `200`
means the server ignored the `Range` header and is re-sending from the start,
so the partial file must be discarded or the result is a corrupt concatenation;
`416` means we already have everything. Getting the `200` case wrong yields a
12 GB file that fails only when the zip is opened.

**Checked against the live server** (2026-09-21): a `HEAD` request is
misleading — DSpace returns an HTML page for `HEAD`. A `GET` redirects to the
real bitstream endpoint and returns `206 Partial Content`, `Accept-Ranges:
bytes`, ZIP magic `PK\x03\x04`, and `Content-Range` giving the exact total of
**7,640,952,520 bytes** (7.12 GiB). Resume is therefore genuinely supported.
That exact byte count replaced an original ±1 GB tolerance check, because an
exact figure makes truncation detectable and a rounded one does not.

The extract path and folder-tree report were tested end to end against a
miniature `LA.zip` built for the purpose, then the synthetic data was deleted.

### Step 6 — `src/audio.py` (Pranjal)

**Why every clip must be the same length.** The boring reason is that a batch
is a rectangular tensor. The reason that matters is that duration would
otherwise become a feature the model can cheat with: fake utterances come from
TTS and voice-conversion systems whose output lengths are distributed
differently from real recordings, so a model free to notice that will learn
"long = real, short = fake", score well on dev, and collapse on eval where the
attacks are different systems. Fixing every clip at 4 seconds removes the
shortcut; E4 then varies that number deliberately.

Clips that are too long are trimmed from the **front**, not randomly cropped.
Random cropping is the usual training-time augmentation and may return in phase
3, but a deterministic trim is what lets `inspect.py` produce a byte-identical
figure.

Repeat-padding rather than zero-padding is the ASVspoof convention: zero-padding
a 1-second clip to 4 seconds hands the model three seconds of digital silence,
which is itself a strong giveaway unrelated to how the audio was produced.

### Step 7 — `scripts/verify.py` (Prateek)

All eight checks from BUILD.md. Count mismatches **warn and continue** rather
than crashing, so an incomplete download shows every symptom at once.

**Added beyond spec:** a `--config` flag. BUILD.md does not ask for it, but
without it the script can only ever point at the real `data/`, and it could not
be tested until after a six-hour download. With it, all eight checks were
exercised against a synthetic dataset with deliberately wrong counts and one
deliberately deleted file. Every check behaved: folders passed, counts reported
`MISMATCH` without crashing, missing attacks were listed, the deleted file was
named, the audio sampler skipped it and reported `n=8` instead of 9, the report
was written, and the exit code was 1.

### Step 8 — log-mel in `src/features.py` (Tania)

`logmel()` derives window and hop in samples from the millisecond values in the
config rather than hardcoding 400 and 160.

Uses a plain `np.log(S + log_offset)`, **not** `librosa.power_to_db`.
`power_to_db` applies `10*log10` and then clips relative to the maximum value
*within each clip*, making the result depend on the loudest part of that
particular recording. A plain log with an explicit offset keeps every clip on
the same absolute scale and leaves the offset as something we configure and
record rather than something hidden in a library.

**Shape was predicted, then checked.** 4.0 s × 16 kHz = 64000 samples; hop 10 ms
= 160 samples; librosa centres frames and pads, so 64000/160 + 1 = 401 frames,
with 80 mel bands → **(80, 401)**. Actual output: `(80, 401)`. The silence case
returns exactly `log(1e-6) = -13.8155` everywhere rather than `-inf`, which is
what the offset is for.

Note: librosa **1.0.0** was installed here, not the 0.10.x the requirements
floor assumes. The `melspectrogram` API used is unchanged between them.

### Step 9 — `src/plots.py` and `scripts/inspect.py` (Tania)

Both spectrograms share one `vmin`/`vmax` computed across the pair. Without
that, each panel would auto-scale to its own range and the two pictures would
not be comparable — brightness differences would reflect the colour mapping
rather than the audio.

**Bug found: `scripts/inspect.py` shadows the standard library.** Python has a
stdlib module called `inspect`. Running `python scripts/inspect.py` puts
`scripts/` at the front of `sys.path`, so every internal `import inspect` in
numpy, librosa and matplotlib found *our* file and re-executed it mid-import.
The error it produced blamed a circular import in `src/audio.py`, which was
innocent. Fixed by removing the script's own directory from `sys.path` before
any third-party import. BUILD.md mandates the command `python
scripts/inspect.py`, so renaming was not an option.

**Bug found: overlapping labels.** The waveform's "time (s)" label collided
with the spectrogram title below it. `bbox_inches="tight"` trims the outer
margin but does not resolve collisions between panels; `constrained_layout=True`
does. `bbox_inches` was then removed, since it fights constrained layout and
can crop unpredictably — which would also threaten byte-for-byte repeatability.

**Determinism was measured, not assumed.** matplotlib writes a
`Software: Matplotlib vX.Y.Z` tag into PNG metadata by default, which alone
would make two runs differ. With it suppressed, two consecutive runs produced
identical SHA-256 hashes and identical file sizes.

## 3. Files

| File | New/changed | Purpose | Owner |
|---|---|---|---|
| `.gitignore` | new | Keeps audio, the dataset and working output out of git | Aishwary |
| `requirements.txt` | new | Dependency floors; torch install order | Aishwary |
| `config.yaml` | new | Every setting that affects a result | Aishwary |
| `src/utils.py` | new | Config loading, seeding, logging | Aishwary |
| `src/protocol.py` | new | Label files → pandas table; fixes label 0/1 | Pranjal |
| `src/audio.py` | new | Load, normalise, fix length | Pranjal |
| `src/features.py` | new | Waveform → log-mel spectrogram | Tania |
| `src/plots.py` | new | Waveform and spectrogram drawing | Tania |
| `scripts/download.py` | new | Resumable 7.12 GB fetch + extract | Prateek |
| `scripts/verify.py` | new | The dataset gate; 8 checks | Prateek |
| `scripts/inspect.py` | new | The phase deliverable figure | Tania |
| `tests/test_basics.py` | new | 19 tests, no dataset needed | Pranjal |
| `conftest.py` | new | Puts the repo root on `sys.path` for pytest | Pranjal |
| `README.md` | new | Project description, quickstart, empty results table | Aishwary |
| `PROGRESS.md` | new | Phase status table | Aishwary |
| `phases/phase-1/REPORT.md` | new | This document | Aishwary |
| `phases/phase-1/HOW_TO_RUN.md` | new | Exact commands and failure modes | Aishwary |

## 4. How to run it

See `HOW_TO_RUN.md` in this folder. The single most important command:

```bash
python scripts/verify.py       # must exit 0 before anything else is trusted
```

## 5. What it produced

**Real, and checked:**

- `pytest -q` → **19 passed**, in ~2 seconds, with no dataset present
- `src/features.py` returns **(80, 401)** for a 4-second clip, matching the
  shape predicted in advance
- `scripts/inspect.py` produces a **byte-identical PNG on re-run** — verified
  by SHA-256 over two consecutive runs against synthetic audio
- `scripts/verify.py` exercises all eight checks and exits 1 on a deliberately
  broken dataset
- `.gitignore` verified on 13 representative paths
- The download URL resolves, serves a real zip, and supports HTTP range resume;
  exact size 7,640,952,520 bytes

**Does not exist yet:**

- `figures/first_look.png` — **the phase deliverable.** Blocked on the dataset.
- `results/verification.txt` from a real run
- `results/inspect.json` from a real run
- `phases/phase-1/artifacts/` is empty
- Any confirmation that the real counts match the published table
- Any git history at all

## 6. What we learned

**Sampling.** A recording is air pressure measured 16,000 times a second.
16 kHz is enough for speech because almost all speech energy sits below 8 kHz,
and the Nyquist limit says a sample rate of 16 kHz captures frequencies up to
8 kHz.

**Frames, windows, hop.** Speech changes too fast to analyse in one block but
is roughly stable within 25 ms, so we cut it into 25 ms windows stepping
forward 10 ms. Since 10 < 25, consecutive windows overlap by 60% — without
that overlap, an event landing on a window boundary would be split and smeared
across two frames.

**Spectrograms.** Run an FFT on every frame and stack the results: time across,
frequency up, brightness showing energy. That picture is what the CNN sees.

**The mel scale, and the argument against it.** Human hearing resolves low
frequencies much more finely than high ones, and the mel scale warps the
frequency axis to match, compressing 257 FFT bins to 80 bands. But we are not
building for human ears: vocoder artefacts tend to live in exactly the
high-frequency detail mel discards. That is precisely why experiment E1 exists
— we use mel as the baseline because it is standard and comparable, then test
it against LFCC and CQCC rather than assuming it is right.

**Why accuracy is useless here.** The data is roughly 1:9 real to fake, so a
model that answers "fake" every time scores about 90% while being worthless.
Hence EER and min t-DCF.

**Why the eval attacks are different from the training attacks.** A01–A06 in
train and dev, A07–A19 in eval only. The test set uses generators the model has
never seen, so the headline number measures generalisation, not memorisation.

## 7. Problems and how we solved them

| Problem | Cause | Resolution |
|---|---|---|
| `from src.protocol import ...` failed under pytest | Default import mode puts only `tests/` on `sys.path` | Empty `conftest.py` at repo root |
| `inspect.py` died with a circular-import error blaming `src/audio.py` | The filename shadows the stdlib `inspect` module, which numpy/librosa/matplotlib import internally | Remove the script's own directory from `sys.path` before third-party imports |
| Waveform axis label overlapped the spectrogram title | `bbox_inches="tight"` trims the outer margin only | `constrained_layout=True`, and drop `bbox_inches` |
| PNG would not have been byte-identical across runs | matplotlib stamps a `Software:` version tag into PNG metadata | `metadata={"Software": None}`, then verified by hashing two runs |
| `HEAD` on the dataset URL returned `text/html` with no length, suggesting a broken link | DSpace serves an HTML page for `HEAD` but the real file for `GET` | Confirmed with a ranged `GET`: correct zip, resume supported |
| Creating `scripts/inspect.py` at step 9 silently broke `download.py` and `verify.py`, which had passed at steps 5 and 7 | The shadowing hazard is the *folder*, not the file: every script in `scripts/` runs with that directory first on `sys.path` | The same `sys.path` fix added to all three scripts, and a final pass re-checks every script and test together |
| A requirements-file checker flagged three lines as malformed | The checker was stricter than pip, which strips inline comments | Not a real defect; comments moved to their own lines anyway to remove the ambiguity |

**Lesson worth recording.** Fixing `inspect.py` alone was not enough, because
the hazard belongs to the folder rather than the file. The two scripts that
broke had both passed their own done-when checks earlier in the phase, and
nothing re-ran them until a final sweep. Checking each step only at the moment
it is written would have shipped two broken scripts.

**Negative result worth recording:** the `HEAD` check was actively misleading.
Had we trusted it, we would have concluded the download link was broken and
gone looking for a mirror. Verifying with the method the script actually uses —
a ranged `GET` — is the lesson.

## 8. Who did what

**No commits exist.** This is not yet a git repository: no `git init`, no
remote, no per-member identities, no branch protection. PROJECT.md §5 treats
`git log` as the record of who did what, and no document substitutes for it.

The Owner column in section 3 records which member's **area** each file belongs
to, so the right person commits it once the repository exists. That is an
assignment, not a claim that they wrote it.

| Member | Contribution | Commits |
|---|---|---|
| Aishwary Bisht | Repo skeleton, config, utils, README, reports | — *(no repo yet)* |
| Pranjal | protocol.py, audio.py, tests, conftest | — |
| Prateek Kukreti | download.py, verify.py | — |
| Tania Sharma | features.py, plots.py, inspect.py | — |

## 9. Checklist

Ticked honestly. Four of ten items are **not** done.

- [x] Fresh clone + `pip install -r requirements.txt` gives a working setup —
      *partially:* all 13 requirement specifiers parse and everything except
      torch/torchaudio installs cleanly into a fresh venv. The torch install
      has not been performed on any team machine.
- [x] `pytest -q` passes without the dataset present — 19 passed
- [ ] `python scripts/verify.py` shows the expected counts and exits 0 —
      **blocked: no dataset.** Verified only against synthetic data.
- [ ] `python scripts/inspect.py` writes `figures/first_look.png` —
      **blocked: no dataset.**
- [x] Running it twice gives an identical file — verified by SHA-256 on
      synthetic audio; must be re-confirmed on the real figure
- [ ] All four team members have done the above on their own machines —
      **not started**
- [ ] 13 literature notes exist in `docs/literature/` — **not started**
- [ ] `git log --stat` shows no audio file was ever committed — **no git
      history exists yet.** The ignore rules that would guarantee it are
      verified.
- [x] `phases/phase-1/REPORT.md` and `HOW_TO_RUN.md` complete — complete as
      far as work that has actually happened
- [ ] Tagged `phase-1-complete` — **not tagged.** Tagging now would mark an
      incomplete phase as finished.

## 10. What's next

**To close phase 1**, in order:

1. `git init`, remote, branch protection, per-member identities (Aishwary)
2. Download and extract the dataset (Prateek, 2–6 hours unattended)
3. `python scripts/verify.py` → must exit 0; copy `results/verification.txt`
   into `artifacts/`
4. `python scripts/inspect.py` → paste the printed ids into `config.yaml`, re-run,
   confirm the PNG is identical, copy it into `artifacts/`
5. All four members run steps 1–5 of `HOW_TO_RUN.md` on their own machines
6. 13 literature notes in `docs/literature/`
7. Update this checklist, then tag `phase-1-complete`

**What phase 2 needs from this one:** a verified dataset, `load_protocol`
returning a stable table with `label` fixed at 0/1, `fix_length` guaranteeing
uniform input, and `logmel` as the reference front-end that LFCC and CQCC will
be written to match in shape and interface.
