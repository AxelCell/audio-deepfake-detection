# Phase 1 — how to run it

Exact commands, in order, with what each should print and how long it takes.

## Prerequisites

- Python 3.9 or newer (we are on 3.12). On Windows use `python`, not
  `python3` — the `python3` alias opens the Microsoft Store.
- About **20 GB free disk**: 7.12 GB for the zip, ~8–9 GB extracted, plus the
  virtual environment. You can delete the zip after extraction.
- Git, with your own `user.name` and `user.email` configured.

## 1. Environment (once per machine, ~30–60 min)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

Install **PyTorch first** from the selector at
https://pytorch.org/get-started/locally/ — pick your OS and CUDA version and
run the command it gives you. A plain `pip install torch` on Windows can
install a CPU-only build, which you will not notice until phase 3 when
training is roughly 40× slower than it should be.

Then:

```bash
pip install -r requirements.txt
```

Most of the time here is PyTorch, which is about 2.5 GB.

## 2. Tests (a few seconds)

```bash
pytest -q
```

Expected: `19 passed`. These deliberately do not need the dataset — they build
miniature protocol files and synthetic audio in a temp folder. If they fail on
a fresh clone, stop and fix that before downloading 7 GB.

## 3. Download the dataset (hours — plan to leave it overnight)

```bash
python scripts/download.py
```

Streams `LA.zip` (7,640,952,520 bytes) with a progress bar, checks the zip
opens, extracts into `data/`, then prints the folder structure.

- **Safe to interrupt.** Ctrl-C and re-run; it sends an HTTP `Range` header and
  resumes from where it stopped rather than starting again.
- **Expect it to be slow.** Measured on a home connection on 2026-09-21:
  roughly 6 MB/min, about 20 hours end to end. That is the server, not you —
  datashare.ed.ac.uk throttles each connection, and three simultaneous range
  requests were measured pulling ~530 KB/s between them while a single stream
  held ~100 KB/s. `download.py` uses one connection on purpose, because
  BUILD.md step 5 specifies a single resumable stream and a single stream is
  far harder to corrupt. Start it and go to bed.
- If you already have the zip: `python scripts/download.py --skip-download`
- Afterwards, delete `data/LA.zip` to free 7 GB.

Do **not** download `PA.zip` (16.4 GB). Replay attacks are out of scope.

## 4. Verify the dataset (~5–15 minutes)

```bash
python scripts/verify.py
```

Checks folders exist, loads all three label files, compares counts against the
published ASVspoof figures, checks the attack distribution, reports the class
balance, confirms all 121,461 audio files are on disk, and opens 100 random
files per partition to confirm 16 kHz mono.

Writes the same report to `results/verification.txt`.

**Must exit 0.** Nothing later in the project should be trusted until it does.
Count mismatches produce `MISMATCH` lines and a non-zero exit but do not
crash — that is deliberate, so you see every problem at once.

## 5. Make the first figure (~10 seconds)

```bash
python scripts/inspect.py
```

Writes `figures/first_look.png` — a 2×2 grid, real on the left and fake on the
right, waveform above and log-mel spectrogram below, both spectrograms on the
same colour scale so they are comparable. Also writes `results/inspect.json`
recording which utterances were used, their durations before and after length
fixing, the feature settings and the seed.

The pair is already frozen in `config.yaml` — `LA_T_8970804` (real) and
`LA_T_3653342` (spoof, attack A06) — so you should get exactly the PNG that is
in the repo, SHA-256 `9402b4e8…`. Check it:

```bash
sha256sum figures/first_look.png      # or: certutil -hashfile ... SHA256
```

If you set either id back to `null`, the script picks one at random from the
train set instead (preferring A04 or A06, whose artefacts are more visible),
prints the ids, and asks you to paste them back into `config.yaml`. That is
how the current pair was chosen.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | venv not activated, or running from the wrong folder | Activate the venv; run from the repo root |
| `pytest` collects 0 tests | Running from inside `tests/` | Run from the repo root |
| Download stalls or dies | Campus network | Re-run it; it resumes |
| `zipfile.BadZipFile` | Truncated download | Re-run `download.py`; it resumes and re-checks the byte count |
| `verify.py` reports MISMATCH counts | Incomplete extraction | Re-run `download.py`, then verify again |
| `verify.py` says files are missing | Extraction interrupted | Same |
| Training is absurdly slow (phase 3) | CPU-only PyTorch | Reinstall torch from the pytorch.org selector |
