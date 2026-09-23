# BUILD.md — Phase 1 (weeks 1–3)

The step-by-step detail for phase 1. `PROJECT.md` section 3 has the process —
kickoff, handoffs, closing the phase. Read that first; this file is the *what*,
that one is the *how*.

Phases 2–8 are in `ROADMAP.md`.

**Goal:** by the end of week 3, anyone on the team can clone this repo, run one
command, and see a picture comparing a real recording to a fake one from the
ASVspoof dataset.

**No model gets trained here.** No neural network, no training loop, no EER.
Those are phase 3.

Build the steps below **in order**, one per working session. After each step,
stop and let us review.

---

## The dataset, in brief

We use **ASVspoof 2019, Logical Access (LA)**. It's the standard benchmark for
this problem, which is what makes our numbers comparable with published work.

- Download page: `https://datashare.ed.ac.uk/handle/10283/3336`
- File: `LA.zip`, **7.1 GB**
- Direct link (check it resolves): `https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip`
- **Do not download `PA.zip`** (16.4 GB) — replay attacks are out of scope.
- Licence: Open Data Commons Attribution.
- Also grab `README.txt` and `asvspoof2019_evaluation_plan.pdf` from that page
  into `docs/`.

After unzipping into `data/`:

```
data/LA/
├── ASVspoof2019_LA_train/flac/     *.flac
├── ASVspoof2019_LA_dev/flac/       *.flac
├── ASVspoof2019_LA_eval/flac/      *.flac
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
└── ASVspoof2019_LA_asv_scores/     (needed in phase 3 for t-DCF)
```

Audio is FLAC, 16 kHz, mono. Utterances are a few seconds each.

### The label files

Five columns separated by spaces:

```
LA_0079 LA_T_1138215 - -   bonafide
LA_0039 LA_E_2834763 - A11 spoof
```

| # | Meaning |
|---|---|
| 1 | speaker id |
| 2 | utterance id — the filename is `<id>.flac` |
| 3 | unused, always `-`, throw it away |
| 4 | which attack made it: `-` if real, `A01`–`A19` if fake |
| 5 | the label: `bonafide` or `spoof` |

Attacks `A01`–`A06` appear in train and dev. Attacks `A07`–`A19` appear **only**
in eval. That's deliberate — the test set uses fake-audio generators the model
has never seen, so our headline number measures generalisation, not
memorisation.

### Expected counts

Use these as a sanity check. If a count differs, **print a clear warning and
keep going** — don't crash. A wrong count usually means an incomplete download.

| Partition | Real | Fake | Total | Attacks |
|---|---|---|---|---|
| train | 2,580 | 22,800 | 25,380 | A01–A06 |
| dev | 2,548 | 22,296 | 24,844 | A01–A06 |
| eval | 7,355 | 63,882 | 71,237 | A07–A19 |

---

## Step 1 — Skeleton
**Owner: Aishwary**

Create `.gitignore`, `requirements.txt`, the folders from the layout in
`CLAUDE.md`, and a starter `README.md`.

`.gitignore` must include:

```
data/
*.flac
*.wav
*.zip
results/
checkpoints/
*.pt
*.npy
.venv/
__pycache__/
.ipynb_checkpoints/
```

`figures/*.png` and `phases/**/artifacts/*` are **not** ignored — those get
committed.

Then explain how to create the venv and install everything. Note that PyTorch
on Windows needs the command from pytorch.org's install selector, not a plain
`pip install torch`, or you may get a CPU-only build.

**Done when:** `pip install -r requirements.txt` works on a fresh machine.

---

## Step 2 — `config.yaml` and `src/utils.py`
**Owner: Aishwary**

`config.yaml` holds every setting that affects a result:

```yaml
seed: 1337

paths:
  data_root: data/LA
  protocols: data/LA/ASVspoof2019_LA_cm_protocols
  figures: figures
  results: results

audio:
  sample_rate: 16000
  duration_seconds: 4.0     # changes in experiment E4
  normalise: peak
  pad_mode: repeat          # repeat the clip if it's too short

features:
  front_end: logmel         # logmel | lfcc | cqcc
  logmel:
    n_fft: 512
    win_length_ms: 25
    hop_length_ms: 10
    n_mels: 80
    f_min: 0
    f_max: 8000
    log_offset: 1.0e-6

inspect:
  real_utt_id: null         # filled in after the first run
  fake_utt_id: null
```

`src/utils.py` needs three small functions:

- `load_config(path="config.yaml")` — reads the YAML, turns the paths into
  absolute paths so scripts work from any folder, returns a dict
- `set_seed(seed)` — sets random, numpy and torch seeds so runs repeat
  identically. Import torch **inside** the function so the other tests can run
  without torch installed.
- `setup_logging()` — configures `logging` with timestamps. We use logging, not
  `print`.

Explain **why** window and hop are in milliseconds rather than samples before
you write it.

**Done when:** `python -c "from src.utils import load_config; print(load_config())"`
prints the config with absolute paths.

---

## Step 3 — `src/protocol.py`
**Owner: Pranjal**

The most important file in this phase. It reads a label file and gives us a
pandas table.

```python
load_protocol(partition, config) -> pd.DataFrame
summarise(df) -> pd.DataFrame
```

`partition` is `"train"`, `"dev"` or `"eval"`. Returns exactly these columns:

| Column | Notes |
|---|---|
| `speaker_id` | |
| `utt_id` | |
| `system_id` | `-` when real |
| `key` | `bonafide` or `spoof` |
| `label` | **0 = real, 1 = fake.** Fix this now, never change it. |
| `partition` | |
| `path` | full path to the `.flac` |

Requirements:

- Use `pandas.read_csv(sep=r"\s+", header=None)` — don't hand-roll a parser
- Import only pandas and the standard library. **No torch, no librosa.** This
  file must be testable before the dataset or torch exist.
- Raise a clear error naming the path if a label file is missing
- Don't check whether every `.flac` exists here; that's `verify.py`'s job

**Done when:** the tests in step 4 pass.

---

## Step 4 — `tests/test_basics.py`
**Owner: Pranjal**

One test file. Build a fake protocol file in a temp folder with six lines
(three real, three fake across two attacks) plus empty `.flac` files, then
check:

- right number of rows, right column names
- `label` is 0 for real, 1 for fake
- `system_id` is `-` on every real row
- `path` points where it should
- a missing label file raises an error naming the path
- an unknown partition name raises an error

These must pass **without the dataset downloaded**.

**Done when:** `pytest -q` is green.

---

## Step 5 — `scripts/download.py`
**Owner: Prateek**

Then **start the download immediately** and keep working while it runs. This is
the phase's big handoff — flag it in the format from `PROJECT.md` section 3.

Pure Python. Streams the 7.1 GB zip in chunks with a `tqdm` progress bar, and
resumes from a partial file using an HTTP `Range` header if it's interrupted —
which it will be. Verifies the zip opens before extracting, extracts into
`data/`, prints the resulting folder tree, and tells us to run `verify.py` next.

Add a `--skip-download` flag for extracting an already-downloaded file.

Remind us to delete `LA.zip` after extraction — it frees 7 GB.

**Done when:** `data/LA/` exists with the structure above.

---

## Step 6 — `src/audio.py`
**Owner: Pranjal**

Three small functions:

- `load_audio(path, sample_rate)` — reads a FLAC with `soundfile`, returns
  float32 mono. Only resample if the file isn't already 16 kHz, and log a
  warning if that ever fires, because it shouldn't.
- `normalise(wav, mode)` — peak normalisation, but return the array unchanged
  if it's all zeros rather than dividing by zero.
- `fix_length(wav, n_samples, pad_mode)` — trims if too long; if too short,
  either repeats the clip until it's long enough or zero-pads. Must return
  exactly `n_samples` samples every time.

Explain **why** every clip has to be the same length before you write it.

Add tests for these: short, exact and long inputs; an all-zeros input;
repeat-padding a 100-sample clip to 250.

**Done when:** `pytest -q` is green.

---

## Step 7 — `scripts/verify.py`
**Owner: Prateek**

The gate that says the dataset is actually usable. It should:

1. Check each expected folder exists
2. Load all three label files and print counts by partition and label
3. Compare against the expected counts above, printing PASS or MISMATCH per
   partition — warn, don't crash
4. Print the attack distribution and confirm train/dev have only A01–A06 and
   eval only A07–A19
5. Print the real:fake ratio per partition with a one-line note on why this
   means we use EER instead of accuracy
6. Check every `utt_id` has a file on disk; report how many are missing and
   list the first ten
7. Sample 100 files per partition (using the configured seed) and confirm
   16 kHz mono, reporting min/median/max duration
8. Write the whole report to `results/verification.txt` as well as the screen

Exit 0 on success, 1 on missing files or count mismatches.

**Done when:** it runs clean on the real dataset.

---

## Step 8 — log-mel in `src/features.py`
**Owner: Tania**

```python
logmel(wav, config) -> np.ndarray   # shape (n_mels, n_frames)
```

Computes a log-mel spectrogram using the settings from `config.yaml`. Work out
window and hop in samples from the millisecond values — don't hardcode 400 and
160. Use `librosa.feature.melspectrogram`, then `np.log(S + log_offset)`. Not
`power_to_db` — we want a plain log so the offset stays explicit and
configurable.

Before writing it, explain in plain language what a mel spectrogram is, why we
use one instead of the raw waveform, and what the 25 ms window and 10 ms hop
actually mean.

Write **only** log-mel. LFCC and CQCC are phase 2.

**Done when:** it returns the shape you predicted for a 4-second clip.

---

## Step 9 — `src/plots.py` and `scripts/inspect.py`
**Owner: Tania. This is the phase deliverable.**

`plots.py` draws waveforms and spectrograms. `inspect.py` ties it together:

```
python scripts/inspect.py
```

- loads config, sets the seed, sets up logging
- reads `inspect.real_utt_id` and `inspect.fake_utt_id` from the config
- if they're `null`, picks one of each at random from the **train** set using
  the seed, prints the chosen ids clearly, and tells us to paste them into
  `config.yaml` so the figure is frozen. For the fake one, prefer attack `A04`
  or `A06` — the older generators leave more visible traces, which makes a
  clearer first picture.
- loads both, normalises, fixes length, computes log-mel for each
- writes `figures/first_look.png`: a 2×2 grid, real on the left and fake on the
  right, waveform on top and spectrogram below. Axes in seconds and Hz, not
  frames and bins. Same colour scale on both spectrograms so they're
  comparable. Titles show the utterance id and the attack id.
- writes `results/inspect.json` recording both ids, their labels, the durations
  before and after length fixing, the feature settings used, the seed, and the
  output path
- logs the spectrogram shape, so we can see what the model will receive

Use matplotlib with the `Agg` backend so it works on a server with no display.

**Done when:** `figures/first_look.png` exists, is committed, and running the
script twice produces an identical file.

---

## Step 10 — `README.md` and phase close
**Owner: Aishwary**

`README.md`: project description, quickstart (clone, venv, install, download,
verify, inspect), the folder layout, the eight-phase table, the six experiments
table, and an **empty results table with the columns already there** — config,
front-end, EER %, min t-DCF, seed spread. Plus the dataset citation, team
members and supervisor.

The empty results table goes in now, not in phase 3.

Then run the close-out procedure in `PROJECT.md` section 3, stage 5.

---

## Phase 1 checklist

- [ ] Fresh clone + `pip install -r requirements.txt` gives a working setup
- [ ] `pytest -q` passes without the dataset present
- [ ] `python scripts/verify.py` shows the expected counts and exits 0
- [ ] `python scripts/inspect.py` writes `figures/first_look.png`
- [ ] Running it twice gives an identical file
- [ ] All four team members have done the above on their own machines
- [ ] 13 literature notes exist in `docs/literature/`
- [ ] `git log --stat` shows no audio file was ever committed
- [ ] `phases/phase-1/REPORT.md` and `HOW_TO_RUN.md` complete
- [ ] Tagged `phase-1-complete`

---

## Stop and ask us if

- the download link 404s
- the real counts differ from the expected table
- any file isn't 16 kHz mono
- something would need data committed to git
- a step seems to need a model, a training loop, or EER — those are phase 3
