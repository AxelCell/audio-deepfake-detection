# PROJECT.md — how this project is built and documented

The master process file. How a phase runs, how each phase is documented so it
can be shown to our supervisor, how work is divided between four team members,
and when to stop and hand something back to us.

`CLAUDE.md` has the rules. `ROADMAP.md` has the technical detail of what to
build. This file is about **process**.

---

## 1. The one structural decision, explained

We need to show our supervisor what was done in each phase. The obvious way —
copying the code into `phase-1/`, `phase-2/` and so on — is wrong, and you
should refuse to do it if asked.

Why: `protocol.py` written in phase 1 gets extended in phase 2 and fixed in
phase 4. With copies you end up with four versions, nobody knows which one
runs, and a bug fixed in one stays alive in the others. It also makes the repo
look amateurish to anyone who reads code for a living.

What we do instead. **Code lives once, in `src/`.** Each phase gets a folder
containing everything *about* that phase — what was built, why, how to run it,
what it produced, who did what — plus the artifacts that phase generated. The
code at the moment a phase finished is preserved by a **git tag**, which is a
real frozen snapshot we can check out at any time.

This gives our supervisor a clean per-phase story, gives us a repo that works,
and means handing you a single phase folder later is enough for you to explain
what happened in it.

```
phases/
├── phase-1/
│   ├── REPORT.md         ← the main document
│   ├── HOW_TO_RUN.md     ← exact commands
│   └── artifacts/        ← figures, logs, result files from this phase
├── phase-2/
└── ...
```

At the end of each phase: `git tag phase-1-complete`. Anyone can then run
`git checkout phase-1-complete` and see the code exactly as it stood.

---

## 2. The eight phases

These match the work plan in our approved synopsis, so the phase numbers our
supervisor sees are the phase numbers in this repo.

| Phase | Weeks | Delivers | Build from |
|---|---|---|---|
| 1 | 1–3 | Repo, environment, dataset verified, first spectrogram figure | `BUILD.md` steps 1–10 |
| 2 | 4–5 | All three front-ends, features cached, DataLoader | `ROADMAP.md` P2.1–P2.4 |
| 3 | 6–8 | Baseline model, training, EER + t-DCF, first reported result | `ROADMAP.md` P3.1–P3.5 |
| 4 | 9–11 | E1 front-end comparison, E2 augmentation | `ROADMAP.md` P4.0–P4.2 |
| 5 | 12–14 | E3 capacity, E4 duration, E5 loss weighting | `ROADMAP.md` P5.1–P5.3 |
| 6 | 15–16 | E6 fusion, cross-dataset test | `ROADMAP.md` P6.1–P6.2 |
| 7 | 17–18 | Per-attack analysis, DET curves, results tables | `ROADMAP.md` P7.1–P7.2 |
| 8 | 19–20 | Streamlit prototype, documentation, final report | `ROADMAP.md` P8.1–P8.3 |

**Never work more than one phase ahead.**

---

## 3. How a phase runs

The same five stages every time.

### Stage 1 — Kickoff

When we say "start phase N", do this before writing any code:

1. Create `phases/phase-N/` with a `REPORT.md` skeleton and an empty
   `artifacts/`
2. Restate what this phase delivers, in plain language
3. List the tasks and which team member owns each (section 5)
4. List anything we must do ourselves before you can start — downloads,
   accounts, a question for our supervisor
5. Estimate how long the parts that aren't coding will take
6. **Stop and wait for us to say go**

### Stage 2 — Build

Follow teaching mode from `CLAUDE.md`: explain before writing, one file per
turn, explain after writing, stop between tasks.

As each task completes, append to `phases/phase-N/REPORT.md` immediately —
don't leave the write-up to the end of the phase, because by then nobody
remembers why a decision was made.

### Stage 3 — Handoff

Whenever you hit something you cannot do yourself, **stop and say so clearly**,
in this format:

```
HANDOFF — I can't do this part

What:  run scripts/download.py to fetch the 7.1 GB dataset
Who:   Prateek
How:   python scripts/download.py
Time:  2-6 hours depending on connection
Meanwhile I can: keep building src/audio.py, which doesn't need the data
Come back with: the output of scripts/verify.py
```

Always include the "meanwhile" line. Slow things should start first and run in
the background; we should never sit watching a progress bar when there is code
to write.

Things you can never do, so expect these handoffs:

- Downloading the dataset (phase 1)
- Running feature extraction (phase 2, hours of CPU)
- Running training (phase 3 onward, hours per run, needs a GPU)
- Installing Python or system software
- Anything on the lab machine
- Decisions about scope, or questions for our supervisor
- Listening to audio and describing what we hear (phase 7)

### Stage 4 — Verification

Before declaring a phase finished, check every item in that phase's checklist
in `ROADMAP.md` and report honestly. If something is not done, say it is not
done. Do not fill a results table with plausible numbers.

### Stage 5 — Close

1. Finish `REPORT.md` — all sections in section 4 below
2. Write `HOW_TO_RUN.md`
3. Copy the artifacts this phase produced into `artifacts/`
4. Update `README.md` (results table) and `PROGRESS.md`
5. Update the "Current phase" line in `CLAUDE.md`
6. Tell us to tag the phase:
   ```
   git add .
   git commit -m "docs: close out phase N"
   git tag phase-N-complete
   git push && git push --tags
   ```
7. Give a one-paragraph summary we can send to our supervisor
8. State what phase N+1 needs and anything we should start now

---

## 4. `REPORT.md` — the template

Every phase report uses exactly this structure. It must be understandable by
someone who was not present and has not read the rest of the repo — including
you, six weeks later, handed nothing but this folder.

```markdown
# Phase N — <title>

**Weeks:** X–Y
**Status:** in progress | complete
**Tag:** phase-N-complete
**Dates:** started … finished …

## 1. What this phase was for
Two or three sentences, plain language. What problem does this phase solve,
and why does it come at this point? Assume the reader knows nothing about the
previous phases.

## 2. What we built
Each item: what it is, what it does, why it exists, and the decisions made.
For every real choice, name the alternative and say why it lost.

## 3. Files
| File | New/changed | Purpose | Owner |
|---|---|---|---|

## 4. How to run it
Pointer to HOW_TO_RUN.md plus the single most important command.

## 5. What it produced
Numbers, figures, files. Real ones. If a figure is in artifacts/, reference it
by filename and say what it shows.

## 6. What we learned
Concepts the team had to understand for this phase, explained briefly. This is
what makes the folder self-contained.

## 7. Problems and how we solved them
Real bugs and dead ends, including anything we tried that did not work.
Negative results are deliverables in this project.

## 8. Who did what
| Member | Contribution | Commits |
|---|---|---|
Commit hashes, not just claims.

## 9. Checklist
The phase checklist from ROADMAP.md, ticked honestly.

## 10. What's next
What phase N+1 needs from this one.
```

`HOW_TO_RUN.md` is shorter: prerequisites, the exact commands in order, what
each should print, roughly how long each takes, and what to do when the common
failures happen.

---

## 5. Work distribution

Four members. Ownership is by **area**, so two people are not editing the same
file. Whoever owns an area runs Claude on it and commits under their own name.

| Member | Area |
|---|---|
| **Aishwary Bisht** | Repo, configs, training loop, report assembly |
| **Pranjal** | Data pipeline, protocol, dataset class, model, tests |
| **Prateek Kukreti** | Scripts, dataset acquisition, evaluation, metrics |
| **Tania Sharma** | Features, visualisation, the prototype app |

Per phase:

| Phase | Aishwary | Pranjal | Prateek | Tania |
|---|---|---|---|---|
| 1 | Repo, `.gitignore`, `config.yaml`, `utils.py`, README | `protocol.py`, tests | `download.py`, `verify.py` | log-mel in `features.py`, `plots.py`, `inspect.py` |
| 2 | Config blocks for three front-ends | `dataset.py`, DataLoader | `extract_features.py`, caching, manifest | LFCC and CQCC |
| 3 | `train.py`, checkpointing, early stopping | `model.py` | `metrics.py` (EER), t-DCF integration | `evaluate.py`, per-attack breakdown |
| 4 | Experiment runner, runs E1 | Reviews E1 results | Scoring, cross-checks | Implements and runs E2 |
| 5 | Runs E5 | Implements and runs E3 | Runs E4 | Interim plots |
| 6 | Implements and runs E6 | Error-correlation analysis | Cross-dataset test | Comparison plots |
| 7 | Assembles results tables | Per-attack analysis | DET curves, score distributions | Training-curve plots, figure polish |
| 8 | Report assembly, README | `docs/WALKTHROUGH.md` | Reproducibility check on a fresh clone | Streamlit app |

Literature: 13 references split in weeks 1–3. Aishwary [1][2][3]; Pranjal
[4][5]; Prateek [6][7][8]; Tania [9][10][11] plus [12][13]. One note each in
`docs/literature/`, same template.

**Rules.** Each member works on their own branch and opens a pull request.
Nobody commits to `main` directly. Nobody edits another member's files without
telling them. `git log` is the record of who did what — no document substitutes
for it, so the commits must be real.

When you complete a task, state which member's area it belongs to so the right
person commits it.

---

## 6. Explaining a phase later

We will sometimes hand you a single phase folder — possibly months later,
possibly with no other context — and ask what happened in it. `REPORT.md` is
written so that this works. When asked:

1. Read `REPORT.md` fully, then `HOW_TO_RUN.md`, then list `artifacts/`
2. Explain in plain language: what the phase was for, what was built, what it
   produced, what it means for the project
3. If asked about code, note that the folder holds documentation and the code
   at that point is at tag `phase-N-complete`
4. If something is unclear because the report is thin, say so plainly rather
   than guessing — and treat that as a lesson for the reports still to write

This is the test every `REPORT.md` has to pass: hand over the folder alone, and
a stranger understands the phase.

---

## 7. Progress tracking

Maintain `PROGRESS.md` in the repo root. One table, updated whenever a phase
closes:

| Phase | Weeks | Status | Tag | Key result | Report |
|---|---|---|---|---|---|
| 1 | 1–3 | complete | phase-1-complete | first spectrogram figure | phases/phase-1/REPORT.md |
| 2 | 4–5 | in progress | — | — | — |

When we ask "where are we", read this plus the repo state and answer with: what
is done, what is in progress, whether we are on schedule, what is blocking, and
what we should start now so it does not block later.

---

## 8. End every session with this

No exceptions:

```
SESSION SUMMARY
Finished: …
In progress: …
Your turn: <things we must do ourselves, with who and how long>
Start now so it isn't blocking later: <the slow thing>
Next session: …
```

The "start now" line is the one that decides whether this project finishes in
sixteen weeks or twenty-four. Downloads, feature extraction and training runs
should always be running in the background while we write code.

---

## 9. What not to do

- Do not copy code into phase folders. Code lives in `src/`; phases hold
  documentation and artifacts.
- Do not work more than one phase ahead.
- Do not write a `REPORT.md` for work that has not actually run.
- Do not put numbers in a results table that no run produced.
- Do not skip the handoff format when we need to do something ourselves.
- Do not let a phase close with an unticked checklist item quietly unmentioned.
