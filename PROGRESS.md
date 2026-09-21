# PROGRESS

Updated whenever a phase closes. One table, plus the current blocker.

| Phase | Weeks | Status | Tag | Key result | Report |
|---|---|---|---|---|---|
| 1 | 1–3 | **in progress** | — | — | `phases/phase-1/REPORT.md` |
| 2 | 4–5 | not started | — | — | — |
| 3 | 6–8 | not started | — | — | — |
| 4 | 9–11 | not started | — | — | — |
| 5 | 12–14 | not started | — | — | — |
| 6 | 15–16 | not started | — | — | — |
| 7 | 17–18 | not started | — | — | — |
| 8 | 19–20 | not started | — | — | — |

## Where phase 1 actually stands

All ten BUILD.md steps are **written and exercised**. The repository is now
under git (`9d88497`), 19 tests pass without the dataset, and `verify.py` and
`inspect.py` have both been run end to end against a synthetic ASVspoof-shaped
dataset: all eight verification checks fire, the figure renders, and two
consecutive runs of `inspect.py` produce a byte-identical PNG
(`f0df95b9...`). The figure settings are proven; only the audio behind them
is still missing.

### Checklist

| Item | State |
|---|---|
| Fresh clone + `pip install -r requirements.txt` | done |
| `pytest -q` passes without the dataset | done — 19 passed |
| Repository under git, no audio in history | done |
| `inspect.py` twice gives an identical file | done, on synthetic audio |
| `phases/phase-1/REPORT.md` + `HOW_TO_RUN.md` | done |
| `verify.py` exits 0 on the real dataset | **blocked — download** |
| `figures/first_look.png` committed | **blocked — download** |
| All four members have run it on their own machine | team task |
| 13 literature notes | template + index only — see below |
| Tagged `phase-1-complete` | not yet — needs the two blocked rows |

## Blocking

1. **The dataset download is running but slow.** Started 2026-09-21 14:45,
   measured at roughly 6 MB/min, which puts the 7.12 GB at somewhere near 20
   hours. It resumes from a partial file, so leave it running and re-run
   `python scripts/download.py` after any interruption. This is the only
   thing standing between us and closing the phase. Owner: Prateek.
2. **13 literature notes not written.** `docs/literature/TEMPLATE.md` and the
   index in `docs/literature/README.md` are in place, with the split from
   PROJECT.md section 5 and the papers named in LEARN.md. The notes
   themselves are each member's own reading and are not something to
   generate — section 4 of the template records EER figures that go straight
   into the final report's related-work table, and a plausible-looking wrong
   number there is worse than a blank one (CLAUDE.md rule 5).
   References [2] and [3] are not named in any planning document and still
   need choosing. Owner: everyone.
3. **No remote, no branch protection, no per-member identities.** The initial
   import is one commit under one name because the repo was created after the
   code was written. PROJECT.md section 5 wants `git log` to be the record of
   who did what; that starts from the next commits. Owner: Aishwary.

## Start now so it isn't blocking later

The download, and the thirteen notes. Both are slow by nature and neither gets
faster by waiting.
