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

All ten BUILD.md steps have been **written**. Steps 1, 2, 3, 4, 6 and 8 are
also **verified** — 19 tests pass without the dataset present.

Three things are written but not yet proven against real data, because the
dataset has not been downloaded:

- `scripts/download.py` — extract path tested against a miniature zip; the
  7.12 GB download itself has not been run
- `scripts/verify.py` — all eight checks exercised against a synthetic
  dataset; never run on the real one
- `scripts/inspect.py` — produces a correct, byte-identical figure from
  synthetic audio; `figures/first_look.png` does **not** exist yet

## Blocking

1. **The dataset is not downloaded.** `data/` is empty. This blocks the phase
   1 deliverable. ~7.12 GB, 2–6 hours, unattended. Owner: Prateek.
2. **This is not a git repository yet.** No `git init`, no remote, no branch
   protection, no per-member identities. Until that exists there is no record
   of who wrote what, which PROJECT.md §5 treats as the primary deliverable of
   the work split. Owner: Aishwary.
3. **13 literature notes** not started. Weeks 1–3, split across all four
   members. Owner: everyone.

## Start now so it isn't blocking later

The download. Everything else in phase 1 is finished or fast; this one thing
takes hours and nothing can close without it.
