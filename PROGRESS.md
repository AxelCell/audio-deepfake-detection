# PROGRESS

Updated whenever a phase closes. One table, plus the current blocker.

| Phase | Weeks | Status | Tag | Key result | Report |
|---|---|---|---|---|---|
| 1 | 1–3 | **complete** | `phase-1-complete` | dataset verified, first figure | `phases/phase-1/REPORT.md` |
| 2 | 4–5 | next | — | — | — |
| 3 | 6–8 | not started | — | — | — |
| 4 | 9–11 | not started | — | — | — |
| 5 | 12–14 | not started | — | — | — |
| 6 | 15–16 | not started | — | — | — |
| 7 | 17–18 | not started | — | — | — |
| 8 | 19–20 | not started | — | — | — |

## Phase 1 closed, 2026-09-21

The dataset is downloaded, extracted and verified, and the figure exists.

`scripts/verify.py` exits 0. All three partitions match the published
ASVspoof 2019 LA counts exactly — train 25,380, dev 24,844, eval 71,237 — with
A01–A06 in train and dev and all thirteen of A07–A19 in eval only. All 121,461
files are present and every sampled file is 16 kHz mono. Measured class
balance is 1 real : 8.8 fake, which is the reason we report EER and not
accuracy.

`figures/first_look.png` compares `LA_T_8970804` (bona fide) against
`LA_T_3653342` (spoof, attack A06). Both ids are frozen in `config.yaml`, and
three consecutive runs produced SHA-256 `9402b4e8…` every time.

Artifacts are in `phases/phase-1/artifacts/`.

## Still open, carried into phase 2

1. **13 literature notes.** `docs/literature/TEMPLATE.md` and the index are in
   place with the per-member split. The notes are each member's own reading.
   References [2] and [3] have not been chosen.
2. **Only one machine.** The checklist wants all four members to run the
   pipeline on their own machines. One has.
3. **Branch protection and per-member identities.** `main` should be
   protected and every member should commit under their own name, so `git log`
   is the record of who did what.
4. **Each member needs their own copy of the dataset.** 7.12 GB, several
   hours, and `scripts/download.py` resumes if it drops. Start it early —
   nothing past phase 1 runs without it.

## Next

Phase 2: LFCC and CQCC front-ends, the dataset class and DataLoader, and
feature caching over all 121,461 files.
