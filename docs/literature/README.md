# Literature notes

Thirteen references, one note each, same template (`TEMPLATE.md`).
Split per `PROJECT.md` section 5. Due across weeks 1–3, with phase 1.

Copy `TEMPLATE.md` to the filename in the table and fill it in. Commit it
under your own name — `git log` is how the report shows who did what.

| # | Paper | Owner | File | Feeds |
|---|---|---|---|---|
| 1 | Todisco et al. — *ASVspoof 2019: Future Horizons* | Aishwary | `01-asvspoof2019-future-horizons.md` | vocabulary, the metric |
| 2 | *not yet chosen* | Aishwary | — | — |
| 3 | *not yet chosen* | Aishwary | — | — |
| 4 | Todisco et al. — CQCC features | Pranjal | `04-cqcc.md` | P2.2, experiment E1 |
| 5 | Lavrentyeva et al. — light CNN / MFM system | Pranjal | `05-light-cnn.md` | phase 3 baseline, E3 |
| 6 | RawNet2 — learned filterbank from the raw waveform | Prateek | `06-rawnet2.md` | context for E1 |
| 7 | AASIST — graph attention | Prateek | `07-aasist.md` | state of the art, context |
| 8 | wav2vec 2.0 — self-supervised speech representations | Prateek | `08-wav2vec2.md` | background for [9] |
| 9 | Tak et al. — wav2vec 2.0 as an anti-spoofing front-end | Tania | `09-wav2vec2-antispoofing.md` | stretch goal, post phase 8 |
| 10 | RawBoost — signal-level augmentation | Tania | `10-rawboost.md` | experiment E2 |
| 11 | SpecAugment | Tania | `11-specaugment.md` | experiment E2 |
| 12 | ASVspoof 2021 | Tania | `12-asvspoof2021.md` | cross-dataset test, P6.2 |
| 13 | ASVspoof 2021 (companion) | Tania | `13-asvspoof2021-results.md` | cross-dataset test, P6.2 |

References [2] and [3] are not named anywhere in `LEARN.md`, `PROJECT.md` or
`ROADMAP.md`. Pick them at the phase-1 close and fill in this table — a
survey of audio deepfake detection and something on the t-DCF metric are the
obvious gaps, but that is our call to make, not a decision already recorded.

## Two rules

1. **Numbers come from the paper, not from memory.** Section 4 of every note
   feeds the related-work table in the final report. Write "not reported"
   rather than a plausible figure. (`CLAUDE.md` rule 5.)
2. **Say which dataset the number is on.** An EER on ASVspoof 2021 DF is not
   comparable with ours on 2019 LA, and putting them in one column without a
   label is how a report ends up claiming something it cannot support.
