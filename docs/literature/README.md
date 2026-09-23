# Literature notes

Thirteen references, one note each, using `TEMPLATE.md`. Split per our
work-split plan, due across weeks 1–3.

Copy the template to the filename below and commit it under your own name.

| # | Paper | Owner | File | Feeds |
|---|---|---|---|---|
| 1 | Todisco et al. — *ASVspoof 2019: Future Horizons* | Aishwary | `01-asvspoof2019-future-horizons.md` | vocabulary, the metric |
| 2 | *not yet chosen* | Aishwary | — | — |
| 3 | *not yet chosen* | Aishwary | — | — |
| 4 | Todisco et al. — CQCC features | Pranjal | `04-cqcc.md` | P2.2, E1 |
| 5 | Lavrentyeva et al. — light CNN / MFM | Pranjal | `05-light-cnn.md` | phase 3 baseline, E3 |
| 6 | RawNet2 — learned filterbank from the raw waveform | Prateek | `06-rawnet2.md` | context for E1 |
| 7 | AASIST — graph attention | Prateek | `07-aasist.md` | state of the art |
| 8 | wav2vec 2.0 — self-supervised representations | Prateek | `08-wav2vec2.md` | background for [9] |
| 9 | Tak et al. — wav2vec 2.0 as an anti-spoofing front-end | Tania | `09-wav2vec2-antispoofing.md` | stretch goal |
| 10 | RawBoost — signal-level augmentation | Tania | `10-rawboost.md` | E2 |
| 11 | SpecAugment | Tania | `11-specaugment.md` | E2 |
| 12 | ASVspoof 2021 | Tania | `12-asvspoof2021.md` | cross-dataset test, P6.2 |
| 13 | ASVspoof 2021 (companion) | Tania | `13-asvspoof2021-results.md` | cross-dataset test, P6.2 |

References [2] and [3] are not named in any of our planning documents, so we
still need to pick them. A survey of audio deepfake detection and something
on t-DCF are the obvious gaps.

Two things to get right: numbers come from the paper rather than memory, and
every number says which dataset it is on. An EER on ASVspoof 2021 DF is not
comparable with ours on 2019 LA, and putting both in one column without a
label is how a report ends up claiming something it cannot support.
