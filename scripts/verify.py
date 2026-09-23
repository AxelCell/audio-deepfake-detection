"""Check the downloaded dataset is complete and usable.

    python scripts/verify.py

Exits 0 if everything checks out, 1 if files are missing or the counts are
wrong. Writes the same report to results/verification.txt.
"""

import os
import sys

# scripts/inspect.py shadows the standard library's inspect module, which
# numpy and librosa import. Replace scripts/ on sys.path with the repo root.
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse                                              # noqa: E402
import statistics                                            # noqa: E402
from pathlib import Path                                     # noqa: E402

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
import soundfile as sf                                       # noqa: E402

from src.protocol import PARTITIONS, load_protocol, summarise  # noqa: E402
from src.utils import load_config, repo_relative, setup_logging  # noqa: E402

# From the ASVspoof 2019 LA documentation. A difference here nearly always
# means the download or extraction did not finish.
EXPECTED = {
    "train": {"real": 2580, "fake": 22800, "total": 25380},
    "dev":   {"real": 2548, "fake": 22296, "total": 24844},
    "eval":  {"real": 7355, "fake": 63882, "total": 71237},
}

# A01-A06 in train and dev, A07-A19 in eval only.
EXPECTED_ATTACKS = {
    "train": {"A%02d" % i for i in range(1, 7)},
    "dev":   {"A%02d" % i for i in range(1, 7)},
    "eval":  {"A%02d" % i for i in range(7, 20)},
}

N_SAMPLED = 100
EXPECTED_SR = 16000


class Report:
    """Collects lines so the same text goes to the screen and to a file."""

    def __init__(self):
        self.lines = []
        self.failed = False

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def ok(self, text):
        self("  PASS     " + text)

    def fail(self, text):
        self("  FAIL     " + text)
        self.failed = True

    def warn(self, text):
        self("  MISMATCH " + text)
        self.failed = True

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print("\nReport written to " + str(path))


def check_folders(config, r):
    r("1. Folder structure")
    data_root = Path(config["paths"]["data_root"])
    targets = [Path(config["paths"]["protocols"])]
    targets += [data_root / s["flac_dir"] / "flac" for s in PARTITIONS.values()]

    for path in targets:
        name = path.parent.name + "/" + path.name
        r.ok(name) if path.is_dir() else r.fail(f"{name} is missing ({path})")
    r()


def load_all(config, r):
    r("2. Label files")
    frames = {}
    for partition in PARTITIONS:
        try:
            frames[partition] = load_protocol(partition, config)
            r.ok("{:<6} {:>7,} rows".format(partition, len(frames[partition])))
        except FileNotFoundError as exc:
            r.fail("{:<6} {}".format(partition, str(exc).splitlines()[0]))
    r()
    return frames


def check_counts(frames, r):
    r("3. Counts against the expected ASVspoof 2019 LA figures")
    r("  {:<10}{:>8}{:>9}{:>9}   {:>14}".format(
        "partition", "real", "fake", "total", "expected total"))
    for partition, df in frames.items():
        real = int((df["label"] == 0).sum())
        fake = int((df["label"] == 1).sum())
        exp = EXPECTED[partition]
        r("  {:<10}{:>8,}{:>9,}{:>9,}   {:>14,}".format(
            partition, real, fake, len(df), exp["total"]))
        if (real, fake, len(df)) == (exp["real"], exp["fake"], exp["total"]):
            r.ok(partition + " matches exactly")
        else:
            r.warn("{}: got real={} fake={} total={}, expected real={} fake={} "
                   "total={}".format(partition, real, fake, len(df),
                                     exp["real"], exp["fake"], exp["total"]))
    r()


def check_attacks(frames, r):
    r("4. Attack distribution")
    for partition, df in frames.items():
        found = {a for a in df["system_id"].unique() if a != "-"}
        expected = EXPECTED_ATTACKS[partition]
        r("  {:<6} {} attacks: {}".format(
            partition, len(found), ",".join(sorted(found))))

        if found == expected:
            r.ok(partition + " has exactly the expected attacks")
            continue
        detail = []
        if found - expected:
            detail.append("unexpected " + str(sorted(found - expected)))
        if expected - found:
            detail.append("missing " + str(sorted(expected - found)))
        r.warn(partition + ": " + "; ".join(detail))

    r("  A01-A06 in train/dev, A07-A19 in eval only. The eval attacks are")
    r("  generators the model never sees in training, so the eval number")
    r("  measures generalisation rather than memorisation.")
    r()


def check_balance(frames, r):
    r("5. Class balance")
    for _, row in summarise(pd.concat(frames.values(), ignore_index=True)).iterrows():
        pct = 100.0 * row["fake"] / row["total"] if row["total"] else 0.0
        r("  {:<6} 1 real : {:.2f} fake   ({:.1f}% of clips are fake)".format(
            row["partition"], row["fake_per_real"], pct))
    r()
    r("  This is why we never report accuracy - answering 'fake' every time")
    r("  scores about 90% here. We use EER and min t-DCF. (CLAUDE.md rule 4)")
    r()


def check_files_exist(frames, r):
    r("6. Audio files on disk")
    for partition, df in frames.items():
        missing = [p for p in df["path"] if not Path(p).exists()]
        if not missing:
            r.ok("{:<6} all {:,} files present".format(partition, len(df)))
            continue
        r.fail("{:<6} {:,} of {:,} files missing".format(
            partition, len(missing), len(df)))
        for path in missing[:10]:
            r("             " + Path(path).name)
        if len(missing) > 10:
            r("             ... and {:,} more".format(len(missing) - 10))
    r()


def check_audio_properties(frames, config, r):
    r("7. Audio properties ({} random files per partition, seed {})".format(
        N_SAMPLED, config["seed"]))
    rng = np.random.default_rng(config["seed"])

    for partition, df in frames.items():
        paths = [p for p in df["path"] if Path(p).exists()]
        if not paths:
            r.fail("{:<6} no readable files to sample".format(partition))
            continue

        picked = rng.choice(np.array(paths, dtype=object),
                            size=min(N_SAMPLED, len(paths)), replace=False)
        durations, bad = [], []
        for path in picked:
            try:
                info = sf.info(str(path))
            except Exception as exc:
                bad.append(f"{Path(path).name}: {exc}")
                continue
            if info.samplerate != EXPECTED_SR or info.channels != 1:
                bad.append("{}: {} Hz, {} channel(s)".format(
                    Path(path).name, info.samplerate, info.channels))
            durations.append(info.frames / float(info.samplerate))

        if durations:
            r("  {:<6} duration min {:.2f}s  median {:.2f}s  max {:.2f}s   "
              "(n={})".format(partition, min(durations),
                              statistics.median(durations), max(durations),
                              len(durations)))
        if not bad:
            r.ok("{:<6} all sampled files are {} Hz mono".format(
                partition, EXPECTED_SR))
        else:
            r.fail("{}: {} file(s) not 16 kHz mono".format(partition, len(bad)))
            for line in bad[:10]:
                r("             " + line)
    r()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    r = Report()
    r("=" * 70)
    r("ASVspoof 2019 LA - dataset verification")
    r("=" * 70)
    r("data_root: " + repo_relative(config["paths"]["data_root"], config))
    r()

    check_folders(config, r)
    frames = load_all(config, r)

    if frames:
        check_counts(frames, r)
        check_attacks(frames, r)
        check_balance(frames, r)
        check_files_exist(frames, r)
        check_audio_properties(frames, config, r)
    else:
        r.fail("no label files could be read - nothing else can be checked")
        r()

    r("=" * 70)
    if r.failed:
        r("RESULT: problems found. See the FAIL and MISMATCH lines above.")
        r("If counts are short the download is probably incomplete:")
        r("  python scripts/download.py       (it resumes, it does not restart)")
    else:
        r("RESULT: all checks passed. The dataset is ready.")
        r("Next:  python scripts/inspect.py")
    r("=" * 70)

    r.save(Path(config["paths"]["results"]) / "verification.txt")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
