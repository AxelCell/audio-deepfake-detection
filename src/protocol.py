"""Read the ASVspoof 2019 LA label files into a pandas table.

Five space-separated columns:

    LA_0079 LA_T_1138215 - -   bonafide
    LA_0039 LA_E_2834763 - A11 spoof

    1 speaker id
    2 utterance id (the audio file is <utt_id>.flac)
    3 unused, always '-'
    4 attack id, '-' if real, A01-A19 if fake
    5 key: 'bonafide' or 'spoof'

pandas and the standard library only - no torch, no librosa - so the tests
for this file run before anyone has downloaded the dataset.
"""

from pathlib import Path

import pandas as pd

# 0 = real, 1 = fake. Every EER we ever report assumes this way round, so
# flipping it later would silently invert all our results.
LABEL_MAP = {"bonafide": 0, "spoof": 1}

# The train file is .trn and the other two are .trl. That's how the dataset
# ships, not a typo.
PARTITIONS = {
    "train": {
        "protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "flac_dir": "ASVspoof2019_LA_train",
    },
    "dev": {
        "protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "flac_dir": "ASVspoof2019_LA_dev",
    },
    "eval": {
        "protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
        "flac_dir": "ASVspoof2019_LA_eval",
    },
}

COLUMNS = ["speaker_id", "utt_id", "system_id", "key", "label", "partition", "path"]


def load_protocol(partition, config):
    """Load one partition's label file as a DataFrame.

    Doesn't check the .flac files exist - that's verify.py's job, and leaving
    it out is what lets this be tested without the dataset.
    """
    if partition not in PARTITIONS:
        raise ValueError(
            f"unknown partition {partition!r}; expected one of {sorted(PARTITIONS)}")

    spec = PARTITIONS[partition]
    protocol_path = Path(config["paths"]["protocols"]) / spec["protocol"]

    if not protocol_path.exists():
        raise FileNotFoundError(
            f"protocol file not found: {protocol_path}\n"
            f"Has the dataset been downloaded? Run: python scripts/download.py")

    df = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "utt_id", "_unused", "system_id", "key"],
        dtype=str,
    )
    df = df.drop(columns=["_unused"])

    unknown = set(df["key"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(
            f"unexpected key value(s) {sorted(unknown)} in {protocol_path}")

    df["label"] = df["key"].map(LABEL_MAP).astype(int)
    df["partition"] = partition

    flac_dir = Path(config["paths"]["data_root"]) / spec["flac_dir"] / "flac"
    df["path"] = df["utt_id"].map(lambda utt: str(flac_dir / f"{utt}.flac"))

    return df[COLUMNS]


def summarise(df):
    """One row per partition: how many real, how many fake, and the ratio."""
    rows = []
    for partition, group in df.groupby("partition", sort=False):
        n_real = int((group["label"] == 0).sum())
        n_fake = int((group["label"] == 1).sum())
        attacks = sorted(a for a in group["system_id"].unique() if a != "-")
        rows.append({
            "partition": partition,
            "real": n_real,
            "fake": n_fake,
            "total": len(group),
            "fake_per_real": round(n_fake / n_real, 2) if n_real else float("nan"),
            "n_attacks": len(attacks),
            "attacks": ",".join(attacks) if attacks else "-",
        })
    return pd.DataFrame(rows)
