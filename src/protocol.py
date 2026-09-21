"""Read ASVspoof 2019 LA label files into a pandas table.

Imports pandas and the standard library only - no torch, no librosa. That is
deliberate: the tests for this file must run on a fresh machine before anyone
has downloaded the dataset or installed a deep learning stack.

A label file has five space-separated columns:

    LA_0079 LA_T_1138215 - -   bonafide
    LA_0039 LA_E_2834763 - A11 spoof

      1 speaker id
      2 utterance id  (the audio file is <utt_id>.flac)
      3 unused, always '-', discarded
      4 attack id: '-' if real, A01-A19 if fake
      5 key: 'bonafide' or 'spoof'
"""

from pathlib import Path

import pandas as pd

# 0 = real, 1 = fake. Fixed in phase 1, never changed.
# Every EER and t-DCF number in the project assumes this orientation, so
# flipping it later would silently invert every result we have ever reported.
LABEL_MAP = {"bonafide": 0, "spoof": 1}

# Which protocol file and which audio folder belong to each partition.
# The train file is '.trn' and the other two are '.trl' - that asymmetry is
# how the dataset ships, not a typo.
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

    partition : 'train', 'dev' or 'eval'
    config    : the dict from utils.load_config()

    Returns a DataFrame with the columns in COLUMNS. Does NOT check that the
    .flac files actually exist - that is scripts/verify.py's job, and keeping
    it out of here is what lets these functions be tested without the dataset.
    """
    if partition not in PARTITIONS:
        raise ValueError(
            f"unknown partition {partition!r}; "
            f"expected one of {sorted(PARTITIONS)}"
        )

    spec = PARTITIONS[partition]
    protocol_path = Path(config["paths"]["protocols"]) / spec["protocol"]

    if not protocol_path.exists():
        raise FileNotFoundError(
            f"protocol file not found: {protocol_path}\n"
            f"Has the dataset been downloaded and extracted? "
            f"Run: python scripts/download.py"
        )

    # read_csv with a whitespace separator rather than a hand-rolled parser:
    # it handles runs of spaces, trailing newlines and blank lines for free.
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
            f"unexpected key value(s) {sorted(unknown)} in {protocol_path}; "
            f"expected only {sorted(LABEL_MAP)}"
        )

    df["label"] = df["key"].map(LABEL_MAP).astype(int)
    df["partition"] = partition

    flac_dir = Path(config["paths"]["data_root"]) / spec["flac_dir"] / "flac"
    df["path"] = df["utt_id"].map(lambda utt: str(flac_dir / f"{utt}.flac"))

    return df[COLUMNS]


def summarise(df):
    """One row per partition: how many real, how many fake, and the ratio.

    Used by scripts/verify.py and handy in a notebook. Returns a DataFrame
    rather than printing, so the caller decides how to display it.
    """
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
