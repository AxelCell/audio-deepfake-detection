"""Make the phase 1 figure: one real clip beside one fake one.

    python scripts/inspect.py

Writes figures/first_look.png and results/inspect.json.
"""

import os
import sys

# Running a script from scripts/ puts that folder first on sys.path, and this
# file is called inspect.py - which shadows the standard library's inspect
# that numpy, librosa and matplotlib all import internally. Replace the entry
# with the repo root rather than adding to it.
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse                                              # noqa: E402
import json                                                  # noqa: E402
import logging                                               # noqa: E402
from pathlib import Path                                     # noqa: E402

from src.audio import fix_length, load_audio, normalise      # noqa: E402
from src.features import logmel                              # noqa: E402
from src.plots import first_look_figure, save_figure         # noqa: E402
from src.protocol import load_protocol                       # noqa: E402
from src.utils import load_config, set_seed, setup_logging   # noqa: E402

# A04 and A06 are older generators and leave more visible traces, which makes
# a clearer first picture.
PREFERRED_ATTACKS = ["A04", "A06"]


def choose_utterances(df, config):
    """Return (real_id, fake_id, picked_at_random)."""
    real_id = config["inspect"]["real_utt_id"]
    fake_id = config["inspect"]["fake_utt_id"]

    # Check whichever ids the config gives us. Without this a typo falls
    # through to .iloc[0] below and dies with a bare IndexError.
    known = set(df["utt_id"])
    for field, utt_id in (("real_utt_id", real_id), ("fake_utt_id", fake_id)):
        if utt_id and utt_id not in known:
            raise ValueError(
                f"inspect.{field} is {utt_id}, which is not in the train "
                f"partition. Check config.yaml, or set it to null to have one "
                f"picked at random.")

    if real_id and fake_id:
        return real_id, fake_id, False

    real_pool = df[df["label"] == 0]
    fake_pool = df[df["label"] == 1]

    preferred = fake_pool[fake_pool["system_id"].isin(PREFERRED_ATTACKS)]
    if len(preferred):
        fake_pool = preferred
    else:
        logging.warning("no A04/A06 utterances found, picking any spoof")

    # set_seed has already run, so sample() is reproducible.
    real_id = real_id or real_pool.sample(1)["utt_id"].iloc[0]
    fake_id = fake_id or fake_pool.sample(1)["utt_id"].iloc[0]
    return real_id, fake_id, True


def prepare(row, config):
    """Load one utterance and carry it through to a log-mel spectrogram."""
    sample_rate = config["audio"]["sample_rate"]
    n_samples = int(sample_rate * config["audio"]["duration_seconds"])

    wav = load_audio(row["path"], sample_rate)
    duration_before = len(wav) / float(sample_rate)

    wav = normalise(wav, config["audio"]["normalise"])
    wav = fix_length(wav, n_samples, config["audio"]["pad_mode"])
    duration_after = len(wav) / float(sample_rate)

    spectrogram = logmel(wav, config)
    logging.info("%s  %s  %.2fs -> %.2fs  spectrogram %s",
                 row["utt_id"], row["key"], duration_before, duration_after,
                 spectrogram.shape)

    return {
        "utt_id": row["utt_id"],
        "system_id": row["system_id"],
        "key": row["key"],
        "label": int(row["label"]),
        "wav": wav,
        "spectrogram": spectrogram,
        "duration_before_s": round(duration_before, 4),
        "duration_after_s": round(duration_after, 4),
        "spectrogram_shape": list(spectrogram.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config["seed"])

    df = load_protocol("train", config)
    real_id, fake_id, was_random = choose_utterances(df, config)

    real = prepare(df[df["utt_id"] == real_id].iloc[0], config)
    fake = prepare(df[df["utt_id"] == fake_id].iloc[0], config)

    figure_path = Path(config["paths"]["figures"]) / "first_look.png"
    save_figure(first_look_figure(real, fake, config), figure_path)
    logging.info("wrote %s", figure_path)

    keep = ("utt_id", "system_id", "key", "label",
            "duration_before_s", "duration_after_s", "spectrogram_shape")
    record = {
        "real": {k: real[k] for k in keep},
        "fake": {k: fake[k] for k in keep},
        "seed": config["seed"],
        "audio": config["audio"],
        "features": config["features"],
        "figure": str(figure_path),
    }
    results_path = Path(config["paths"]["results"]) / "inspect.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logging.info("wrote %s", results_path)

    print()
    print("-" * 70)
    print("real: " + real["utt_id"])
    print("fake: " + fake["utt_id"] + "  (attack " + fake["system_id"] + ")")
    print("figure: " + str(figure_path))
    if was_random:
        print("\nThese were picked at random. Paste them into config.yaml so")
        print("everyone reproduces the same picture:\n")
        print("inspect:")
        print("  real_utt_id: " + real["utt_id"])
        print("  fake_utt_id: " + fake["utt_id"])
    print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
