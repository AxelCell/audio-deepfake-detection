"""Produce the phase 1 deliverable: one picture of a real clip beside a fake one.

    python scripts/inspect.py

Writes figures/first_look.png and results/inspect.json.

On the first run the utterance ids are chosen at random from the train set
using the configured seed. The script prints them and asks you to paste them
into config.yaml, which freezes the figure for good.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Read this before moving the imports below.
#
# This file is called inspect.py, and Python's standard library already has a
# module called `inspect`. Running `python scripts/inspect.py` puts scripts/
# at the front of sys.path, so any `import inspect` anywhere in the process
# finds THIS file instead of the standard one. numpy, librosa and matplotlib
# all import `inspect` internally, so they end up re-executing this script
# halfway through its own imports and the whole thing dies with a confusing
# "circular import" error pointing at src/audio.py, which is innocent.
#
# So: drop the script's own directory from sys.path, then add the repo root.
# BUILD.md specifies the command `python scripts/inspect.py`, so renaming the
# file is not an option.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or os.getcwd()) != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import argparse                                              # noqa: E402
import json                                                  # noqa: E402
import logging                                               # noqa: E402
from pathlib import Path                                     # noqa: E402

from src.audio import fix_length, load_audio, normalise      # noqa: E402
from src.features import logmel                              # noqa: E402
from src.plots import first_look_figure, save_figure         # noqa: E402
from src.protocol import load_protocol                       # noqa: E402
from src.utils import load_config, set_seed, setup_logging   # noqa: E402

# A04 and A06 are among the older generators in the dataset. They leave more
# visible traces than the newer ones, which makes a clearer first picture -
# the point of this figure is to show the team what an artefact looks like,
# not to find the hardest possible example.
PREFERRED_ATTACKS = ["A04", "A06"]


def choose_utterances(df, config):
    """Return (real_utt_id, fake_utt_id), from config if set, else at random.

    Returns a third value saying whether they were chosen at random, so the
    caller knows whether to tell the user to freeze them in config.yaml.
    """
    real_id = config["inspect"]["real_utt_id"]
    fake_id = config["inspect"]["fake_utt_id"]

    # Validate whichever ids the config supplies, even if only one is set.
    # Checking only when BOTH are present would let a typo through to
    # `df[df["utt_id"] == real_id].iloc[0]` further down, which fails with a
    # bare IndexError that says nothing about config.yaml being the cause.
    known = set(df["utt_id"])
    for field, utt_id in (("real_utt_id", real_id), ("fake_utt_id", fake_id)):
        if utt_id and utt_id not in known:
            raise ValueError(
                "inspect." + field + " is " + utt_id + ", which is not in the "
                "train partition. Check the spelling in config.yaml, or set it "
                "back to null to have one picked at random.")

    if real_id and fake_id:
        return real_id, fake_id, False

    real_pool = df[df["label"] == 0]
    fake_pool = df[df["label"] == 1]

    preferred = fake_pool[fake_pool["system_id"].isin(PREFERRED_ATTACKS)]
    if len(preferred):
        fake_pool = preferred
    else:
        logging.warning("no %s utterances found, picking any spoof",
                        " or ".join(PREFERRED_ATTACKS))

    # set_seed() has already been called, so .sample() is reproducible.
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
    logging.info(
        "%s  %s  %.2fs -> %.2fs  spectrogram %s",
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
    parser.add_argument(
        "--config", default="config.yaml",
        help="path to the config file (used by the tests; you want the default)")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config["seed"])

    df = load_protocol("train", config)
    real_id, fake_id, was_random = choose_utterances(df, config)

    real_row = df[df["utt_id"] == real_id].iloc[0]
    fake_row = df[df["utt_id"] == fake_id].iloc[0]

    real = prepare(real_row, config)
    fake = prepare(fake_row, config)

    figure_path = Path(config["paths"]["figures"]) / "first_look.png"
    save_figure(first_look_figure(real, fake, config), figure_path)
    logging.info("wrote %s", figure_path)

    record = {
        "real": {k: real[k] for k in (
            "utt_id", "system_id", "key", "label",
            "duration_before_s", "duration_after_s", "spectrogram_shape")},
        "fake": {k: fake[k] for k in (
            "utt_id", "system_id", "key", "label",
            "duration_before_s", "duration_after_s", "spectrogram_shape")},
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
        print()
        print("These were picked at random. Paste them into config.yaml so the")
        print("figure is frozen and everyone reproduces the same picture:")
        print()
        print("inspect:")
        print("  real_utt_id: " + real["utt_id"])
        print("  fake_utt_id: " + fake["utt_id"])
    print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
