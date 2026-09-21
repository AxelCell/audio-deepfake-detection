"""Tests that run on a fresh clone, with no dataset and no torch.

Everything builds its own miniature dataset in a temp folder, so the whole
file runs in a few seconds and never touches data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audio import fix_length, load_audio, normalise
from src.features import expected_n_frames, logmel, ms_to_samples
from src.protocol import COLUMNS, load_protocol, summarise
from src.utils import load_config

# Three real and three fake across two attacks, including the always-'-'
# third column that load_protocol throws away.
FAKE_PROTOCOL = """\
LA_0001 LA_T_0000001 - - bonafide
LA_0001 LA_T_0000002 - A01 spoof
LA_0002 LA_T_0000003 - - bonafide
LA_0002 LA_T_0000004 - A02 spoof
LA_0003 LA_T_0000005 - - bonafide
LA_0003 LA_T_0000006 - A01 spoof
"""


@pytest.fixture
def mini_dataset(tmp_path):
    """Build data/LA/... in a temp folder and return a config dict for it."""
    data_root = tmp_path / "data" / "LA"
    protocols = data_root / "ASVspoof2019_LA_cm_protocols"
    flac_dir = data_root / "ASVspoof2019_LA_train" / "flac"
    protocols.mkdir(parents=True)
    flac_dir.mkdir(parents=True)

    (protocols / "ASVspoof2019.LA.cm.train.trn.txt").write_text(FAKE_PROTOCOL)

    # Empty files. load_protocol must not care - checking the audio is
    # verify.py's job.
    for i in range(1, 7):
        (flac_dir / f"LA_T_000000{i}.flac").touch()

    return {"paths": {"data_root": str(data_root), "protocols": str(protocols)}}


@pytest.fixture(scope="module")
def config():
    return load_config()


# --- protocol.py ---

def test_row_count_and_columns(mini_dataset):
    df = load_protocol("train", mini_dataset)
    assert len(df) == 6
    assert list(df.columns) == COLUMNS


def test_label_is_zero_for_real_and_one_for_fake(mini_dataset):
    df = load_protocol("train", mini_dataset)
    assert set(df.loc[df["key"] == "bonafide", "label"]) == {0}
    assert set(df.loc[df["key"] == "spoof", "label"]) == {1}
    assert df["label"].sum() == 3
    assert pd.api.types.is_integer_dtype(df["label"])


def test_system_id_is_dash_on_every_real_row(mini_dataset):
    df = load_protocol("train", mini_dataset)
    assert (df.loc[df["label"] == 0, "system_id"] == "-").all()
    assert set(df.loc[df["label"] == 1, "system_id"]) == {"A01", "A02"}


def test_path_points_at_the_flac_file(mini_dataset):
    df = load_protocol("train", mini_dataset)
    expected = (Path(mini_dataset["paths"]["data_root"])
                / "ASVspoof2019_LA_train" / "flac" / "LA_T_0000001.flac")
    assert Path(df.loc[0, "path"]) == expected
    assert all(Path(p).exists() for p in df["path"])


def test_missing_protocol_file_names_the_path(tmp_path):
    config = {"paths": {"data_root": str(tmp_path), "protocols": str(tmp_path)}}
    with pytest.raises(FileNotFoundError) as excinfo:
        load_protocol("train", config)
    assert "ASVspoof2019.LA.cm.train.trn.txt" in str(excinfo.value)


def test_unknown_partition_raises(mini_dataset):
    with pytest.raises(ValueError) as excinfo:
        load_protocol("validation", mini_dataset)
    assert "validation" in str(excinfo.value)


def test_summarise_counts(mini_dataset):
    row = summarise(load_protocol("train", mini_dataset)).iloc[0]
    assert row["partition"] == "train"
    assert row["real"] == 3
    assert row["fake"] == 3
    assert row["total"] == 6
    assert row["n_attacks"] == 2
    assert row["attacks"] == "A01,A02"


# --- audio.py ---

def test_fix_length_trims_when_too_long():
    wav = np.arange(500, dtype=np.float32)
    out = fix_length(wav, 250)
    assert len(out) == 250
    assert np.array_equal(out, wav[:250])        # takes the front, not a random crop


def test_fix_length_leaves_exact_input_alone():
    wav = np.arange(250, dtype=np.float32)
    assert np.array_equal(fix_length(wav, 250), wav)


def test_fix_length_repeat_pads_100_to_250():
    wav = np.arange(100, dtype=np.float32)
    out = fix_length(wav, 250, pad_mode="repeat")
    assert len(out) == 250
    assert np.array_equal(out[:100], wav)
    assert np.array_equal(out[100:200], wav)
    assert np.array_equal(out[200:], wav[:50])   # third copy, trimmed


def test_fix_length_zero_pads():
    out = fix_length(np.ones(100, dtype=np.float32), 250, pad_mode="zero")
    assert len(out) == 250
    assert out[:100].all() and not out[100:].any()


def test_fix_length_rejects_bad_pad_mode():
    with pytest.raises(ValueError):
        fix_length(np.ones(10, dtype=np.float32), 20, pad_mode="mirror")


def test_normalise_scales_peak_to_one():
    out = normalise(np.array([0.0, 0.25, -0.5], dtype=np.float32), "peak")
    assert np.isclose(np.max(np.abs(out)), 1.0)
    assert np.isclose(out[1], 0.5)               # ratios preserved


def test_normalise_leaves_all_zeros_alone():
    wav = np.zeros(100, dtype=np.float32)
    out = normalise(wav, "peak")
    assert np.array_equal(out, wav)              # no divide by zero
    assert not np.isnan(out).any()


def test_load_audio_returns_float32_mono(tmp_path):
    import soundfile as sf
    path = tmp_path / "tone.flac"
    sr = 16000
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    sf.write(path, tone, sr)

    wav = load_audio(str(path), sr)
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert len(wav) == sr


# --- features.py ---

def test_ms_to_samples():
    assert ms_to_samples(25, 16000) == 400
    assert ms_to_samples(10, 16000) == 160
    # same milliseconds, different rate, different sample count - which is
    # the reason the config stores milliseconds
    assert ms_to_samples(25, 8000) == 200
    assert ms_to_samples(25, 22050) == 551


def test_logmel_shape_matches_prediction(config):
    sr = config["audio"]["sample_rate"]
    n = int(sr * config["audio"]["duration_seconds"])
    wav = (0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype(np.float32)

    S = logmel(wav, config)
    assert S.shape == (config["features"]["logmel"]["n_mels"],
                       expected_n_frames(n, config))
    assert S.shape == (80, 401)          # the number predicted in BUILD.md
    assert S.dtype == np.float32


def test_logmel_is_finite_on_silence(config):
    """log(0) is -inf. The log_offset exists to stop that."""
    sr = config["audio"]["sample_rate"]
    S = logmel(np.zeros(sr, dtype=np.float32), config)
    assert np.isfinite(S).all()
    assert np.allclose(S, np.log(config["features"]["logmel"]["log_offset"]))


def test_logmel_is_deterministic(config):
    sr = config["audio"]["sample_rate"]
    wav = (0.4 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    assert np.array_equal(logmel(wav, config), logmel(wav, config))
