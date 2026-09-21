"""Waveform -> log-mel spectrogram.

Phase 1 only has log-mel. LFCC and CQCC come in phase 2.
"""

import numpy as np


def ms_to_samples(milliseconds, sample_rate):
    """Milliseconds to a whole number of samples.

    Window and hop are stored in config.yaml as milliseconds because that is
    the quantity that actually means something - the vocal tract can't reshape
    itself much faster than 25 ms, so the signal is roughly steady inside one
    window. "400 samples" only says the same thing while the audio is 16 kHz.
    """
    return int(round(milliseconds / 1000.0 * sample_rate))


def logmel(wav, config):
    """Log-mel spectrogram, shape (n_mels, n_frames), float32.

    With the defaults (4 s at 16 kHz, 10 ms hop, 80 mels) that is (80, 401):
    64000 / 160 = 400 frames, plus one because librosa pads the edges.
    """
    # librosa pulls in numba and is slow to import, so keep it out of the
    # module level - protocol.py and the tests don't need it.
    import librosa

    sample_rate = config["audio"]["sample_rate"]
    settings = config["features"]["logmel"]

    power = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=sample_rate,
        n_fft=settings["n_fft"],
        win_length=ms_to_samples(settings["win_length_ms"], sample_rate),
        hop_length=ms_to_samples(settings["hop_length_ms"], sample_rate),
        n_mels=settings["n_mels"],
        fmin=settings["f_min"],
        fmax=settings["f_max"],
        power=2.0,
    )

    # Plain log, not librosa.power_to_db. power_to_db clips relative to the
    # loudest point of each clip, which makes the result depend on that one
    # recording. A plain log with the offset from config.yaml keeps every clip
    # on the same scale.
    return np.log(power + settings["log_offset"]).astype(np.float32)


def expected_n_frames(n_samples, config):
    """How many frames logmel() will return. Used by the tests."""
    hop = ms_to_samples(
        config["features"]["logmel"]["hop_length_ms"],
        config["audio"]["sample_rate"],
    )
    return 1 + n_samples // hop
