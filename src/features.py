"""Turn a waveform into the time-frequency picture the model sees.

Phase 1 has one front-end: log-mel. LFCC and CQCC arrive in phase 2, and
experiment E1 then compares all three.

Note for phase 2: the official ASVspoof LFCC and CQCC implementations are
MATLAB. The versions we write will be Python re-implementations, not the
originals, and must say so in a comment.
"""

import numpy as np


def ms_to_samples(milliseconds, sample_rate):
    """Convert a duration in milliseconds to a whole number of samples.

    Window and hop live in config.yaml as milliseconds rather than samples on
    purpose. '25 ms' is a claim about speech - the vocal tract cannot reshape
    itself much faster than that, so the signal is roughly stable inside one
    window. '400 samples' only means the same thing if the audio happens to be
    16 kHz. Storing the physical quantity and deriving the sample count keeps
    the meaning fixed if the sample rate ever changes, and matches how every
    paper states its settings.
    """
    return int(round(milliseconds / 1000.0 * sample_rate))


def logmel(wav, config):
    """Log-mel spectrogram of a waveform.

    Returns an array of shape (n_mels, n_frames), float32.

    For the project default - 4.0 s at 16 kHz, hop 10 ms, 80 mel bands - that
    is (80, 401): 64000 samples / 160 per hop = 400, plus 1 because librosa
    centres the frames and pads the edges.
    """
    # librosa is imported here rather than at module level because it is slow
    # to import (numba), and protocol.py and the tests must stay light.
    import librosa

    sample_rate = config["audio"]["sample_rate"]
    settings = config["features"]["logmel"]

    win_length = ms_to_samples(settings["win_length_ms"], sample_rate)
    hop_length = ms_to_samples(settings["hop_length_ms"], sample_rate)

    power = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=sample_rate,
        n_fft=settings["n_fft"],
        win_length=win_length,
        hop_length=hop_length,
        n_mels=settings["n_mels"],
        fmin=settings["f_min"],
        fmax=settings["f_max"],
        power=2.0,
    )

    # A plain log, not librosa.power_to_db. power_to_db applies 10*log10 and
    # then clips relative to the maximum value in each clip, which makes the
    # result depend on the loudest part of that particular recording. A plain
    # log with an explicit offset from config.yaml is simpler, keeps every
    # clip on the same absolute scale, and leaves the offset as something we
    # can change and record rather than something hidden inside a library.
    return np.log(power + settings["log_offset"]).astype(np.float32)


def expected_n_frames(n_samples, config):
    """How many frames logmel() will return for a clip of n_samples.

    Used by the tests and by inspect.py so the shape is something we predict
    and then check, rather than something we discover.
    """
    hop = ms_to_samples(
        config["features"]["logmel"]["hop_length_ms"],
        config["audio"]["sample_rate"],
    )
    return 1 + n_samples // hop
