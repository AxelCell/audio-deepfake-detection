"""Load a waveform and get it into the shape the rest of the pipeline wants.

Always in this order: load_audio -> normalise -> fix_length.
"""

import logging

import numpy as np
import soundfile as sf


def load_audio(path, sample_rate):
    """Read an audio file as a 1-D float32 array."""
    wav, file_rate = sf.read(path, dtype="float32", always_2d=False)

    if wav.ndim > 1:
        logging.warning("%s has %d channels, averaging to mono", path, wav.shape[1])
        wav = wav.mean(axis=1)

    # ASVspoof LA is already 16 kHz mono, so this should never fire. Warn
    # rather than resample quietly, because it means a file isn't what we
    # think it is.
    if file_rate != sample_rate:
        logging.warning("%s is %d Hz, expected %d Hz - resampling",
                        path, file_rate, sample_rate)
        import librosa
        wav = librosa.resample(wav, orig_sr=file_rate, target_sr=sample_rate)

    return np.ascontiguousarray(wav, dtype=np.float32)


def normalise(wav, mode="peak"):
    """Divide by the largest absolute sample.

    Recording levels vary for reasons that have nothing to do with whether the
    speech is real or fake, so without this the model can read loudness as
    evidence.
    """
    if mode in (None, "none"):
        return wav
    if mode != "peak":
        raise ValueError(f"unknown normalise mode {mode!r}; expected 'peak' or 'none'")

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak == 0.0:
        # Digitally silent clips do turn up. Don't divide by zero.
        logging.warning("all-zero waveform, returning it unchanged")
        return wav

    return (wav / peak).astype(np.float32)


def fix_length(wav, n_samples, pad_mode="repeat"):
    """Return exactly n_samples samples.

    Too long: keep the first n_samples. Too short: either tile the clip until
    it's long enough, or pad with zeros.

    'repeat' is the default because zero-padding a 1 s clip out to 4 s hands
    the model three seconds of silence, and the amount of silence is itself a
    giveaway that has nothing to do with how the audio was made.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if len(wav) == n_samples:
        return wav
    if len(wav) > n_samples:
        return wav[:n_samples]

    if pad_mode == "repeat":
        if len(wav) == 0:
            raise ValueError("cannot repeat-pad an empty waveform")
        reps = int(np.ceil(n_samples / len(wav)))
        return np.tile(wav, reps)[:n_samples]

    if pad_mode == "zero":
        out = np.zeros(n_samples, dtype=wav.dtype)
        out[:len(wav)] = wav
        return out

    raise ValueError(f"unknown pad_mode {pad_mode!r}; expected 'repeat' or 'zero'")
