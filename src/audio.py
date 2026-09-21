"""Load waveforms and put them in the shape the rest of the pipeline expects.

Three steps, always in this order:

    load_audio  -> read the file as float32 mono at the right sample rate
    normalise   -> put the amplitude on a predictable scale
    fix_length  -> make every clip exactly the same number of samples
"""

import logging

import numpy as np
import soundfile as sf


def load_audio(path, sample_rate):
    """Read an audio file as a 1-D float32 array at `sample_rate`.

    ASVspoof LA is already 16 kHz mono, so the resample branch should never
    fire. It logs a warning when it does, because that means a file is not
    what we think it is - which is worth knowing about rather than silently
    fixing.
    """
    wav, file_rate = sf.read(path, dtype="float32", always_2d=False)

    if wav.ndim > 1:
        # (samples, channels) -> mono. Should not happen with this dataset.
        logging.warning("%s has %d channels, averaging to mono", path, wav.shape[1])
        wav = wav.mean(axis=1)

    if file_rate != sample_rate:
        logging.warning(
            "%s is %d Hz, expected %d Hz - resampling. Every file in "
            "ASVspoof LA should already be 16 kHz, so check this file.",
            path, file_rate, sample_rate,
        )
        # Imported here rather than at module level: librosa is slow to import
        # and this branch is not supposed to run.
        import librosa
        wav = librosa.resample(wav, orig_sr=file_rate, target_sr=sample_rate)

    return np.ascontiguousarray(wav, dtype=np.float32)


def normalise(wav, mode="peak"):
    """Scale the waveform. 'peak' divides by the largest absolute sample.

    Recording levels vary between utterances for reasons that have nothing to
    do with whether the speech is real or fake. Normalising stops the model
    reading loudness as evidence.

    An all-zero (digitally silent) clip is returned unchanged rather than
    divided by zero. Silent files do occur in speech corpora.
    """
    if mode in (None, "none"):
        return wav

    if mode != "peak":
        raise ValueError(f"unknown normalise mode {mode!r}; expected 'peak' or 'none'")

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak == 0.0:
        logging.warning("all-zero waveform, returning it unchanged")
        return wav

    return (wav / peak).astype(np.float32)


def fix_length(wav, n_samples, pad_mode="repeat"):
    """Return exactly `n_samples` samples - always, whatever came in.

    Too long  -> keep the first n_samples (deterministic, so figures repeat)
    Too short -> 'repeat' tiles the clip until it is long enough, then trims;
                 'zero' pads the end with silence.

    'repeat' is the ASVspoof convention and is the default in config.yaml.
    Zero-padding a 1-second clip out to 4 seconds hands the model three
    seconds of digital silence, which is itself a strong giveaway that has
    nothing to do with how the audio was produced.
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
