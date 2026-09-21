"""Drawing functions. Figures are built here; scripts decide what to draw.

Agg is selected before pyplot is imported so these work on a machine with no
display - the lab server, or anything over SSH.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from src.features import ms_to_samples  # noqa: E402


def plot_waveform(ax, wav, sample_rate, title):
    """Amplitude against time, with the x axis in seconds rather than samples."""
    seconds = np.arange(len(wav)) / float(sample_rate)
    ax.plot(seconds, wav, linewidth=0.5)
    ax.set_xlim(0, seconds[-1] if len(seconds) else 1)
    ax.set_ylim(-1.05, 1.05)           # waveforms are peak-normalised to +/-1
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(title, fontsize=10)


def plot_spectrogram(ax, spectrogram, config, title, vmin, vmax):
    """Log-mel spectrogram with axes in seconds and Hz, not frames and bins.

    vmin and vmax are passed in rather than computed here so that two panels
    can share one colour scale. Without that, each panel would auto-scale to
    its own range and the two pictures would not be comparable - differences
    in brightness would mean nothing.
    """
    import librosa.display

    sample_rate = config["audio"]["sample_rate"]
    settings = config["features"]["logmel"]

    image = librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=ms_to_samples(settings["hop_length_ms"], sample_rate),
        n_fft=settings["n_fft"],
        fmin=settings["f_min"],
        fmax=settings["f_max"],
        x_axis="time",
        y_axis="mel",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title, fontsize=10)
    return image


def first_look_figure(real, fake, config):
    """The phase 1 deliverable: a 2x2 grid comparing a real clip to a fake one.

    real and fake are dicts with keys: utt_id, system_id, wav, spectrogram.
    Layout is real on the left, fake on the right, waveform on top,
    spectrogram below.
    """
    sample_rate = config["audio"]["sample_rate"]

    # One colour scale across both spectrograms.
    vmin = float(min(real["spectrogram"].min(), fake["spectrogram"].min()))
    vmax = float(max(real["spectrogram"].max(), fake["spectrogram"].max()))

    # constrained_layout, not tight_layout: without it the waveform's
    # 'time (s)' label collides with the spectrogram title directly below it.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    plot_waveform(axes[0, 0], real["wav"], sample_rate,
                  "REAL (bona fide)  -  " + real["utt_id"])
    plot_waveform(axes[0, 1], fake["wav"], sample_rate,
                  "FAKE (spoof, " + fake["system_id"] + ")  -  " + fake["utt_id"])

    plot_spectrogram(axes[1, 0], real["spectrogram"], config,
                     "log-mel  -  " + real["utt_id"], vmin, vmax)
    image = plot_spectrogram(
        axes[1, 1], fake["spectrogram"], config,
        "log-mel  -  " + fake["utt_id"] + "  (" + fake["system_id"] + ")",
        vmin, vmax)

    fig.colorbar(image, ax=axes[1, :], label="log power", pad=0.02)
    fig.suptitle(
        "ASVspoof 2019 LA - real vs fake, same colour scale", fontsize=12)
    return fig


def save_figure(fig, path):
    """Save a PNG that is byte-identical when the script is re-run.

    matplotlib writes a 'Software: Matplotlib vX.Y.Z' tag into PNG metadata by
    default. That alone would make two runs differ, which would break the
    phase 1 requirement that running inspect.py twice produces an identical
    file. Setting it to None leaves it out.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" here - it fights with constrained_layout and can
    # crop unpredictably, which would also threaten byte-for-byte repeatability.
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)
