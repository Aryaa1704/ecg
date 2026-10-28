
import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    fs: float = 250.0,
    order: int = 3,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Expected data shape:
        (n_trials, n_channels, n_samples)
    """
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(
            "Expected data shape: (n_trials, n_channels, n_samples)."
        )

    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if order < 1:
        raise ValueError("Filter order must be at least 1.")

    nyquist = fs / 2.0

    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            f"Invalid frequency range: {lowcut}-{highcut} Hz "
            f"for sampling frequency {fs} Hz."
        )

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, data, axis=-1)

    return filtered


def apply_baseline_correction(
    data: np.ndarray,
    baseline_start: float,
    baseline_end: float,
    sfreq: float = 250.0,
) -> np.ndarray:
    """
    Subtract the mean of a specified baseline interval
    from each trial and EEG channel.

    Times are measured relative to the beginning of each epoch.
    """
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(
            "Expected data shape: (n_trials, n_channels, n_samples)."
        )

    if sfreq <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if baseline_start < 0 or baseline_start >= baseline_end:
        raise ValueError(
            "Baseline start must be non-negative and smaller "
            "than baseline end."
        )

    start_sample = int(round(baseline_start * sfreq))
    end_sample = int(round(baseline_end * sfreq))

    if end_sample > data.shape[-1]:
        raise ValueError("Baseline interval exceeds epoch length.")

    if start_sample >= data.shape[-1] or end_sample <= start_sample:
        raise ValueError("Invalid baseline sample interval.")

    baseline = np.mean(
        data[:, :, start_sample:end_sample],
        axis=-1,
        keepdims=True,
    )

    corrected = data - baseline

    return corrected


def preprocess_eeg(
    X: np.ndarray,
    sfreq: float = 250.0,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    filter_order: int = 3,
) -> np.ndarray:
    """
    Apply the basic EEG preprocessing pipeline.

    Input:
        X: EEG epochs shaped
           (n_trials, n_channels, n_samples)

    Pipeline:
        EEG epochs -> 0.5-30 Hz Butterworth band-pass filter

    Baseline correction is kept separate because its validity
    depends on the experimental epoch definition.
    """
    X_processed = butter_bandpass_filter(
        X,
        lowcut=lowcut,
        highcut=highcut,
        fs=sfreq,
        order=filter_order,
    )

    return X_processed


def print_preprocessing_audit(
    X_before: np.ndarray,
    X_after: np.ndarray,
    sfreq: float = 250.0,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    filter_order: int = 3,
) -> None:
    """Print preprocessing validation information."""

    print("\n" + "=" * 60)
    print("PREPROCESSING AUDIT")
    print("=" * 60)

    print(f"Input shape       : {X_before.shape}")
    print(f"Output shape      : {X_after.shape}")

    print(f"Input mean        : {np.mean(X_before):.6e}")
    print(f"Output mean       : {np.mean(X_after):.6e}")

    print(f"Input std         : {np.std(X_before):.6e}")
    print(f"Output std        : {np.std(X_after):.6e}")

    print("\nFilter configuration:")
    print("Type              : Butterworth band-pass")
    print(f"Low cutoff        : {lowcut} Hz")
    print(f"High cutoff       : {highcut} Hz")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Filter order      : {filter_order}")

    print("=" * 60)


if __name__ == "__main__":
    from pathlib import Path
    import mne

    # Import using the package path so this works with:
    # python -m src.preprocessing
    from src.epoching import extract_motor_imagery_epochs

    DATA_PATH = Path("data/BCI_2a/A01T.gdf")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    print(f"Loading dataset: {DATA_PATH}")

    raw = mne.io.read_raw_gdf(
        str(DATA_PATH),
        preload=True,
        verbose=False,
    )

    # Cue-relative 1-4 seconds correspond to approximately
    # trial-relative 3-6 seconds in BCI Competition IV 2a.
    X, y, epochs = extract_motor_imagery_epochs(
        raw,
        tmin=1.0,
        tmax=4.0,
    )

    print(f"\nExtracted EEG data: {X.shape}")
    print(f"Labels shape      : {y.shape}")
    print(f"Number of classes : {len(np.unique(y))}")

    # Apply the 0.5-30 Hz, third-order band-pass filter.
    X_processed = preprocess_eeg(
        X,
        sfreq=250.0,
        lowcut=0.5,
        highcut=30.0,
        filter_order=3,
    )

    print_preprocessing_audit(
        X,
        X_processed,
        sfreq=250.0,
        lowcut=0.5,
        highcut=30.0,
        filter_order=3,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Processed EEG shape: {X_processed.shape}")
    print(f"Labels shape       : {y.shape}")
    print("Next step: CSP feature extraction.")
