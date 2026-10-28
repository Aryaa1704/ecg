
import numpy as np
from mne.decoding import CSP


def extract_csp_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 4,
):
    """
    Fit CSP using training data and transform training/test epochs.

    Inputs:
        X_train: (n_trials, n_channels, n_samples)
        y_train: (n_trials,)
        X_test:  (n_trials, n_channels, n_samples)

    Returns:
        X_train_features
        X_test_features
        csp: fitted CSP transformer
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "X_train and X_test must have shape "
            "(n_trials, n_channels, n_samples)."
        )

    if y_train.ndim != 1 or len(y_train) != len(X_train):
        raise ValueError(
            "y_train must contain one label per training trial."
        )

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Training and test data must have matching "
            "channel and sample dimensions."
        )

    if len(np.unique(y_train)) < 2:
        raise ValueError("CSP requires at least two training classes.")

    if not 1 <= n_components <= X_train.shape[1]:
        raise ValueError(
            "n_components must be between 1 and the number of channels."
        )

    # Fit CSP ONLY on training data to prevent data leakage.
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
        component_order="mutual_info",
    )

    X_train_features = csp.fit_transform(X_train, y_train)
    X_test_features = csp.transform(X_test)

    return X_train_features, X_test_features, csp


def print_csp_audit(
    X_train_features: np.ndarray,
    X_test_features: np.ndarray,
) -> None:
    """Print validation information for CSP features."""

    print("\n" + "=" * 60)
    print("CSP FEATURE EXTRACTION AUDIT")
    print("=" * 60)

    print(f"Training feature shape: {X_train_features.shape}")
    print(f"Test feature shape    : {X_test_features.shape}")

    print(
        "Training features finite:",
        bool(np.isfinite(X_train_features).all()),
    )
    print(
        "Test features finite    :",
        bool(np.isfinite(X_test_features).all()),
    )

    print("=" * 60)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.epoching import extract_motor_imagery_epochs
    from src.preprocessing import preprocess_eeg
    import mne
    from pathlib import Path

    data_path = Path("data/BCI_2a/A01T.gdf")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    print(f"Loading dataset: {data_path}")

    raw = mne.io.read_raw_gdf(
        str(data_path),
        preload=True,
        verbose=False,
    )

    X, y, epochs = extract_motor_imagery_epochs(
        raw,
        tmin=1.0,
        tmax=4.0,
    )

    X = preprocess_eeg(
        X,
        sfreq=250.0,
        lowcut=0.5,
        highcut=30.0,
        filter_order=3,
    )

    # Stratified split preserves class proportions.
    indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]

    X_train_features, X_test_features, csp = (
        extract_csp_features(
            X_train,
            y_train,
            X_test,
            n_components=4,
        )
    )

    print_csp_audit(
        X_train_features,
        X_test_features,
    )

    print("\nCSP feature extraction completed successfully.")
    print("Classes:", np.unique(y_train))
