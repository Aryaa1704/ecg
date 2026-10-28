
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from sklearn.model_selection import train_test_split

from src.epoching import extract_motor_imagery_epochs
from src.preprocessing import preprocess_eeg
from src.features_csp import extract_csp_features
from src.models import train_lda_classifier, predict_classes
from src.evaluation import evaluate_classifier, plot_confusion_matrix


DATA_PATH = Path("data/BCI_2a/A01T.gdf")
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_CSP_COMPONENTS = 4


def main():
    print("=" * 65)
    print("EEG MOTOR IMAGERY CLASSIFICATION PIPELINE")
    print("Experiment: 4-class CSP + LDA")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print("\n[1/6] Loading EEG dataset...")

    raw = mne.io.read_raw_gdf(
        str(DATA_PATH),
        preload=True,
        verbose=False,
    )

    # 2. Extract four-class motor imagery epochs
    print("\n[2/6] Extracting motor imagery epochs...")

    X, y, epochs = extract_motor_imagery_epochs(
        raw,
        tmin=1.0,
        tmax=4.0,
    )

    print(f"EEG data shape: {X.shape}")
    print(f"Labels shape  : {y.shape}")
    print(f"Class labels  : {np.unique(y)}")

    # 3. Filter EEG
    print("\n[3/6] Applying band-pass filter...")

    X = preprocess_eeg(
        X,
        sfreq=250.0,
        lowcut=0.5,
        highcut=30.0,
        filter_order=3,
    )

    if not np.isfinite(X).all():
        raise ValueError("Filtered EEG contains NaN or infinity.")

    # 4. Stratified trial-level split
    print("\n[4/6] Splitting trials into train and test sets...")

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Training trials: {len(y_train)}")
    print(f"Test trials    : {len(y_test)}")

    # 5. Fit CSP on training data only
    print("\n[5/6] Extracting CSP features...")

    X_train_features, X_test_features, csp = (
        extract_csp_features(
            X_train,
            y_train,
            X_test,
            n_components=N_CSP_COMPONENTS,
        )
    )

    print(f"Training CSP features: {X_train_features.shape}")
    print(f"Test CSP features    : {X_test_features.shape}")

    # Train LDA on training features only
    print("Training LDA classifier...")

    model = train_lda_classifier(
        X_train_features,
        y_train,
    )

    y_pred = predict_classes(
        model,
        X_test_features,
    )

    # 6. Evaluate and save results
    print("\n[6/6] Evaluating predictions...")

    metrics = evaluate_classifier(y_test, y_pred)

    # Save scalar metrics
    metrics_table = pd.DataFrame(
        [
            {
                "experiment": "4-class CSP + LDA",
                "dataset": "BCI Competition IV 2a",
                "subject": "A01",
                "session": "A01T internal stratified split",
                "train_trials": len(y_train),
                "test_trials": len(y_test),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "csp_components": N_CSP_COMPONENTS,
                "filter_low_hz": 0.5,
                "filter_high_hz": 30.0,
                "filter_order": 3,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "cohen_kappa": metrics["cohen_kappa"],
            }
        ]
    )

    metrics_path = TABLES_DIR / "csp_lda_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)

    # Save individual test predictions
    predictions_table = pd.DataFrame(
        {
            "trial_index": test_idx,
            "true_label": y_test,
            "predicted_label": y_pred,
        }
    )

    predictions_path = TABLES_DIR / "csp_lda_predictions.csv"
    predictions_table.to_csv(predictions_path, index=False)

    # Save confusion matrix as CSV
    confusion_path = TABLES_DIR / "csp_lda_confusion_matrix.csv"
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual_Left", "Actual_Right", "Actual_Feet", "Actual_Tongue"],
        columns=["Pred_Left", "Pred_Right", "Pred_Feet", "Pred_Tongue"],
    ).to_csv(confusion_path)

    # Save confusion matrix figure
    figure_path = FIGURES_DIR / "csp_lda_confusion_matrix.png"

    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path=figure_path,
        show=False,
    )

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED")
    print("=" * 65)
    print(f"Metrics CSV      : {metrics_path}")
    print(f"Predictions CSV  : {predictions_path}")
    print(f"Confusion CSV    : {confusion_path}")
    print(f"Confusion figure : {figure_path}")
    print("\nNote: This is an internal A01T split result,")
    print("not an independent A01E session evaluation.")


if __name__ == "__main__":
    main()
