
from pathlib import Path
import traceback

import mne
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)

from src.epoching import extract_motor_imagery_epochs
from src.preprocessing import preprocess_eeg
from src.features_csp import extract_csp_features
from src.models import train_lda_classifier, predict_classes


# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("data/BCI_2a")
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
CSP_COMPONENTS = [4, 6, 8]

N_SPLITS = 5
RANDOM_STATE = 42

METRICS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "cohen_kappa",
]


# ==================================================
# LOAD AND PREPROCESS SUBJECT
# ==================================================

def prepare_subject(subject):
    """Load one subject and prepare its motor-imagery epochs."""

    file_path = DATA_DIR / f"{subject}T.gdf"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    print(f"Reading {file_path}", flush=True)

    raw = mne.io.read_raw_gdf(
        str(file_path),
        preload=True,
        verbose="ERROR",
    )

    # Function returns X, y, and the MNE Epochs object.
    X, y, epochs = extract_motor_imagery_epochs(
        raw,
        tmin=1.0,
        tmax=4.0,
    )

    # Apply the project's existing preprocessing function.
    X = preprocess_eeg(X)
    y = np.asarray(y).ravel()

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG data, got shape {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"Epoch/label mismatch: {len(X)} epochs, {len(y)} labels"
        )

    classes, counts = np.unique(y, return_counts=True)

    if len(classes) != 4:
        raise ValueError(
            f"Expected 4 classes, found {classes.tolist()}"
        )

    if np.min(counts) < N_SPLITS:
        raise ValueError(
            f"Insufficient trials per class for {N_SPLITS}-fold CV: "
            f"{dict(zip(classes.tolist(), counts.tolist()))}"
        )

    print(
        f"{subject}: X={X.shape}, classes={classes.tolist()}",
        flush=True,
    )

    return X, y


# ==================================================
# CROSS-VALIDATION FOR ONE CSP CONFIGURATION
# ==================================================

def evaluate_components(subject, X, y, n_components):
    """Evaluate CSP + LDA using stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit CSP only on the training fold.
        X_train_features, X_test_features, _ = (
            extract_csp_features(
                X_train,
                y_train,
                X_test,
                n_components=n_components,
            )
        )

        model = train_lda_classifier(
            X_train_features,
            y_train,
        )

        predictions = predict_classes(
            model,
            X_test_features,
        )

        scores = {
            "subject": subject,
            "n_components": n_components,
            "fold": fold,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                predictions,
            ),
            "cohen_kappa": cohen_kappa_score(
                y_test,
                predictions,
            ),
        }

        fold_results.append(scores)

        print(
            f"  {subject} | CSP={n_components} | "
            f"Fold {fold}/{N_SPLITS} | "
            f"Accuracy={scores['accuracy']:.4f}",
            flush=True,
        )

    return fold_results


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(all_folds, failures):
    """Save intermediate and final results to CSV files."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    folds_path = (
        TABLES_DIR / "csp_component_comparison_folds.csv"
    )
    subject_path = (
        TABLES_DIR / "csp_component_comparison_by_subject.csv"
    )
    summary_path = (
        TABLES_DIR / "csp_component_comparison_summary.csv"
    )
    failures_path = (
        TABLES_DIR / "csp_component_comparison_failures.csv"
    )

    folds_df = pd.DataFrame(all_folds)
    folds_df.to_csv(folds_path, index=False)

    pd.DataFrame(
        failures,
        columns=["subject", "n_components", "error"],
    ).to_csv(failures_path, index=False)

    if folds_df.empty:
        return

    # Mean and standard deviation across the five folds
    # for every subject and component configuration.
    grouped = folds_df.groupby(
        ["subject", "n_components"]
    )[METRICS].agg(["mean", "std"])

    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat in grouped.columns
    ]

    grouped.reset_index().to_csv(
        subject_path,
        index=False,
    )

    # First calculate each subject's mean across folds.
    subject_means = (
        folds_df.groupby(["subject", "n_components"])[METRICS]
        .mean()
        .reset_index()
    )

    # Then give each subject equal weight in the overall summary.
    summary_rows = []

    for n_components, group in subject_means.groupby(
        "n_components"
    ):
        row = {
            "n_components": int(n_components),
            "n_subjects": int(group["subject"].nunique()),
        }

        for metric in METRICS:
            row[f"{metric}_mean_across_subjects"] = (
                group[metric].mean()
            )
            row[f"{metric}_std_across_subjects"] = (
                group[metric].std(ddof=1)
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "n_components"
    )

    summary_df.to_csv(summary_path, index=False)


# ==================================================
# MAIN
# ==================================================

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_folds = []
    failures = []

    print("=" * 65)
    print("CSP COMPONENT COMPARISON: 4 vs 6 vs 8")
    print(f"Subjects: {len(SUBJECTS)}")
    print(f"Folds per configuration: {N_SPLITS}")
    print(f"Configurations: {CSP_COMPONENTS}")
    print("=" * 65, flush=True)

    for subject in SUBJECTS:
        print(f"\n{'=' * 55}", flush=True)
        print(f"Processing {subject}", flush=True)

        try:
            X, y = prepare_subject(subject)

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            failures.append({
                "subject": subject,
                "n_components": "all",
                "error": error_message,
            })

            log_path = (
                LOGS_DIR
                / f"{subject}_component_comparison_error.txt"
            )
            log_path.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )

            print(
                f"FAILED {subject} during data preparation: "
                f"{error_message}",
                flush=True,
            )

            save_results(all_folds, failures)
            continue

        for n_components in CSP_COMPONENTS:
            print(
                f"\nTesting {subject} with "
                f"{n_components} CSP components",
                flush=True,
            )

            try:
                results = evaluate_components(
                    subject,
                    X,
                    y,
                    n_components,
                )

                all_folds.extend(results)

            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"

                failures.append({
                    "subject": subject,
                    "n_components": n_components,
                    "error": error_message,
                })

                log_path = (
                    LOGS_DIR
                    / f"{subject}_csp{n_components}_error.txt"
                )
                log_path.write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )

                print(
                    f"FAILED {subject}, CSP={n_components}: "
                    f"{error_message}",
                    flush=True,
                )

            # Save after each configuration so progress is retained.
            save_results(all_folds, failures)

    print("\n" + "=" * 65)
    print("COMPARISON FINISHED")
    print("=" * 65)

    if not all_folds:
        print(
            "No successful results were generated. "
            "Check results/logs for error details."
        )
        return

    folds_df = pd.DataFrame(all_folds)

    print(
        f"Successful fold evaluations: {len(folds_df)}"
    )
    print(
        f"Failed configurations/preparations: {len(failures)}"
    )

    summary_path = (
        TABLES_DIR / "csp_component_comparison_summary.csv"
    )

    summary_df = pd.read_csv(summary_path)

    display_columns = [
        "n_components",
        "n_subjects",
        "accuracy_mean_across_subjects",
        "macro_f1_mean_across_subjects",
        "balanced_accuracy_mean_across_subjects",
        "cohen_kappa_mean_across_subjects",
    ]

    print("\nOverall summary (scores are fractions):")
    print(
        summary_df[display_columns].to_string(index=False)
    )

    print("\nSaved files:")
    print(
        TABLES_DIR / "csp_component_comparison_folds.csv"
    )
    print(
        TABLES_DIR / "csp_component_comparison_by_subject.csv"
    )
    print(summary_path)
    print(
        TABLES_DIR / "csp_component_comparison_failures.csv"
    )


if __name__ == "__main__":
    main()
