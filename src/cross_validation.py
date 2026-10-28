
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import mne

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
)

from src.epoching import extract_motor_imagery_epochs
from src.preprocessing import preprocess_eeg
from src.features_csp import extract_csp_features
from src.models import train_lda_classifier, predict_classes


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/BCI_2a")
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

N_SPLITS = 5
RANDOM_STATE = 42
N_CSP_COMPONENTS = 4

SFREQ = 250.0
LOWCUT = 0.5
HIGHCUT = 30.0
FILTER_ORDER = 3


# --------------------------------------------------
# SINGLE SUBJECT: 5-FOLD CROSS-VALIDATION
# --------------------------------------------------

def run_subject_cv(X, y, subject):
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        print(
            f"\n[{subject}] Fold {fold}/{N_SPLITS}",
            flush=True,
        )

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Fit CSP only on the current training fold.
        X_train_features, X_test_features, _ = (
            extract_csp_features(
                X_train,
                y_train,
                X_test,
                n_components=N_CSP_COMPONENTS,
            )
        )

        model = train_lda_classifier(
            X_train_features,
            y_train,
        )

        y_pred = predict_classes(
            model,
            X_test_features,
        )

        scores = {
            "subject": subject,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                y_pred,
            ),
            "cohen_kappa": cohen_kappa_score(
                y_test,
                y_pred,
            ),
        }

        fold_results.append(scores)

        print(
            f"Accuracy={scores['accuracy']:.4f} | "
            f"Macro-F1={scores['macro_f1']:.4f} | "
            f"Balanced Acc={scores['balanced_accuracy']:.4f} | "
            f"Kappa={scores['cohen_kappa']:.4f}",
            flush=True,
        )

    return pd.DataFrame(fold_results)


# --------------------------------------------------
# LOAD AND PROCESS ONE SUBJECT
# --------------------------------------------------

def process_subject(subject):
    file_path = DATA_DIR / f"{subject}T.gdf"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {file_path}"
        )

    print("\n" + "=" * 65, flush=True)
    print(f"PROCESSING SUBJECT {subject}", flush=True)
    print(f"Dataset: {file_path}", flush=True)
    print("=" * 65, flush=True)

    raw = mne.io.read_raw_gdf(
        str(file_path),
        preload=True,
        verbose=False,
    )

    X, y, epochs = extract_motor_imagery_epochs(
        raw,
        tmin=1.0,
        tmax=4.0,
    )

    if len(np.unique(y)) != 4:
        raise ValueError(
            f"{subject}: expected four classes, "
            f"found {np.unique(y)}"
        )

    class_counts = np.bincount(y, minlength=4)

    if np.any(class_counts < N_SPLITS):
        raise ValueError(
            f"{subject}: insufficient trials for "
            f"{N_SPLITS}-fold stratified CV. "
            f"Class counts: {class_counts}"
        )

    print(f"Epoch shape: {X.shape}", flush=True)
    print(f"Class counts: {class_counts}", flush=True)

    # Fixed filtering; CSP remains fitted inside each fold.
    X = preprocess_eeg(
        X,
        sfreq=SFREQ,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        filter_order=FILTER_ORDER,
    )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{subject}: filtered EEG contains NaN or infinity."
        )

    fold_results = run_subject_cv(X, y, subject)

    return fold_results


# --------------------------------------------------
# SUMMARIZE SUBJECT-LEVEL RESULTS
# --------------------------------------------------

def summarize_subjects(all_folds):
    metric_names = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "cohen_kappa",
    ]

    subject_rows = []

    for subject, group in all_folds.groupby("subject", sort=True):
        row = {
            "subject": subject,
            "completed_folds": len(group),
        }

        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)

        subject_rows.append(row)

    subject_summary = pd.DataFrame(subject_rows)

    print("\n" + "=" * 75)
    print("SUBJECT-WISE CROSS-VALIDATION SUMMARY")
    print("=" * 75)
    print(subject_summary.to_string(index=False, float_format="%.4f"))

    return subject_summary


def summarize_all_subjects(subject_summary):
    """
    Give each subject equal weight by averaging the
    subject-level cross-validation means.
    """
    metric_names = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "cohen_kappa",
    ]

    rows = []

    for metric in metric_names:
        column = f"{metric}_mean"
        values = subject_summary[column].dropna()

        rows.append(
            {
                "metric": metric,
                "n_subjects": len(values),
                "mean_across_subjects": values.mean(),
                "std_across_subjects": (
                    values.std(ddof=1)
                    if len(values) > 1
                    else np.nan
                ),
                "min_subject_mean": values.min(),
                "max_subject_mean": values.max(),
            }
        )

    overall = pd.DataFrame(rows)

    print("\n" + "=" * 75)
    print("OVERALL SUMMARY ACROSS SUBJECTS")
    print("=" * 75)
    print(overall.to_string(index=False, float_format="%.4f"))

    return overall


# --------------------------------------------------
# MAIN: PROCESS A01T TO A09T
# --------------------------------------------------

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    failed_subjects = []

    for index, subject in enumerate(SUBJECTS, start=1):
        print(
            f"\nSUBJECT PROGRESS: {index}/{len(SUBJECTS)}",
            flush=True,
        )

        try:
            fold_results = process_subject(subject)
            all_fold_results.append(fold_results)

            # Save progress after each successful subject.
            current_folds = pd.concat(
                all_fold_results,
                ignore_index=True,
            )

            current_folds.to_csv(
                TABLES_DIR / "all_subject_cv_folds.csv",
                index=False,
            )

            current_summary = summarize_subjects(current_folds)

            current_summary.to_csv(
                TABLES_DIR / "subject_cv_summary.csv",
                index=False,
            )

            current_overall = summarize_all_subjects(
                current_summary
            )

            current_overall.to_csv(
                TABLES_DIR / "overall_cv_summary.csv",
                index=False,
            )

            print(
                f"\n{subject} completed and results saved.",
                flush=True,
            )

        except Exception as exc:
            error_text = traceback.format_exc()

            failed_subjects.append(
                {
                    "subject": subject,
                    "error": str(exc),
                }
            )

            error_path = LOGS_DIR / f"{subject}_error.txt"
            error_path.write_text(
                error_text,
                encoding="utf-8",
            )

            print(
                f"\nERROR processing {subject}: {exc}",
                flush=True,
            )
            print(
                f"Full traceback saved to {error_path}",
                flush=True,
            )

    # Save final results for every subject that completed.
    if all_fold_results:
        all_folds = pd.concat(
            all_fold_results,
            ignore_index=True,
        )

        all_folds.to_csv(
            TABLES_DIR / "all_subject_cv_folds.csv",
            index=False,
        )

        subject_summary = summarize_subjects(all_folds)

        subject_summary.to_csv(
            TABLES_DIR / "subject_cv_summary.csv",
            index=False,
        )

        overall_summary = summarize_all_subjects(
            subject_summary
        )

        overall_summary.to_csv(
            TABLES_DIR / "overall_cv_summary.csv",
            index=False,
        )

    failure_report = pd.DataFrame(
        failed_subjects,
        columns=["subject", "error"],
    )

    failure_report.to_csv(
        TABLES_DIR / "failed_subjects.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("BATCH PROCESSING FINISHED")
    print("=" * 65)
    print(
        f"Successful subjects: "
        f"{len(all_fold_results)}/{len(SUBJECTS)}"
    )
    print(f"Failed subjects: {len(failed_subjects)}")
    print(f"Results directory: {TABLES_DIR}")

    if failed_subjects:
        print("\nCheck failed_subjects.csv and results/logs/.")

    print("\nThis experiment evaluates each subject separately.")
    print("It does not evaluate generalization to unseen subjects.")


if __name__ == "__main__":
    main()
