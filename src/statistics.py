
import numpy as np
import pandas as pd


def summarize_cv_results(
    fold_scores,
    metric_name="accuracy",
):
    """
    Summarize cross-validation scores across folds.

    Parameters
    ----------
    fold_scores : array-like
        One metric score per validation fold.
    metric_name : str
        Name of the metric being summarized.

    Returns
    -------
    dict
        Number of folds, mean, standard deviation,
        and 95% confidence interval for the mean.
    """
    scores = np.asarray(fold_scores, dtype=float)

    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError(
            "Provide at least two cross-validation fold scores."
        )

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinity.")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=1))
    standard_error = std_score / np.sqrt(len(scores))

    # Normal approximation; suitable as a descriptive interval
    # for a quick report, not a definitive CV uncertainty estimate.
    ci_lower = mean_score - 1.96 * standard_error
    ci_upper = mean_score + 1.96 * standard_error

    result = {
        "metric": metric_name,
        "n_folds": len(scores),
        "mean": mean_score,
        "std": std_score,
        "ci95_lower": max(0.0, float(ci_lower)),
        "ci95_upper": min(1.0, float(ci_upper)),
    }

    print("\n" + "=" * 55)
    print(f"CROSS-VALIDATION SUMMARY: {metric_name}")
    print("=" * 55)
    print(f"Fold scores : {scores}")
    print(f"Mean        : {mean_score:.4f}")
    print(f"Std dev     : {std_score:.4f}")
    print(
        "Approx. 95% CI: "
        f"[{result['ci95_lower']:.4f}, "
        f"{result['ci95_upper']:.4f}]"
    )
    print("=" * 55)

    return result


def summarize_subject_results(
    subject_results,
    metric_columns=None,
):
    """
    Summarize one row per subject.

    subject_results should be a pandas DataFrame containing
    subject IDs and already-computed subject-level metrics.
    """
    if not isinstance(subject_results, pd.DataFrame):
        raise TypeError("subject_results must be a pandas DataFrame.")

    if subject_results.empty:
        raise ValueError("No subject results were provided.")

    if metric_columns is None:
        metric_columns = [
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "cohen_kappa",
        ]

    available = [
        column
        for column in metric_columns
        if column in subject_results.columns
    ]

    if not available:
        raise ValueError("No requested metric columns were found.")

    summary_rows = []

    for metric in available:
        values = pd.to_numeric(
            subject_results[metric],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        summary_rows.append(
            {
                "metric": metric,
                "n_subjects": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                "min": values.min(),
                "max": values.max(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 55)
    print("SUBJECT-LEVEL SUMMARY")
    print("=" * 55)
    print(summary.to_string(index=False))
    print("=" * 55)

    return summary
