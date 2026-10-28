
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


TABLES = Path("results/tables")
FIGURES = Path("results/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

comparison_file = TABLES / "csp_component_comparison_folds.csv"
subject_file = TABLES / "csp_component_comparison_by_subject.csv"
summary_file = TABLES / "csp_component_comparison_summary.csv"

for path in [comparison_file, subject_file, summary_file]:
    if not path.exists():
        raise FileNotFoundError(f"Required results file missing: {path}")

folds = pd.read_csv(comparison_file)
subjects = pd.read_csv(subject_file)
summary = pd.read_csv(summary_file)

# Use subject-level means so each subject has equal weight.
subject_means = (
    folds.groupby(["subject", "n_components"], as_index=False)
    [["accuracy", "macro_f1", "balanced_accuracy"]]
    .mean()
)

# --------------------------------------------------
# 1. Overall accuracy comparison
# --------------------------------------------------
accuracy = (
    subject_means.groupby("n_components")["accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(
    accuracy["n_components"].astype(str),
    accuracy["mean"] * 100,
    yerr=accuracy["std"] * 100,
    capsize=5,
)
plt.xlabel("Number of CSP components")
plt.ylabel("Mean accuracy across subjects (%)")
plt.title("CSP-LDA: Accuracy Comparison")
plt.tight_layout()
plt.savefig(FIGURES / "csp_accuracy_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# 2. Macro-F1 comparison
# --------------------------------------------------
macro_f1 = (
    subject_means.groupby("n_components")["macro_f1"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(
    macro_f1["n_components"].astype(str),
    macro_f1["mean"] * 100,
    yerr=macro_f1["std"] * 100,
    capsize=5,
)
plt.xlabel("Number of CSP components")
plt.ylabel("Macro-F1 across subjects (%)")
plt.title("CSP-LDA: Macro-F1 Comparison")
plt.tight_layout()
plt.savefig(FIGURES / "csp_macro_f1_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# 3. Subject-wise accuracy comparison
# --------------------------------------------------
pivot = subject_means.pivot(
    index="subject",
    columns="n_components",
    values="accuracy",
).sort_index()

ax = (pivot * 100).plot(
    kind="bar",
    figsize=(11, 6),
    width=0.8,
)

ax.set_xlabel("Subject")
ax.set_ylabel("Mean cross-validation accuracy (%)")
ax.set_title("Subject-wise CSP Component Comparison")
ax.legend(title="CSP components")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / "csp_subject_wise_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# 4. Export a report-ready subject/configuration table
# --------------------------------------------------
report_table = subject_means.copy()

for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
    report_table[metric] = report_table[metric] * 100

report_table = report_table.rename(columns={
    "accuracy": "accuracy_percent",
    "macro_f1": "macro_f1_percent",
    "balanced_accuracy": "balanced_accuracy_percent",
})

report_table.to_csv(
    TABLES / "csp_component_report_table.csv",
    index=False,
)

# --------------------------------------------------
# 5. Print overall results
# --------------------------------------------------
print("\nOverall CSP comparison")
print(summary.to_string(index=False))

print("\nSubject-wise best setting (exploratory)")
best_idx = subject_means.groupby("subject")["accuracy"].idxmax()
best = subject_means.loc[best_idx, ["subject", "n_components", "accuracy"]]
best = best.sort_values("subject")
best["accuracy_percent"] = (best.pop("accuracy") * 100).round(2)
print(best.to_string(index=False))

print("\nGenerated figures:")
print(FIGURES / "csp_accuracy_comparison.png")
print(FIGURES / "csp_macro_f1_comparison.png")
print(FIGURES / "csp_subject_wise_comparison.png")
print(TABLES / "csp_component_report_table.csv")
