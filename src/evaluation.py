
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = [
    "Left Hand",
    "Right Hand",
    "Feet",
    "Tongue",
]


def evaluate_classifier(
    y_true,
    y_pred,
    class_names=CLASS_NAMES,
):
    """
    Calculate classification metrics and return results.

    y_true: Ground-truth labels for held-out trials.
    y_pred: Predicted labels for the same trials.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have equal, non-zero lengths."
        )

    if not np.isin(y_true, np.arange(len(class_names))).all():
        raise ValueError("Unexpected class labels in y_true.")

    if not np.isin(y_pred, np.arange(len(class_names))).all():
        raise ValueError("Unexpected class labels in y_pred.")

    labels = np.arange(len(class_names))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm,
    }

    print("\n" + "=" * 65)
    print("CLASSIFICATION EVALUATION")
    print("=" * 65)

    print(f"Number of evaluated trials : {len(y_true)}")
    print(f"Accuracy                   : {accuracy:.4f} ({accuracy:.2%})")
    print(f"Macro-F1                   : {macro_f1:.4f}")
    print(f"Balanced Accuracy          : {balanced_accuracy:.4f}")
    print(f"Cohen's Kappa              : {kappa:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print("Rows = actual class; columns = predicted class")
    print("Class order:", ", ".join(class_names))
    print(cm)

    print("=" * 65)

    return results


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=CLASS_NAMES,
    save_path=None,
    show=True,
):
    """
    Plot the confusion matrix.

    Optionally save the figure to save_path.
    """

    labels = np.arange(len(class_names))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of trials")

    ax.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="CSP + LDA Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = cm.max() / 2.0 if cm.size else 0

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return cm
