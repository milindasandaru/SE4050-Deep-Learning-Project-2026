import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels (0 or 1)
    y_prob : array-like
        Predicted probabilities for class 1
    threshold : float
        Probability threshold for converting probabilities to labels

    Returns
    -------
    dict
        Classification metrics
    """

    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

    return metrics


def plot_confusion_matrix(y_true, y_prob, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    """

    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"]
    )

    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """
    Plot ROC curve.
    """

    y_prob = np.asarray(y_prob).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation accuracy/loss.
    """

    history_dict = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(
        history_dict["accuracy"],
        label="Training Accuracy"
    )

    axes[0].plot(
        history_dict["val_accuracy"],
        label="Validation Accuracy"
    )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(
        history_dict["loss"],
        label="Training Loss"
    )

    axes[1].plot(
        history_dict["val_loss"],
        label="Validation Loss"
    )

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()