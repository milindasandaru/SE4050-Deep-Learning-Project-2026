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


# Dataset label convention:
# 0 = Phishing
# 1 = Legitimate


def calculate_metrics(y_true, y_prob, threshold=0.5, pos_label=0):
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels (0 = Phishing, 1 = Legitimate).
    y_prob : array-like
        Predicted probabilities for class 1 (Legitimate).
    threshold : float
        Probability threshold for classifying Legitimate (class 1).
    pos_label : int
        Class to treat as the positive class of interest (default 0 for Phishing).

    Returns
    -------
    dict
        accuracy, precision, recall, f1, and roc_auc.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    if pos_label == 0:
        y_true_pos = (y_true == 0).astype(int)
        y_pred_pos = (y_pred == 0).astype(int)
        y_prob_pos = 1.0 - y_prob
    else:
        y_true_pos = (y_true == 1).astype(int)
        y_pred_pos = (y_pred == 1).astype(int)
        y_prob_pos = y_prob

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true_pos, y_pred_pos, zero_division=0)),
        "recall": float(recall_score(y_true_pos, y_pred_pos, zero_division=0)),
        "f1": float(f1_score(y_true_pos, y_pred_pos, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_pos, y_prob_pos))
    }

    return metrics


def plot_confusion_matrix(y_true, y_prob, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix and optionally save to file.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Phishing (0)", "Legitimate (1)"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    """
    Plot ROC curve for detecting Phishing (0) and optionally save to file.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Phishing probability = 1 - P(Legitimate)
    y_prob_phishing = 1.0 - y_prob
    y_true_phishing = (y_true == 0).astype(int)

    fpr, tpr, _ = roc_curve(y_true_phishing, y_prob_phishing)
    auc = roc_auc_score(y_true_phishing, y_prob_phishing)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Phishing ROC-AUC = {auc:.6f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Baseline"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot training and validation accuracy/loss and optionally save to file.
    """
    history_dict = history.history if hasattr(history, "history") else history

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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()