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


def calculate_metrics(y_true, y_prob_legitimate, threshold=0.5):
    """
    Calculate binary classification metrics with Phishing (class 0)
    treated as the positive class.

    Parameters
    ----------
    y_true : array-like
        True labels:
        0 = Phishing
        1 = Legitimate

    y_prob_legitimate : array-like
        Model probability for class 1 (Legitimate).

    threshold : float
        Probability threshold for classifying Legitimate.

    Returns
    -------
    dict
        Accuracy, phishing precision, phishing recall,
        phishing F1-score and phishing ROC-AUC.
    """

    y_true = np.asarray(y_true).ravel()
    y_prob_legitimate = np.asarray(y_prob_legitimate).ravel()

    # Keras sigmoid output represents P(class 1) = P(Legitimate)
    #
    # Therefore:
    # P(Phishing) = 1 - P(Legitimate)
    y_prob_phishing = 1.0 - y_prob_legitimate

    # Class prediction:
    # probability of Legitimate >= threshold -> 1
    # otherwise -> 0 (Phishing)
    y_pred = (y_prob_legitimate >= threshold).astype(int)

    # Treat Phishing (class 0) as the positive class.
    y_true_phishing = (y_true == 0).astype(int)
    y_pred_phishing = (y_pred == 0).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true_phishing,
            y_pred_phishing,
            zero_division=0
        ),

        "recall": recall_score(
            y_true_phishing,
            y_pred_phishing,
            zero_division=0
        ),

        "f1": f1_score(
            y_true_phishing,
            y_pred_phishing,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true_phishing,
            y_prob_phishing
        )
    }

    return metrics


def plot_confusion_matrix(
    y_true,
    y_prob_legitimate,
    title="Confusion Matrix"
):
    """
    Plot confusion matrix using the actual dataset label convention.

    Class mapping:
    0 = Phishing
    1 = Legitimate
    """

    y_true = np.asarray(y_true).ravel()
    y_prob_legitimate = np.asarray(y_prob_legitimate).ravel()

    y_pred = (y_prob_legitimate >= 0.5).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Phishing", "Legitimate"]
    )

    disp.plot(cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true,
    y_prob_legitimate,
    title="ROC Curve"
):
    """
    Plot ROC curve for detecting Phishing.

    Since the model outputs P(Legitimate), the phishing
    probability is calculated as:

        P(Phishing) = 1 - P(Legitimate)
    """

    y_true = np.asarray(y_true).ravel()
    y_prob_legitimate = np.asarray(y_prob_legitimate).ravel()

    # Convert to phishing probability.
    y_prob_phishing = 1.0 - y_prob_legitimate

    # Convert labels so:
    # 1 = Phishing
    # 0 = Legitimate
    y_true_phishing = (y_true == 0).astype(int)

    fpr, tpr, _ = roc_curve(
        y_true_phishing,
        y_prob_phishing
    )

    auc = roc_auc_score(
        y_true_phishing,
        y_prob_phishing
    )

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
        color="gray",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history,
    title="Training History"
):
    """
    Plot training and validation accuracy/loss.
    """

    history_dict = history.history

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

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

