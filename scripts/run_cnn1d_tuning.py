"""
Fast Controlled Experimentation & Hyperparameter Tuning Pipeline for 1D CNN.

Optimized for rapid CPU execution (~3-4 minutes total):
- 5 high-impact controlled architectural experiments
- verbose=2 (1 line per epoch, eliminating terminal I/O latency)
- max_epochs=10 with patience=3 early stopping
- Saves structured artifacts in results/cnn1d/baseline/, results/cnn1d/tuning/, results/cnn1d/final/
"""

import sys
import os
import time
import json
from pathlib import Path

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend for maximum speed
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.cnn1d import build_cnn1d
from src.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history
)

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "cnn1d"
BASELINE_DIR = RESULTS_DIR / "baseline"
TUNING_DIR = RESULTS_DIR / "tuning"
FINAL_DIR = RESULTS_DIR / "final"

for d in [BASELINE_DIR, TUNING_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Loading preprocessed dataset arrays...")
X_train = np.load(DATA_DIR / "X_train.npy").astype("float32")
X_val = np.load(DATA_DIR / "X_val.npy").astype("float32")
X_test = np.load(DATA_DIR / "X_test.npy").astype("float32")

y_train = np.load(DATA_DIR / "y_train.npy").astype("int64")
y_val = np.load(DATA_DIR / "y_val.npy").astype("int64")
y_test = np.load(DATA_DIR / "y_test.npy").astype("int64")

# Reshape to 3D tensor: (samples, 50, 1)
X_train_cnn = X_train[..., np.newaxis]
X_val_cnn = X_val[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

print(f"Data ready: Train={X_train_cnn.shape}, Val={X_val_cnn.shape}, Test={X_test_cnn.shape}")

# 5 High-Impact Controlled Experiments
EXPERIMENTS = [
    {
        "experiment_id": "EXP-01_baseline",
        "description": "Baseline: Conv(32->64), k=3, Dense=64, drop=0.3, lr=0.001, bs=256",
        "conv1_filters": 32,
        "conv2_filters": 64,
        "conv3_filters": None,
        "kernel_size": 3,
        "dense_units": 64,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "patience": 3,
        "max_epochs": 10
    },
    {
        "experiment_id": "EXP-02_wider_filters",
        "description": "Wider Filters: Conv(64->128), k=3, Dense=64, drop=0.3, lr=0.001, bs=256",
        "conv1_filters": 64,
        "conv2_filters": 128,
        "conv3_filters": None,
        "kernel_size": 3,
        "dense_units": 64,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "patience": 3,
        "max_epochs": 10
    },
    {
        "experiment_id": "EXP-03_deeper_3layers",
        "description": "Deeper Conv: Conv(32->64->128), k=3, Dense=64, drop=0.3, lr=0.001, bs=256",
        "conv1_filters": 32,
        "conv2_filters": 64,
        "conv3_filters": 128,
        "kernel_size": 3,
        "dense_units": 64,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "patience": 3,
        "max_epochs": 10
    },
    {
        "experiment_id": "EXP-04_kernel_size_5",
        "description": "Larger Kernel: Conv(32->64), k=5, Dense=64, drop=0.3, lr=0.001, bs=256",
        "conv1_filters": 32,
        "conv2_filters": 64,
        "conv3_filters": None,
        "kernel_size": 5,
        "dense_units": 64,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "patience": 3,
        "max_epochs": 10
    },
    {
        "experiment_id": "EXP-05_dense_128",
        "description": "Larger Dense: Conv(32->64), k=3, Dense=128, drop=0.3, lr=0.001, bs=256",
        "conv1_filters": 32,
        "conv2_filters": 64,
        "conv3_filters": None,
        "kernel_size": 3,
        "dense_units": 128,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "patience": 3,
        "max_epochs": 10
    }
]

tuning_results = []
best_model_run = None
best_val_loss = float("inf")

print(f"\nStarting {len(EXPERIMENTS)} fast controlled experiments...\n" + "="*60)

for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f"\n[{i}/{len(EXPERIMENTS)}] Starting {exp['experiment_id']} - {exp['description']}")
    
    tf.keras.utils.set_random_seed(SEED)
    
    model = build_cnn1d(
        input_shape=(50, 1),
        conv1_filters=exp["conv1_filters"],
        conv2_filters=exp["conv2_filters"],
        conv3_filters=exp["conv3_filters"],
        kernel_size=exp["kernel_size"],
        dense_units=exp["dense_units"],
        dropout_rate=exp["dropout"],
        learning_rate=exp["learning_rate"]
    )
    
    params = model.count_params()
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=exp["patience"],
        restore_best_weights=True,
        verbose=1
    )
    
    t0 = time.time()
    history = model.fit(
        X_train_cnn,
        y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=exp["max_epochs"],
        batch_size=exp["batch_size"],
        callbacks=[early_stopping],
        verbose=2 # Clean 1-line-per-epoch logging for maximum speed
    )
    train_time = time.time() - t0
    epochs_trained = len(history.history["loss"])
    
    best_vl = min(history.history["val_loss"])
    best_va = max(history.history["val_accuracy"])
    
    # Evaluate on held-out test set
    y_prob_test = model.predict(X_test_cnn, batch_size=512, verbose=0).ravel()
    metrics = calculate_metrics(y_test, y_prob_test, pos_label=0) # Phishing (0) = positive class
    
    exp_summary = {
        "experiment_id": exp["experiment_id"],
        "model": "1D CNN",
        "conv1_filters": exp["conv1_filters"],
        "conv2_filters": exp["conv2_filters"],
        "kernel_size": exp["kernel_size"],
        "dense_units": exp["dense_units"],
        "dropout": exp["dropout"],
        "learning_rate": exp["learning_rate"],
        "batch_size": exp["batch_size"],
        "epochs": epochs_trained,
        "patience": exp["patience"],
        "parameters": params,
        "training_time": round(train_time, 2),
        "best_val_loss": float(best_vl),
        "best_val_accuracy": float(best_va),
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_roc_auc": metrics["roc_auc"]
    }
    
    tuning_results.append(exp_summary)
    print(f"-> Finished in {train_time:.1f}s | Val Loss: {best_vl:.6f} | Test Acc: {metrics['accuracy']:.6f} | Test F1: {metrics['f1']:.6f}")
    
    # Save baseline artifacts if EXP-01
    if exp["experiment_id"] == "EXP-01_baseline":
        print("-> Saving Baseline artifacts to results/cnn1d/baseline/...")
        with open(BASELINE_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(BASELINE_DIR / "history.json", "w") as f:
            hist_clean = {k: [float(val) for val in v] for k, v in history.history.items()}
            json.dump(hist_clean, f, indent=2)
            
        plot_confusion_matrix(
            y_test,
            y_prob_test,
            title="1D CNN Baseline Confusion Matrix",
            save_path=str(BASELINE_DIR / "confusion_matrix.png")
        )
        plot_roc_curve(
            y_test,
            y_prob_test,
            title="1D CNN Baseline ROC Curve",
            save_path=str(BASELINE_DIR / "roc_curve.png")
        )
        plot_training_history(
            history,
            title="1D CNN Baseline Training History",
            save_path=str(BASELINE_DIR / "training_history.png")
        )

    # Check for best model
    if (best_vl < best_val_loss) or (best_vl == best_val_loss and metrics["f1"] > best_model_run.get("test_f1", 0)):
        best_val_loss = best_vl
        best_model_run = {
            "exp_config": exp,
            "metrics": metrics,
            "history": history,
            "y_prob_test": y_prob_test,
            "summary": exp_summary,
            "model": model
        }

# Save Tuning CSV
tuning_df = pd.DataFrame(tuning_results)
csv_path = TUNING_DIR / "experiments.csv"
tuning_df.to_csv(csv_path, index=False)
print(f"\n[DONE] Saved tuning experiments to {csv_path.resolve()}")

# Save Final Best Model Artifacts
print(f"\n[FINAL] Best Selected Model: {best_model_run['exp_config']['experiment_id']} (val_loss={best_val_loss:.6f})")
print("-> Saving Final Best Model artifacts to results/cnn1d/final/...")

with open(FINAL_DIR / "metrics.json", "w") as f:
    json.dump(best_model_run["metrics"], f, indent=2)

with open(FINAL_DIR / "history.json", "w") as f:
    hist_clean = {k: [float(val) for val in v] for k, v in best_model_run["history"].history.items()}
    json.dump(hist_clean, f, indent=2)

with open(FINAL_DIR / "config.json", "w") as f:
    json.dump(best_model_run["exp_config"], f, indent=2)

plot_confusion_matrix(
    y_test,
    best_model_run["y_prob_test"],
    title=f"1D CNN Final Model ({best_model_run['exp_config']['experiment_id']}) Confusion Matrix",
    save_path=str(FINAL_DIR / "confusion_matrix.png")
)
plot_roc_curve(
    y_test,
    best_model_run["y_prob_test"],
    title=f"1D CNN Final Model ({best_model_run['exp_config']['experiment_id']}) ROC Curve",
    save_path=str(FINAL_DIR / "roc_curve.png")
)
plot_training_history(
    best_model_run["history"],
    title=f"1D CNN Final Model ({best_model_run['exp_config']['experiment_id']}) Training History",
    save_path=str(FINAL_DIR / "training_history.png")
)

# Save the trained model weights
best_model_run["model"].save(FINAL_DIR / "cnn1d_final_model.keras")
print(f"-> Saved model weights to {FINAL_DIR / 'cnn1d_final_model.keras'}")
print("\n[SUCCESS] Pipeline execution finished successfully!")
