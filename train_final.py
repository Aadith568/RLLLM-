"""
train_final.py
==============
After the RL agent finds the best hyperparameters, this module trains a
PyTorch model for up to FINAL_EPOCHS epochs.

Features:
  - PyTorch BiLSTM model
  - Early stopping (patience=5 on val_loss)
  - ReduceLROnPlateau learning rate schedule
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from data.imdb_loader import load_imdb_csv
from config import FINAL_EPOCHS, UNITS
from models.bilstm import BiLSTM, device

OUTPUT_DIR = "outputs"
MIN_DROPOUT = 0.4  # Enforce stricter dropout in final training

# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved: {save_path}")

def _plot_accuracy_loss(history, save_dir):
    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in history['accuracy']], "b-o", markersize=3, label="Train Accuracy")
    ax.plot(epochs, [a * 100 for a in history['val_accuracy']], "r-o", markersize=3, label="Val Accuracy")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['loss'], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, history['val_loss'], "r-o", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def run(best_params, x_train, y_train, x_test, y_test, vocab_size,
        epochs=FINAL_EPOCHS):
    """
    Train a PyTorch model for up to `epochs` epochs with early stopping.

    Parameters
    ----------
    best_params : dict  keys: lr, dropout, batch_size
    """

    lr         = best_params["lr"]
    dropout    = max(best_params["dropout"], MIN_DROPOUT)
    batch_size = best_params["batch_size"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FINAL TRAINING (PyTorch)  —  Up to 50 Epochs")
    print("=" * 60)
    print(f"  LR={lr}  Units={UNITS} (Fixed)  Dropout={dropout}  Batch={batch_size}")
    print(f"  Device={device}")
    print("=" * 60 + "\n")

    # 1. Prepare DataLoaders
    # Wrap in tensors
    x_train_t = torch.tensor(x_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Build PyTorch Model
    model = BiLSTM(vocab_size, embedding_dim=128, hidden_units=UNITS, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )

    # 3. Callbacks Setup (Variables)
    early_stop_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {
        'accuracy': [], 'val_accuracy': [],
        'loss': [], 'val_loss': []
    }
    
    precisions, recalls, f1s = [], [], []

    # 4. Train Loop
    print("[TRAIN] Starting PyTorch training loop...")
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            
            # Handle case where output is exactly 1-dimensional but batch size might be 1
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                preds = (outputs > 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history['accuracy'].append(epoch_train_acc)
        history['loss'].append(epoch_train_loss)
        history['val_accuracy'].append(epoch_val_acc)
        history['val_loss'].append(epoch_val_loss)
        
        # Metrics
        p = precision_score(all_targets, all_preds, zero_division=0)
        r = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs}")
        print(f"{len(train_loader)}/{len(train_loader)} ━━━━━━━━━━━━━━━━━━━━ - "
              f"accuracy: {epoch_train_acc:.4f} - loss: {epoch_train_loss:.4f} - "
              f"val_accuracy: {epoch_val_acc:.4f} - val_loss: {epoch_val_loss:.4f} - "
              f"learning_rate: {current_lr:.4e}")
        print(f" — val_precision: {p:.4f} — val_recall: {r:.4f} — val_f1: {f1:.4f}")
        
        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        
        # --- Early Stopping check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print("  [+] Validation loss improved, saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  [-] Early stopping triggered after {epoch} epochs.")
                break

    # 4. Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n[EVAL] Best model weights restored from early stopping checkpoint.")

    # 5. Final Evaluation
    print("\n[EVAL] Predicting on test set with best weights...")
    model.eval()
    all_preds_best = []
    all_targets_best = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
            preds = (outputs > 0.5).float()
            all_preds_best.extend(preds.cpu().numpy())
            all_targets_best.extend(batch_y.cpu().numpy())
            
    best_preds = np.array(all_preds_best).astype(int)
    best_labels = np.array(all_targets_best).astype(int)

    final_acc = np.mean(best_preds == best_labels)

    # 6. Plots
    print("\n[PLOT] Generating metric plots...")
    _plot_confusion_matrix(best_labels, best_preds,
                           os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    _plot_accuracy_loss(history, OUTPUT_DIR)
    
    # Plot PRF
    eps = range(1, len(precisions) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, [p * 100 for p in precisions], "g-o", markersize=3, label="Precision")
    ax.plot(eps, [r * 100 for r in recalls],    "b-o", markersize=3, label="Recall")
    ax.plot(eps, [f * 100 for f in f1s],        "r-o", markersize=3, label="F1 Score")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Precision / Recall / F1 per Epoch (Validation)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    prf_path = os.path.join(OUTPUT_DIR, "metrics_per_epoch.png")
    fig.savefig(prf_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved: {prf_path}")

    # 7. Final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS (PyTorch Best Checkpoint)")
    print("=" * 60)
    print(classification_report(best_labels, best_preds,
                                target_names=["Negative", "Positive"]))
    print(f"  Best Validation Accuracy : {final_acc:.4f}")
    print("=" * 60)
    print(f"\n[DONE] All plots saved to '{OUTPUT_DIR}/' directory.")

    return final_acc


if __name__ == "__main__":
    print("[DATA] Loading IMDB dataset...")
    x_train, y_train, x_test, y_test, vocab_size = load_imdb_csv()

    # Manual fallback for standalone testing
    best_params = {
        "lr":         0.001,
        "dropout":    0.30,
        "batch_size": 64,
    }

    run(best_params, x_train, y_train, x_test, y_test, vocab_size)
