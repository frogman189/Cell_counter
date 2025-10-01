import os
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Dict, List
from scipy.ndimage import label as connected_components


def count_predictions(prediction: Dict, confidence_threshold: float = 0.5):
    """Count predicted objects above confidence threshold"""
    if 'scores' in prediction and len(prediction['scores']) > 0:
        valid_predictions = prediction['scores'] >= confidence_threshold
        return valid_predictions.sum().item()
    return 0


def count_from_mask(mask, threshold=0.5):
    binary = (mask > threshold).astype(np.uint8)
    _, num_objects = connected_components(binary)
    return num_objects
    


def calculate_counting_metrics(predictions, ground_truths, thresholds):
    """Calculate MSE/MAE and threshold-based accuracy for counts."""
    predictions = np.array(predictions, dtype=float)
    ground_truths = np.array(ground_truths, dtype=float)

    mse = np.mean((predictions - ground_truths) ** 2)
    mae = np.mean(np.abs(predictions - ground_truths))

    metrics = {
        'mse': mse,
        'mae': mae,
        'mean_pred': float(np.mean(predictions)),
        'mean_gt': float(np.mean(ground_truths)),
    }
    for threshold in thresholds:
        correct = np.abs(predictions - ground_truths) <= threshold
        metrics[f'acc_thresh_{threshold}'] = float(np.mean(correct) * 100.0)
    return metrics


def print_metrics(metrics: dict,
                  split: str = "val",
                  model: str | None = None,
                  n_images: int | None = None,
                  decimals: int = 3,
                  show_rmse: bool = False) -> None:
    """
    Nicely print the dict returned by calculate_counting_metrics(...).
    """
    
    title_bits = [f"Results — {split}"]
    if model:
        title_bits.insert(0, model)
    title = " | ".join(title_bits)
    bar = "-" * len(title)

    def fmt(x): return f"{x:.{decimals}f}"

    lines = [title, bar]
    lines.append(f"MSE: {fmt(metrics['mse'])}   MAE: {fmt(metrics['mae'])}")
    if show_rmse:
        lines[-1] += f"   RMSE: {fmt(math.sqrt(metrics['mse']))}"

    if 'mean_pred' in metrics and 'mean_gt' in metrics:
        lines.append(f"Mean count — Pred: {fmt(metrics['mean_pred'])} | GT: {fmt(metrics['mean_gt'])}")

    if n_images is not None:
        lines.append(f"Images: {n_images}")

    # --- Aligned threshold table (.2f for percentages) ---
    thr_pairs = sorted(
        ((int(k.rsplit("_", 1)[-1]), float(v)) for k, v in metrics.items() if k.startswith("acc_thresh_")),
        key=lambda x: x[0]
    )
    if thr_pairs:
        headers = [f"±{t}" for t, _ in thr_pairs]
        values  = [f"{v:.2f}%" for _, v in thr_pairs]  # <-- .2f here

        # ensure columns are wide enough for '100.00%' (7 chars) or header, whichever is longer
        min_col = 7
        widths = [max(len(h), len(val), min_col) for h, val in zip(headers, values)]
        sep = " | "

        header_row = sep.join(h.center(w) for h, w in zip(headers, widths))
        divider    = sep.join("-" * w     for w in widths)
        value_row  = sep.join(val.rjust(w) for val, w in zip(values, widths))

        lines.append("\nAcc @ thresholds")
        lines.append(header_row)
        lines.append(divider)
        lines.append(value_row)
    else:
        lines.append("Acc @ thresholds: N/A")

    print("\n".join(lines))


def plot_training_results(train_losses, val_losses, val_metrics_history, output_dir,
                          loss_filename="loss_curve.png", acc_filename="accuracy_thresholds.png"):
    """
    Saves two separate plots:
    1. Training vs Validation Loss.
    2. Accuracy @ thresholds 0, 3, 10, 20.

    Args:
        train_losses (List[float])
        val_losses (List[float])
        val_metrics_history (List[dict])
        output_dir (str): Directory to save the plots.
        loss_filename (str): Filename for the loss plot.
        acc_filename (str): Filename for the accuracy plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    # ---- Plot 1: Training & Validation Loss ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, loss_filename)
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss curve saved to {loss_path}")

    # ---- Plot 2: Accuracy Thresholds ----
    plt.figure(figsize=(8, 5))
    acc_thresholds = [0, 3, 10, 20]
    colors = ['red', 'orange', 'purple', 'cyan']

    for threshold, color in zip(acc_thresholds, colors):
        acc_values = [metrics.get(f'acc_thresh_{threshold}', 0.0) for metrics in val_metrics_history]
        plt.plot(epochs, acc_values, label=f'Accuracy @ ±{threshold}', color=color)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy at Different Thresholds")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(output_dir, acc_filename)
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy thresholds curve saved to {acc_path}")