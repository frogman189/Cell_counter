import os
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Dict, List
from scipy.ndimage import label as connected_components


def count_predictions(prediction: Dict, confidence_threshold: float = 0.5):
    """
    Counts predicted objects in a Mask R-CNN or similar output dictionary 
    that exceed a specified confidence threshold.

    Args:
        prediction (Dict): A dictionary containing model predictions. 
                           Expected to have a 'scores' key with a torch.Tensor or numpy.ndarray
                           of confidence scores.
        confidence_threshold (float, optional): The minimum score required for a prediction 
                                                to be counted. Defaults to 0.5.

    Returns:
        int: The number of valid predicted objects (i.e., scores >= threshold).
    """
    # Check if 'scores' key exists and if there are any predictions
    if 'scores' in prediction and len(prediction['scores']) > 0:
        # Create a boolean array where True indicates the score is above the threshold
        valid_predictions = prediction['scores'] >= confidence_threshold
        # Sum the True values to get the count and return as a standard Python int
        return valid_predictions.sum().item()
    return 0


def count_from_mask(mask, threshold=0.5):
    """
    Counts distinct, connected objects in a density or segmentation mask 
    using connected components labeling.

    This is typically used for methods that output a density map or segmentation mask 
    from which the count must be inferred.

    Args:
        mask (np.ndarray): The predicted density map or segmentation mask.
        threshold (float, optional): The value used to binarize the mask. 
                                     Pixels above this are considered part of an object. 
                                     Defaults to 0.5.

    Returns:
        int: The number of connected components (objects) found in the binarized mask.
    """
    # Binarize the mask: 1 for values above threshold, 0 otherwise
    binary = (mask > threshold).astype(np.uint8)
    # Apply connected components labeling from scipy.ndimage
    _, num_objects = connected_components(binary)
    return num_objects
    
    
def calculate_counting_metrics(predictions, ground_truths, thresholds):
    """
    Calculates common regression metrics (MSE, MAE) and threshold-based 
    accuracy for cell counting results.

    Args:
        predictions (List[float]): A list or array of predicted cell counts.
        ground_truths (List[float]): A list or array of true (ground truth) cell counts.
        thresholds (List[int]): A list of integer absolute error thresholds 
                                (e.g., [3, 10, 20]) for which to calculate accuracy.

    Returns:
        Dict: A dictionary containing the calculated metrics: 'mse', 'mae', 
              'mean_pred', 'mean_gt', and 'acc_thresh_X' for each threshold X.
    """
    # Convert inputs to numpy arrays for vectorized operations
    predictions = np.array(predictions, dtype=float)
    ground_truths = np.array(ground_truths, dtype=float)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((predictions - ground_truths) ** 2)
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - ground_truths))

    metrics = {
        'mse': mse,
        'mae': mae,
        'mean_pred': float(np.mean(predictions)),
        'mean_gt': float(np.mean(ground_truths)),
    }
    
    # Calculate accuracy for each specified absolute error threshold
    for threshold in thresholds:
        # Check where the absolute error is less than or equal to the threshold
        correct = np.abs(predictions - ground_truths) <= threshold
        # Accuracy is the mean of the boolean array (proportion of True), scaled to percentage
        metrics[f'acc_thresh_{threshold}'] = float(np.mean(correct) * 100.0)
    
    return metrics


def print_metrics(metrics: dict,
                  split: str = "val",
                  model: str | None = None,
                  n_images: int | None = None,
                  decimals: int = 3,
                  show_rmse: bool = False) -> None:
    """
    Nicely prints the dictionary of counting metrics returned by calculate_counting_metrics(...).

    Args:
        metrics (dict): The dictionary of counting metrics.
        split (str, optional): The dataset split (e.g., "val", "test"). Defaults to "val".
        model (str | None, optional): The name of the model being evaluated. Defaults to None.
        n_images (int | None, optional): The number of images used in the evaluation. Defaults to None.
        decimals (int, optional): The number of decimal places for MSE, MAE, and mean counts. Defaults to 3.
        show_rmse (bool, optional): If True, also prints the Root Mean Squared Error (RMSE). Defaults to False.
    """
    
    # Construct the title for the output
    title_bits = [f"Results — {split}"]
    if model:
        title_bits.insert(0, model)
    title = " | ".join(title_bits)
    bar = "-" * len(title)

    # Helper function for consistent formatting of non-percentage numbers
    def fmt(x): return f"{x:.{decimals}f}"

    lines = [title, bar]
    # Add MSE and MAE to the output lines
    lines.append(f"MSE: {fmt(metrics['mse'])}   MAE: {fmt(metrics['mae'])}")
    if show_rmse:
        # Calculate and append RMSE if requested
        lines[-1] += f"   RMSE: {fmt(math.sqrt(metrics['mse']))}"

    # Add mean prediction and mean ground truth counts
    if 'mean_pred' in metrics and 'mean_gt' in metrics:
        lines.append(f"Mean count — Pred: {fmt(metrics['mean_pred'])} | GT: {fmt(metrics['mean_gt'])}")

    if n_images is not None:
        lines.append(f"Images: {n_images}")

    # --- Aligned threshold table (.2f for percentages) ---
    # Extract threshold-based accuracy metrics
    thr_pairs = sorted(
        ((int(k.rsplit("_", 1)[-1]), float(v)) for k, v in metrics.items() if k.startswith("acc_thresh_")),
        key=lambda x: x[0]
    )
    
    if thr_pairs:
        # Prepare headers and values for the table
        headers = [f"±{t}" for t, _ in thr_pairs]
        values  = [f"{v:.2f}%" for _, v in thr_pairs]  # Accuracy values formatted to 2 decimal places

        # Determine column widths for alignment
        min_col = 7 # Minimum width to fit '100.00%'
        widths = [max(len(h), len(val), min_col) for h, val in zip(headers, values)]
        sep = " | "

        # Format and join table rows
        header_row = sep.join(h.center(w) for h, w in zip(headers, widths))
        divider    = sep.join("-" * w      for w in widths)
        value_row  = sep.join(val.rjust(w) for val, w in zip(values, widths)) # Right-justify values

        lines.append("\nAcc @ thresholds")
        lines.append(header_row)
        lines.append(divider)
        lines.append(value_row)
    else:
        lines.append("Acc @ thresholds: N/A")

    print("\n".join(lines))


def plot_training_results(train_losses: List[float], val_losses: List[float], val_metrics_history: List[Dict], output_dir: str,
                          loss_filename: str = "loss_curve.png", acc_filename: str = "accuracy_thresholds.png") -> None:
    """
    Saves two separate plots to the specified output directory:
    1. Training Loss vs Validation Loss over epochs.
    2. Accuracy @ specified thresholds (0, 3, 10, 20) over epochs.

    Args:
        train_losses (List[float]): List of training loss values per epoch.
        val_losses (List[float]): List of validation loss values per epoch.
        val_metrics_history (List[Dict]): List of dictionaries, where each dictionary 
                                          contains validation metrics for an epoch.
        output_dir (str): Directory where the plots will be saved.
        loss_filename (str, optional): Filename for the loss plot. Defaults to "loss_curve.png".
        acc_filename (str, optional): Filename for the accuracy plot. Defaults to "accuracy_thresholds.png".
    """
    # Create output directory if it doesn't exist
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
    acc_thresholds = [0, 3, 10, 20] # Standard thresholds for plotting
    colors = ['red', 'orange', 'purple', 'cyan']

    for threshold, color in zip(acc_thresholds, colors):
        # Extract the accuracy value for the current threshold across all epochs
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