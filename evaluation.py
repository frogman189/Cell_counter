import os
import glob
import time
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from preprocess import prepare_dataset, LiveCellDataset
from models import get_model, load_model
from utils.metrics import calculate_counting_metrics
from utils.constants import DEVICE, MODEL_NAME, dataset_paths, model_pathes
from train_CellCounter import collate_fn, DensityLoss

# === toggle here ===
TEST = True  # True -> evaluate 'test', False -> evaluate 'val'


def find_best_checkpoint(run_dir: str) -> str:
    """
    Find the most recent best model checkpoint file.
    
    Args:
        run_dir: Directory containing model checkpoints
        
    Returns:
        str: Path to the most recent best model checkpoint
        
    Raises:
        FileNotFoundError: If no checkpoint files are found
    """
    mw_dir = os.path.join(run_dir, "model_weights")
    pattern = os.path.join(mw_dir, "best_model_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files matching {pattern}")
    # pick latest by mtime
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_saved_cfg_into(run_dir: str):
    """
    Load saved configuration from model training.
    
    Args:
        run_dir: Directory containing model configuration
        
    Returns:
        dict: Model configuration dictionary, or None if loading fails
    """
    config_path = os.path.join(run_dir, "model_weights", "config.json")
    if not os.path.exists(config_path):
        print(f"[Eval] No config.json found at {config_path}; using current train_cfg as-is.")
    try:
        # Load the configuration JSON into a dictionary
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        return model_config
    except Exception as e:
        print(f"file is corrupted {config_path} - {e}")


# def write_eval_report(split, metrics, dt, out_path):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # Collect thresholds dynamically (e.g., acc_thresh_0, acc_thresh_1, ...)
#     thr_pairs = sorted(
#         [(int(k.rsplit("_", 1)[-1]), metrics[k]) for k in metrics if k.startswith("acc_thresh_")],
#         key=lambda x: x[0]
#     )
#     thr_header = "  ".join([f"±{t:>3}" for t, _ in thr_pairs]) or "N/A"
#     thr_values = "  ".join([f"{v:>5.1f}%" for _, v in thr_pairs]) or "N/A"

#     # Nicely aligned key–value section
#     kv = []
#     def add(label, val):
#         if val is not None:
#             kv.append(f"{label:<20} {val}")

#     add("Split", split)
#     add("Images", metrics.get("num_images"))
#     add("Avg Loss", f"{metrics.get('loss', float('nan')):.6f}")
#     if "mean_gt" in metrics:        add("Mean GT", f"{metrics['mean_gt']:.3f}")
#     if "mean_pred" in metrics:      add("Mean Pred (rounded)", f"{metrics['mean_pred']:.3f}")
#     if "mean_pred_raw" in metrics:  add("Mean Pred (raw sum)", f"{metrics['mean_pred_raw']:.3f}")
#     add("Elapsed", f"{dt:.2f}s")

#     report = (
#         "=======================\n"
#         "     Evaluation Run    \n"
#         "=======================\n"
#         f"Timestamp              {timestamp}\n\n"
#         + "\n".join(kv) + "\n\n"
#         + "Accuracy @ absolute error thresholds\n"
#         + "-----------------------------------\n"
#         + thr_header + "\n"
#         + thr_values + "\n"
#     )

#     with open(out_path, "w") as f:
#         f.write(report)

#     print(f"[Eval] Wrote results to: {out_path}")
#     print(report)

def write_eval_report(split, metrics, dt, out_path):
    """
    Write evaluation results to a formatted report file.
    
    Args:
        split: Dataset split being evaluated ('test' or 'val')
        metrics: Dictionary containing evaluation metrics
        dt: Evaluation time in seconds
        out_path: Output file path for the report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # thresholds → [(thr, acc), ...]
    thr_pairs = sorted(
        [(int(k.rsplit("_", 1)[-1]), metrics[k]) for k in metrics if k.startswith("acc_thresh_")],
        key=lambda x: x[0]
    )

    # Simple KV section
    kv = []
    def add(label, val):
        if val is not None:
            kv.append(f"{label:<20} {val}")

    add("Split", split)
    add("Images", metrics.get("num_images"))
    add("Avg Loss", f"{metrics.get('loss', float('nan')):.6f}")
    if "mean_gt" in metrics:        add("Mean GT", f"{metrics['mean_gt']:.3f}")
    if "mean_pred" in metrics:      add("Mean Pred (rounded)", f"{metrics['mean_pred']:.3f}")
    if "mean_pred_raw" in metrics:  add("Mean Pred (raw sum)", f"{metrics['mean_pred_raw']:.3f}")
    add("Elapsed", f"{dt:.2f}s")

    # ---- Accuracy table (minimal) ----
    if thr_pairs:
        headers = [f"±{t}" for t, _ in thr_pairs]
        values  = [f"{v:.2f}%" for _, v in thr_pairs]
        col_w = max(4, max(len(h) for h in headers + values))

        header_row = "| " + " | ".join(h.center(col_w) for h in headers) + " |"
        sep_row    = "|-" + "-|-".join("-"*col_w for _ in headers) + "-|"
        value_row  = "| " + " | ".join(v.center(col_w) for v in values) + " |"
        top_bot    = "_" * len(header_row)

        acc_table = "\n".join([top_bot, header_row, sep_row, value_row, top_bot])
    else:
        acc_table = "No accuracy thresholds found."

    report = (
        "=======================\n"
        "     Evaluation Run    \n"
        "=======================\n"
        f"Timestamp              {timestamp}\n\n"
        + "\n".join(kv) + "\n\n"
        + "Accuracy @ absolute error thresholds\n"
        + acc_table + "\n"
    )

    with open(out_path, "w") as f:
        f.write(report)

    print(f"[Eval] Wrote results to: {out_path}")
    print(report)



@torch.no_grad()
def evaluate_split(model, loader, device, criterion):
    """
    Evaluates the model on a given data loader split (validation or test).
    
    Calculates the average loss, predicted counts, and various counting metrics.
    
    Args:
        model (torch.nn.Module): The density map prediction model.
        loader (torch.utils.data.DataLoader): Data loader for the split to evaluate.
        device (torch.device): The device (CPU/GPU) to run the evaluation on.
        criterion (torch.nn.Module): The loss function (used for reporting the average loss).
        
    Returns:
        dict: A dictionary containing evaluation metrics including loss, num_images, 
              mean_pred_raw, MAE, MSE, and accuracy thresholds.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_pred_counts = []
    all_gt_counts = []

    for images, density_maps, gt_counts in loader:
        images = images.to(device, non_blocking=True)
        density_maps = density_maps.to(device)
        gt_counts = gt_counts.to(device).float()

        pred_density = model(images)  # (N,1,H,W)
        # report the same loss as training's validation for completeness
        loss = criterion(pred_density, density_maps, gt_count=gt_counts)
        total_loss += loss.item()
        n_batches += 1

        # predicted counts = sum of density
        # Calculates the total predicted count by summing all elements in the density map for each image.
        pred_counts = pred_density.sum(dim=(1, 2, 3)).cpu().tolist()
        all_pred_counts.extend(pred_counts)
        all_gt_counts.extend(gt_counts.cpu().tolist())

    # same rounding as training
    # Rounds the raw float predictions before calculating the counting metrics (MAE, MSE, Acc).
    metrics = calculate_counting_metrics(
        [int(round(x)) for x in all_pred_counts],
        [int(x) for x in all_gt_counts],
        thresholds=[0, 1, 3, 5, 10, 20],
    )
    metrics["loss"] = total_loss / max(1, n_batches)
    metrics["num_images"] = len(all_gt_counts)
    metrics["mean_pred_raw"] = float(sum(all_pred_counts) / len(all_pred_counts)) if all_pred_counts else 0.0
    return metrics



def main():
    """
    Main function to orchestrate the evaluation process.
    
    Loads the best model checkpoint and configuration, sets up the evaluation 
    dataset and loader, runs the evaluation, and saves the final report.
    """
    # ---------- paths & checkpoint ----------
    run_dir = model_pathes[MODEL_NAME]
    ckpt_path = find_best_checkpoint(run_dir)
    train_cfg = load_saved_cfg_into(run_dir)

    # ---------- data ----------
    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    split = "test" if TEST else "val"
    # Image size depends on the specific model architecture being used.
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    eval_dataset = LiveCellDataset(dataset_dict[split], img_size=img_size)

    # Sets up the DataLoader with parameters loaded from the training configuration.
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], pin_memory=True if str(DEVICE).startswith("cuda") else False, collate_fn=collate_fn)

    # ---------- model ----------
    model = get_model()
    model.to(DEVICE)
    # we don't need optimizer/scheduler for eval
    load_model(ckpt_path, model, optimizer=None, scheduler=None, device=DEVICE)

    # ---------- criterion (for avg loss reporting) ----------
    criterion = DensityLoss(w_density=train_cfg['w_density'], w_count=train_cfg['w_count'])

    # ---------- evaluate ----------
    t0 = time.time()
    metrics = evaluate_split(model, eval_loader, DEVICE, criterion)
    dt = time.time() - t0

    # ---------- save ----------
    out_name = "evaluation_test.txt" if TEST else "evaluation_val.txt"
    out_path = os.path.join(run_dir, out_name)
    write_eval_report(split, metrics, dt, out_path)


if __name__ == "__main__":
    main()