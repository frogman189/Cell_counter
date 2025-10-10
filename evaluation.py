import os
import glob
import time
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import cv2

# from preprocess import prepare_dataset, LiveCellDataset
from preprocess import load_LiveCellDataSet
from models import get_model, load_model
from utils.metrics import calculate_counting_metrics
from utils.constants import DEVICE, MODEL_NAME, dataset_paths, model_pathes
from train_CellCounter import collate_fn

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

    if "mse" in metrics:  add("MSE:", f"{metrics['mse']:.3f}")
    if "mse" in metrics:
        rmse = metrics['mse'] ** 0.5
        add("RMSE:", f"{rmse:.3f}")
    if "mae" in metrics:  add("MAE:", f"{metrics['mae']:.3f}")
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
def evaluate_split(model, loader, device,
                    det_score_thresh: float = 0.5,  # for Mask R-CNN counting: confidence threshold for a prediction to be counted
                    seg_bin_thresh: float = 0.5,    # for UNet: foreground threshold on cell class probability
                    thresholds = (0, 1, 3, 5, 10, 20)): #, criterion
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
    #total_loss = 0.0

    all_pred_counts = []
    all_gt_counts = []


    for images, targets, cell_counts in loader:
        if MODEL_NAME == 'Mask_R_CNN_ResNet50':
            
            # images: list[tensor], targets: list[dict], cell_counts: None
            images = [img.to(device).float() for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            counts = cell_counts.to(device) if cell_counts is not None else None

            # predictions (eval path) - returns a list of dictionaries with results
            outputs = model(images)  # list of dicts
            for out in outputs:
                scores = out.get('scores', None)
                if scores is not None:
                    # Count instances whose confidence score exceeds the threshold
                    pred_c = int((scores >= det_score_thresh).sum().item())
                else:
                    # Fallback count: total number of predicted boxes/masks
                    pred_c = int(len(out.get('boxes', [])))
                all_pred_counts.append(pred_c)

            # GT counts come from the separate cell_counts tensor from the DataLoader
            if counts is not None:
                all_gt_counts.extend(counts.detach().cpu().numpy().tolist())

            

        elif MODEL_NAME == 'Unet':
            with torch.no_grad():
                # images: [B,3,H,W], targets: class map [B,H,W], cell_counts: None
                images = images.to(device).float()
                class_maps = targets.to(device).long()
                counts = cell_counts.to(device) if cell_counts is not None else None

                logits = model(images)  # [B,C,H,W]
                

                # Pred → count via connected components on predicted foreground
                probs = torch.softmax(logits, dim=1)
                # Cell probability is the second channel (index 1) if multi-class, or sigmoid if single-channel logit
                cell_probs = probs[:, 1] if logits.shape[1] > 1 else torch.sigmoid(logits[:, 0])
                # Threshold the probability map to get a binary mask
                bin_masks = (cell_probs >= seg_bin_thresh).to(torch.uint8).cpu().numpy()

                for bm in bin_masks:
                    # Calculate connected components (instance count)
                    # Label 0 is background, so the number of components is n-1
                    n, _ = cv2.connectedComponents(bm, connectivity=8)
                    all_pred_counts.append(int(max(n - 1, 0)))

                # GT count from the separate cell_counts tensor
                if counts is not None:
                    all_gt_counts.extend(counts.detach().cpu().numpy().tolist())

        else:
            with torch.no_grad():
                # Density / count models
                # images: [B,3,H,W], targets: density [B,1,H,W] (or dummy), cell_counts: [B] or None
                images = images.to(device)
                dens_or_target = targets.to(device) if targets is not None else None
                counts = cell_counts.to(device) if cell_counts is not None else None

                preds = model(images)  # density map [B,1,H,W] OR counts [B]

                if preds.dim() == 4:  # density -> integrate to counts
                    # Sum the density map across all dimensions (excluding batch) to get predicted count
                    pred_counts = preds.sum(dim=(1, 2, 3)).detach().cpu().numpy()
                else:                # direct regressor
                    # Predictions are already the counts
                    pred_counts = preds.detach().cpu().numpy()

                all_pred_counts.extend(pred_counts.tolist())

                if counts is not None:
                    all_gt_counts.extend(counts.detach().cpu().numpy().tolist())
                elif dens_or_target is not None:
                    # Fallback: GT count from integrating the density map (if count not explicitly given)
                    all_gt_counts.extend(dens_or_target.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist())
                else:
                    # no GT available; skip (keeps lengths aligned for other branches)
                    pass


    # Round to integers for count metrics
    preds_int = [int(round(float(x))) for x in all_pred_counts]
    gts_int   = [int(round(float(x))) for x in all_gt_counts]

    metrics = calculate_counting_metrics(preds_int, gts_int, list(thresholds))
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
    split_name = 'test' if TEST else 'val'
    eval_dataset   = load_LiveCellDataSet(mode=split_name)

    # Sets up the DataLoader with parameters loaded from the training configuration.
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], pin_memory=True if str(DEVICE).startswith("cuda") else False, collate_fn=collate_fn)

    # ---------- model ----------
    model = get_model()
    model.to(DEVICE)
    # we don't need optimizer/scheduler for eval
    load_model(ckpt_path, model, optimizer=None, scheduler=None, device=DEVICE)

    # ---------- criterion (for avg loss reporting) ----------
    #criterion = DensityLoss(w_density=train_cfg['w_density'], w_count=train_cfg['w_count'])

    # ---------- evaluate ----------
    t0 = time.time()
    metrics = evaluate_split(model, eval_loader, DEVICE) #, criterion
    dt = time.time() - t0

    # ---------- save ----------
    out_name = "evaluation_test.txt" if TEST else "evaluation_val.txt"
    out_path = os.path.join(run_dir, out_name)
    write_eval_report(split_name, metrics, dt, out_path)


if __name__ == "__main__":
    main()