import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.data import DataLoader
from preprocess import prepare_dataset, LiveCellDataset
import time
import gc

from utils.logger_utils import setup_logging
from utils.constants import TIME, model_args, train_cfg, dataset_paths, SAVE_MODEL, DEVICE, RUN_EXP, MODEL_NAME
from models import get_model, save_model
from utils.metrics import calculate_counting_metrics, plot_training_results
from utils.loss import select_loss


DEBUG = False


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))  # Cosine decay
    return LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    images, density_maps, cell_counts = zip(*batch)
    images = torch.stack(images, 0)
    density_maps = torch.stack(density_maps, 0)  # keep float
    cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
    return images, density_maps, cell_counts


def train_cfg_for_optuna(trial, train_cfg):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # log=True, will use log scale to interplolate between lr
    weight_count = trial.suggest_float("weight_count", 0.0, 1.2, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    train_cfg['learning_rate'] = lr
    train_cfg['w_count'] = weight_count
    train_cfg['batch_size'] = batch_size
    train_cfg['num_epochs'] = 10
    return train_cfg





def train_epoch(model, optimizer, criterion, train_loader, device): #, scheduler
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    #is_regressor = MODEL_NAME in ("ViT_Count", "ConvNeXt_Count")
    
    for batch_idx, (images, density_maps, gt_counts) in enumerate(train_loader):
        images = images.to(device).float()
        density_maps = density_maps.to(device).float()
        gt_counts = gt_counts.to(device).float()

        optimizer.zero_grad()
        pred_density = model(images)
        #loss = criterion(pred_density, density_maps, gt_count=gt_counts)
        loss = criterion(pred_density, gt_counts)
        
        # ---- DEBUGGING CHECKS ----
        if batch_idx % 50 == 0 and DEBUG:  # Print every 50 batches to avoid clutter
            print(f"\nBatch {batch_idx}:")
            print(f"Pred density range: {pred_density.min().item():.3f} - {pred_density.max().item():.3f}")
            print(f"True density range: {density_maps.min().item():.3f} - {density_maps.max().item():.3f}")
            print(f"Pred counts: {pred_density.sum(dim=(1,2,3)).cpu().tolist()}")
            print(f"True counts: {density_maps.sum(dim=(1,2,3)).cpu().tolist()}")
            print(f"GT counts: {gt_counts.cpu().tolist()}")
        # --------------------------
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred_counts = []
    all_gt_counts = []
    
    with torch.no_grad():
        for images, density_maps, gt_counts in val_loader:
            images = images.to(device)
            density_maps = density_maps.to(device)
            gt_counts = gt_counts.to(device).float()

            pred_density = model(images)
            #loss = criterion(pred_density, density_maps, gt_count=gt_counts)
            loss = criterion(pred_density, gt_counts)
            total_loss += loss.item()
            
            # Calculate predicted counts
            #pred_counts = pred_density.sum(dim=(1,2,3)).cpu().numpy()
            pred_counts = pred_density
            all_pred_counts.extend(pred_counts)
            all_gt_counts.extend(gt_counts.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_counting_metrics(
        [int(round(x.item())) for x in all_pred_counts],  # <-- convert to float
        [int(x) for x in all_gt_counts],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def train(model, train_dataset, val_dataset, train_cfg, device=DEVICE, optuna=False, trial=None):

    if optuna:
        train_cfg = train_cfg_for_optuna(trial, train_cfg)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=train_cfg['num_workers'], pin_memory=True if device == 'cuda' else False) # num_workers=train_cfg['num_workers']
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=train_cfg['num_workers'], pin_memory=True if device == 'cuda' else False) # num_workers=train_cfg['num_workers'], 

    if train_cfg['optimizer_name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer_name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer_name'] == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=train_cfg['learning_rate'])

    total_training_steps = len(train_loader) * train_cfg['num_epochs']
    warmup_steps = int(0.1 * total_training_steps)  # 10% warm-up

    #scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    #criterion = DensityLoss(w_density=train_cfg['w_density'], w_count=train_cfg['w_count'])
    #criterion = CountLoss()
    criterion = nn.SmoothL1Loss(beta=1.0)

    logger, log_file_path, output_dir = setup_logging(train_cfg, TIME, model_args, optuna=optuna, run_exp=RUN_EXP)

    since = time.time()
    epochs = train_cfg['num_epochs']
    train_losses = []
    val_losses = []
    val_metrics_history = [] 
    best_val_loss = float('inf')
    best_val_accuracy = -1.0
    best_checkpoint = {}
    best_val_metrics = {}

    for epoch in range(epochs):
        start_epoch = time.time()
        logger.warning('Epoch {}/{}'.format(epoch+1, epochs))
        logger.warning('-' * 10)

        train_loss = train_epoch(model, optimizer, criterion, train_loader, device) #, scheduler
        train_losses.append(train_loss)

        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        

        lr = optimizer.param_groups[0]['lr']
        logger.warning(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

                    # Print detailed metrics every 5 epochs
        if val_metrics and (epoch + 1) % 2 == 0:
            logger.warning("Detailed Counting Metrics:")
            logger.warning(f"  Mean GT: {val_metrics['mean_gt']:.1f}, Mean Pred: {val_metrics['mean_pred']:.1f}")
            for thresh in [0, 1, 3, 5, 10, 20]:
                acc = val_metrics[f'acc_thresh_{thresh}']
                logger.warning(f"  Accuracy @ threshold {thresh}: {acc:.1f}%")

        if val_metrics[f'acc_thresh_3'] > best_val_accuracy:
            best_val_accuracy = val_metrics[f'acc_thresh_3']
            best_val_metrics = val_metrics
            # Save checkpoint
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }

        logger.warning('Epoch complete in {:.0f}h {:.0f}m {:.0f}s'.format((time.time() - start_epoch) // 3600, ((time.time() - start_epoch) % 3600) // 60, (time.time() - start_epoch) % 60))

    if SAVE_MODEL and not optuna and RUN_EXP:
        save_model(best_checkpoint, train_cfg, output_dir)

    time_elapsed = time.time() - since
    logger.warning('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    logger.warning("------  The best Statistics  ------")
    logger.warning(f"  Mean GT: {best_val_metrics['mean_gt']:.1f}, Mean Pred: {best_val_metrics['mean_pred']:.1f}")
    for thresh in [0, 1, 3, 5, 10, 20]:
        acc = best_val_metrics[f'acc_thresh_{thresh}']
        logger.warning(f"  Accuracy @ threshold {thresh}: {acc:.1f}%")

    if not optuna and RUN_EXP:
        plot_training_results(train_losses, val_losses, val_metrics_history, output_dir)

    if optuna:
        print("------  The best Statistics  ------")
        print(f"  Mean GT: {best_val_metrics['mean_gt']:.1f}, Mean Pred: {best_val_metrics['mean_pred']:.1f}")
        for thresh in [0, 1, 3, 5, 10, 20]:
            acc = best_val_metrics[f'acc_thresh_{thresh}']
            print(f"  Accuracy @ threshold {thresh}: {acc:.1f}%")
        
        return best_val_accuracy

def main():
    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    train_dataset = LiveCellDataset(dataset_dict['train'], img_size=img_size)
    val_dataset   = LiveCellDataset(dataset_dict['val'],   img_size=img_size)

    try:
        model = get_model()
        model.to(DEVICE)
        train(model, train_dataset, val_dataset, train_cfg, DEVICE)

    finally:
        # Free GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("Freed GPU memory")



import torch
import numpy as np
from cellpose import models

def _to_cellpose_img_and_channels(img_t: torch.Tensor, mode: str):
    """
    img_t: [C,H,W] float tensor in [0,1] or [0,255]
    mode: 'grayscale' or 'rgb'
    returns: np_img (H,W) or (H,W,3), channels tuple for Cellpose
    """
    img = img_t.detach().cpu().float()
    # scale to 0..255 uint8 (Cellpose is fine with float too; uint8 is typical)
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.clamp(0, 255)

    if mode == 'grayscale':
        if img.ndim == 3 and img.shape[0] > 1:
            # take luminance if multi-channel but want grayscale
            r, g, b = img[0], img[1], img[2]
            img = 0.299*r + 0.587*g + 0.114*b
        else:
            img = img[0] if img.ndim == 3 else img
        np_img = img.numpy().astype(np.uint8)    # (H,W)
        channels = [0, 0]                        # cytoplasm from channel 0, no nuclei channel
    elif mode == 'rgb':
        # ensure 3 channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        np_img = img.numpy().transpose(1, 2, 0).astype(np.uint8)  # (H,W,3)
        channels = [2, 1]  # (R as cytoplasm, G as nucleus) — standard Cellpose choice for RGB
    else:
        raise ValueError("mode must be 'grayscale' or 'rgb'")
    return np_img, channels

def _build_cellpose(model_type: str, use_gpu: bool):
    """Construct Cellpose model for both old (Cellpose) and new (CellposeModel) versions."""
    gpu_flag = (use_gpu and torch.cuda.is_available())
    try:
        # Older Cellpose (<=2.x)
        return models.Cellpose(model_type=model_type, gpu=gpu_flag)
    except AttributeError:
        # Newer Cellpose (>=3.x / v4)
        return models.CellposeModel(model_type=model_type, gpu=gpu_flag)

def _cellpose_eval(cp, np_img, channels, diameter):
    """
    Call cp.eval across versions:
    - Newer versions don't accept net_avg
    - Return masks regardless of return tuple length/shape
    """
    try:
        # Newer Cellpose (no net_avg kwarg)
        result = cp.eval(
            np_img,
            channels=channels,
            diameter=diameter,
            augment=False,
            batch_size=1
        )
    except TypeError:
        # Older Cellpose (accepts net_avg)
        result = cp.eval(
            np_img,
            channels=channels,
            diameter=diameter,
            net_avg=True,
            augment=False,
            batch_size=1
        )

    # result may be (masks, flows, styles, diams) or shorter; be defensive
    if isinstance(result, (list, tuple)):
        masks = result[0]
    else:
        masks = result
    if isinstance(masks, list):  # sometimes batched returns are lists
        masks = masks[0]
    return masks

def evaluate_cellpose(val_loader, mode='grayscale', model_type='cyto', diameter=None,
                      use_gpu=True, batch_verbose=False):
    """
    mode: 'grayscale' or 'rgb' depending on your dataset images
    model_type: 'cyto', 'nuclei', 'cyto2', 'cyto3', or 'cpsam' (v4)
    diameter: set ~ average cell diameter in pixels for better results (else None = auto)
    """
    cp = _build_cellpose(model_type=model_type, use_gpu=use_gpu)

    all_pred, all_gt = [], []

    # Cellpose works per-image; iterate items inside each batch
    with torch.no_grad():
        for batch in val_loader:
            # your val_loader yields: (images, density_maps, gt_counts) or (images, _, gt_counts)
            if len(batch) == 3:
                images, _, gt_counts = batch
            else:
                images, gt_counts = batch  # if no density maps
            B = images.size(0)

            for i in range(B):
                np_img, channels = _to_cellpose_img_and_channels(images[i], mode)
                masks = _cellpose_eval(cp, np_img, channels, diameter)
                pred_count = int(masks.max()) if masks is not None else 0

                all_pred.append(pred_count)
                all_gt.append(int(gt_counts[i].item()))
                if batch_verbose:
                    print(f"Pred {pred_count} | GT {int(gt_counts[i].item())}")

    # Your metric function (expects ints)
    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics


# --- example usage (as in your snippet) ---
# if __name__ == "__main__":
#     # main()
#     dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'],
#                                    dataset_paths['path_to_livecell_images'],
#                                    dataset_paths['path_to_labels'])
#     img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
#     val_dataset = LiveCellDataset(dataset_dict['val'], img_size=img_size)
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=train_cfg['batch_size'],
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=train_cfg['num_workers'],
#         pin_memory=True if DEVICE == 'cuda' else False
#     )

#     metrics = evaluate_cellpose(
#         val_loader,
#         mode='grayscale',   # or 'rgb' if your images are 3-channel
#         model_type='cyto',  # try 'nuclei' for nuclear stains; 'cpsam' if you installed v4 models
#         diameter=None,      # set to ~cell diameter in px (e.g., 20) for more stable segmentation
#         use_gpu=True
#     )
#     print(metrics)


import os, glob
import numpy as np
import torch

# ================== image utils ==================
def to_hw_or_hwc(img_t: torch.Tensor) -> np.ndarray:
    """
    Accepts [C,H,W] float tensor in [0,1] or [0,255]; returns (H,W) or (H,W,3) uint8.
    """
    x = img_t.detach().cpu().float()
    if x.max() <= 1.0: x = x * 255.0
    x = x.clamp(0, 255)
    if x.shape[0] == 1:
        return x[0].numpy().astype(np.uint8)            # grayscale
    return x.permute(1, 2, 0).numpy().astype(np.uint8)  # RGB

# ================== robust LACSS imports ==================
def _import_lacss_predictor_cls():
    """
    Return (PredictorClass, flavor) trying several LACSS APIs.
    """
    import importlib
    tried = []

    # 1) lacss.deploy.Predictor (newer API)
    try:
        mod = importlib.import_module("lacss.deploy")
        if hasattr(mod, "Predictor"):
            return getattr(mod, "Predictor"), "deploy.Predictor"
        tried.append("lacss.deploy.Predictor")
    except Exception as e:
        tried.append(f"lacss.deploy.Predictor ! {type(e).__name__}")

    # 2) lacss.deploy.predict.Predictor
    try:
        mod = importlib.import_module("lacss.deploy.predict")
        if hasattr(mod, "Predictor"):
            return getattr(mod, "Predictor"), "deploy.predict.Predictor"
        tried.append("lacss.deploy.predict.Predictor")
    except Exception as e:
        tried.append(f"lacss.deploy.predict.Predictor ! {type(e).__name__}")

    # 3) older alias: lacss.deploy.inference.Inferer
    try:
        mod = importlib.import_module("lacss.deploy.inference")
        if hasattr(mod, "Inferer"):
            return getattr(mod, "Inferer"), "deploy.inference.Inferer"
        tried.append("lacss.deploy.inference.Inferer")
    except Exception as e:
        tried.append(f"lacss.deploy.inference.Inferer ! {type(e).__name__}")

    raise ImportError("Could not find a LACSS Predictor class. Tried: " + " | ".join(tried))

# ================== predictor builder ==================
def build_lacss_predictor(id_or_path: str, **kwargs):
    """
    id_or_path can be:
      - an alias (e.g., 'small-2dL', 'lacss3-small-l'), or
      - a local file, or
      - a directory containing a model file.
    kwargs are passed to the predictor ctor if supported (silently ignored if not).
    """
    Predictor, flavor = _import_lacss_predictor_cls()

    def _try_ctor(**ctor_kwargs):
        # Try several constructor signatures
        # Common ones: Predictor(model_path=...), Predictor(ckpt_path=...), Predictor(path), Predictor(alias)
        for sig in [
            {"model_path": id_or_path, **ctor_kwargs},
            {"ckpt_path": id_or_path, **ctor_kwargs},
            {"weights": id_or_path, **ctor_kwargs},
            {"path": id_or_path, **ctor_kwargs},
            {"url": id_or_path, **ctor_kwargs},
        ]:
            try:
                return Predictor(**sig)
            except TypeError:
                continue
            except Exception as e:
                # If it's clearly not a signature issue, re-raise
                if not isinstance(e, TypeError):
                    continue
        # Try positional only
        try:
            return Predictor(id_or_path, **ctor_kwargs)
        except Exception:
            return None

    # First, try as given
    pred = _try_ctor(**kwargs)
    if pred is not None:
        return pred

    # If it's a directory, try files inside with common extensions
    if os.path.isdir(id_or_path):
        candidates = []
        for pat in ("*.npz", "*.ckpt", "*.pt", "*.pth", "*"):
            candidates.extend(glob.glob(os.path.join(id_or_path, pat)))
        for c in candidates:
            pred = _try_ctor()
            if pred is not None:
                return pred

    # Try appending common extensions
    base = id_or_path.rstrip("/\\")
    for ext in (".npz", ".ckpt", ".pt", ".pth"):
        if os.path.exists(base + ext):
            pred = _try_ctor()
            if pred is not None:
                return pred

    # Last attempt: maybe alias needs no kwargs at all
    pred = _try_ctor()
    if pred is not None:
        return pred

    raise ValueError(f"Could not construct LACSS predictor for '{id_or_path}' using API {flavor}")

# ================== inference wrappers ==================
def _predict_single(predictor, img_np: np.ndarray,
                    output_type: str = "label",
                    reshape_to=None,
                    min_area: float = 0.0,
                    score_threshold: float = 0.5,
                    segmentation_threshold: float = 0.5):
    """
    Try the available predict methods across LACSS versions.
    """
    # 1) predictor.predict(...)
    try:
        return predictor.predict(
            img_np,
            output_type=output_type,
            reshape_to=reshape_to,
            min_area=min_area,
            score_threshold=score_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_iou=1.0,
            normalize=True,
        )
    except AttributeError:
        pass
    except TypeError:
        # retry with a reduced arg set
        try:
            return predictor.predict(img_np, output_type=output_type)
        except Exception:
            pass

    # 2) predictor.predict_on_large_image(...)
    try:
        return predictor.predict_on_large_image(
            img_np,
            output_type=output_type,
            reshape_to=reshape_to,
            min_area=min_area,
            score_threshold=score_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_iou=0.0,
        )
    except AttributeError:
        pass
    except TypeError:
        try:
            return predictor.predict_on_large_image(img_np, output_type=output_type)
        except Exception:
            pass

    # 3) older Inferer: callable or .infer()
    try:
        return predictor(img_np)  # __call__
    except Exception:
        try:
            return predictor.infer(img_np)
        except Exception:
            pass

    raise RuntimeError("No compatible LACSS prediction method found on this predictor instance.")

# ================== counting utils ==================
def lacss_count_from_pred(pred) -> int:
    if isinstance(pred, dict):
        if "pred_label" in pred and pred["pred_label"] is not None:
            lab = pred["pred_label"]
            return int(np.max(lab)) if lab.size else 0
        for key in ("segmentation", "label"):
            if key in pred and pred[key] is not None:
                lab = pred[key]
                return int(np.max(lab)) if hasattr(lab, "size") and lab.size else 0
        for key in ("pred_masks", "instances", "masks"):
            if key in pred and pred[key] is not None:
                m = pred[key]
                if isinstance(m, np.ndarray):  # NxHxW
                    return int(m.shape[0])
                if isinstance(m, (list, tuple)):
                    return int(len(m))
    if isinstance(pred, np.ndarray):
        return int(np.max(pred))
    return 0

# ================== main eval ==================
def evaluate_lacss(val_loader,
                   id_or_path: str,
                   reshape_to=None,
                   min_area: float = 0.0,
                   score_threshold: float = 0.5,
                   segmentation_threshold: float = 0.5,
                   batch_verbose: bool = False):
    """
    Runs LACSS inference and computes your counting metrics.
    """
    predictor = build_lacss_predictor(id_or_path)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, _, gt_counts = batch
            else:
                images, gt_counts = batch

            for i in range(images.size(0)):
                img_np = to_hw_or_hwc(images[i])
                pred = _predict_single(
                    predictor,
                    img_np,
                    output_type="label",
                    reshape_to=reshape_to,
                    min_area=min_area,
                    score_threshold=score_threshold,
                    segmentation_threshold=segmentation_threshold,
                )
                pred_count = lacss_count_from_pred(pred)
                all_pred.append(int(pred_count))
                all_gt.append(int(gt_counts[i].item()))
                if batch_verbose:
                    print(f"Pred {pred_count} | GT {int(gt_counts[i].item())}")

    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics



if __name__ == "__main__":
    # main()
    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'],
                                   dataset_paths['path_to_livecell_images'],
                                   dataset_paths['path_to_labels'])
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    val_dataset = LiveCellDataset(dataset_dict['val'], img_size=img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=train_cfg['num_workers'],
        pin_memory=True if DEVICE == 'cuda' else False
    )
    print("srarted!!!")
    metrics = evaluate_lacss(val_loader, "/home/meidanzehavi/Cell_counter/lacss3-small-l")
    print(metrics)
