import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.data import DataLoader
from preprocess import load_LiveCellDataSet
import time
import gc
import numpy as np
import cv2
import copy

from utils.logger_utils import setup_logging
# Importing constants and utilities
from utils.constants import TIME, model_args, train_cfg, dataset_paths, SAVE_MODEL, DEVICE, RUN_EXP, MODEL_NAME
from models import get_model, save_model
from utils.metrics import calculate_counting_metrics, plot_training_results
from utils.loss import select_loss


DEBUG = False  # Set to True to enable debug prints during training and validation

def _clone_state_dict_to_cpu(sd: dict) -> dict:
    # Freeze exact values now: detach → move to CPU → clone storage
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in sd.items()}


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a learning rate scheduler with a linear warmup followed by a cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the linear warmup phase.
        num_training_steps (int): The total number of training steps.

    Returns:
        LambdaLR: The learning rate scheduler.
    """
    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # Max(0.0, ...) ensures LR doesn't drop below zero
        return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))  # Cosine decay
    return LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    """
    Custom collate function to handle different data formats required by various models.

    Separates images, targets (masks/density maps/None), and cell counts from the batch.
    For models that require batching (e.g., density/count/UNet), it stacks tensors.
    For Mask R-CNN, it keeps images and targets as lists of tensors/dicts.

    Args:
        batch (list): A list of samples, where each sample is a tuple
                      (image, target, cell_count).

    Returns:
        tuple: (images, targets, cell_counts) formatted for the specific model.
    """
    if MODEL_NAME == 'Mask_R_CNN_ResNet50':
        # Mask R-CNN expects lists of tensors/dicts
        images, targets, cell_counts = zip(*batch)  # each: image [3,H,W], target dict
        cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
        return list(images), list(targets), cell_counts
    elif MODEL_NAME == 'Unet':
        # UNet expects batched tensors [B,C,H,W] and target maps [B,H,W]
        images, class_maps, cell_counts = zip(*batch)  # image [3,H,W], target [H,W] (long)
        images = torch.stack(images, dim=0)            # [B,3,H,W]
        class_maps = torch.stack(class_maps, dim=0)    # [B,H,W]
        cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
        return images, class_maps, cell_counts
    else:
        # Density/direct count regressors expect batched tensors
        images, density_maps, cell_counts = zip(*batch)
        images = torch.stack(images, 0)
        density_maps = torch.stack(density_maps, 0)  # keep float, density map has shape [B,1,H,W]
        cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
        return images, density_maps, cell_counts


def train_cfg_for_optuna(trial, train_cfg):
    """
    Updates the training configuration dictionary with hyperparameters suggested by Optuna.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        train_cfg (dict): The base training configuration dictionary.

    Returns:
        dict: The updated training configuration dictionary.
    """
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # log=True, will use log scale to interplolate between lr

    # Mask_R_CNN_ResNet50: smaller batch sizes due to memory constraints
    if MODEL_NAME == 'Mask_R_CNN_ResNet50':
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    else:
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    # Suggest weights for models using the custom Density+SSIM loss
    if MODEL_NAME in {'UNetDensity', 'DeepLabDensity', 'MicroCellUNet'}:
        weight_density = trial.suggest_float("weight_density", 0.5, 4.0, step=0.1)
        weight_ssim = trial.suggest_float("weight_ssim", 0.5, 4.0, step=0.1)
        train_cfg['w_density'] = weight_density
        train_cfg['w_ssim'] = weight_ssim

    # Suggest delta for models using Huber Loss (Smooth L1 Loss)
    if MODEL_NAME in {'ConvNeXt_Count', 'ViT_Count'}:
        huber_delta = trial.suggest_float("huber_delta", 0.0, 10.0, step=0.5)
        train_cfg['huber_delta'] = huber_delta

    train_cfg['learning_rate'] = lr
    train_cfg['batch_size'] = batch_size
    train_cfg['num_epochs'] = 10 # keep short for optuna trials
    return train_cfg


def train_epoch(model, optimizer, criterion, train_loader, device): # , scheduler
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module or None): The loss function. None for Mask R-CNN (uses internal losses).
        train_loader (DataLoader): DataLoader for the training data.
        device (str or torch.device): The device to run the training on.

    Returns:
        float: The average loss over the training epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets, cell_counts) in enumerate(train_loader):
        optimizer.zero_grad()

        if MODEL_NAME == 'Mask_R_CNN_ResNet50':
            # images: list[tensor], targets: list[dict], cell_counts: None
            # Move images and targets (including internal tensors) to the device
            images = [img.to(device).float() for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            # Model forward pass in training mode returns a dictionary of losses
            loss_dict = model(images, targets)      # torchvision returns dict of losses in train()
            loss = sum(loss_dict.values())

            if DEBUG and batch_idx % 50 == 0:
                print(f"Batch {batch_idx} MaskRCNN losses:",
                      {k: round(v.item(), 4) for k, v in loss_dict.items()})

        elif MODEL_NAME == 'Unet':
            # images: [B,3,H,W], targets: class map [B,H,W], cell_counts: None
            images = images.to(device).float()
            targets = targets.to(device).long()
            preds = model(images)                   # logits [B,C,H,W]
            # Criterion expects predictions, semantic map target, and count (None for this model)
            loss = criterion(preds, targets, None)

            if DEBUG and batch_idx % 50 == 0:
                print(f"Batch {batch_idx} UNet logits range: "
                      f"{preds.min().item():.3f}..{preds.max().item():.3f}")

        else:
            # density/count models: images [B,3,H,W], targets=density [B,1,H,W], cell_counts [B]
            images = images.to(device).float()
            targets = targets.to(device).float()
            # cell_counts is used as ground truth for models that output only count (e.g., ViT_Count)
            cell_counts = None if cell_counts is None else cell_counts.to(device).float()

            preds = model(images)                   # density map [B,1,H,W] OR counts [B]
            # Criterion takes predictions, density map, and ground truth count
            loss = criterion(preds, targets, cell_counts)

            if DEBUG and batch_idx % 50 == 0:
                if preds.dim() == 4:
                    # density branch debug
                    print(f"\nBatch {batch_idx}:")
                    print(f"Pred density range: {preds.min().item():.3f} - {preds.max().item():.3f}")
                    print(f"True density range: {targets.min().item():.3f} - {targets.max().item():.3f}")
                    print(f"Pred counts: {preds.sum(dim=(1,2,3)).detach().cpu().tolist()}")
                    print(f"True counts: {targets.sum(dim=(1,2,3)).detach().cpu().tolist()}")
                    if cell_counts is not None:
                        print(f"GT counts: {cell_counts.detach().cpu().tolist()}")
                else:
                    # direct count regressors debug
                    print(f"\nBatch {batch_idx}:")
                    print(f"Pred counts: {preds.detach().cpu().tolist()}")
                    if cell_counts is not None:
                        print(f"GT counts: {cell_counts.detach().cpu().tolist()}")

        loss.backward()
        optimizer.step()
        # if scheduler is not None:
        #     scheduler.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device, *,
                     det_score_thresh: float = 0.5,  # for Mask R-CNN counting: confidence threshold for a prediction to be counted
                     seg_bin_thresh: float = 0.5,    # for UNet: foreground threshold on cell class probability
                     thresholds = (0, 1, 3, 5, 10, 20)):
    """
    Performs one epoch of validation and calculates counting metrics.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module or None): The loss function.
        device (str or torch.device): The device to run the validation on.
        det_score_thresh (float): Confidence threshold for counting in Mask R-CNN.
        seg_bin_thresh (float): Probability threshold for generating binary mask in UNet.
        thresholds (tuple): Error thresholds for calculating counting accuracy.

    Returns:
        dict: A dictionary containing the average validation loss and counting metrics.
    """
    model.eval()
    total_loss = 0.0
    loss_batches = 0

    all_pred_counts = []
    all_gt_counts = []

    
    for images, targets, cell_counts in val_loader:
        if MODEL_NAME == 'Mask_R_CNN_ResNet50':
            model.eval()
            with torch.no_grad():
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

            # model.train() # switch back to train mode for loss calculation below    
            # with torch.enable_grad():      # override outer no_grad for this block
            #     # Validation loss calculation for Mask R-CNN (requires forward pass in train mode)
            #     loss_dict = model(images, targets)      # torchvision returns dict of losses in train()
            #     loss = sum(loss_dict.values())

            # # Note: Loss calculation here is slightly flawed as it uses a train-mode forward call
            # # outside of train_epoch, which is the original behavior.
            # total_loss += loss.item()
            # loss_batches += 1

        elif MODEL_NAME == 'Unet':
            with torch.no_grad():
                # images: [B,3,H,W], targets: class map [B,H,W], cell_counts: None
                images = images.to(device).float()
                class_maps = targets.to(device).long()
                counts = cell_counts.to(device) if cell_counts is not None else None

                logits = model(images)  # [B,C,H,W]
                if criterion is not None:
                    loss = criterion(logits, class_maps, None)
                    total_loss += loss.item()
                    loss_batches += 1

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

                if criterion is not None:
                    # Calculate loss using predictions, density map/dummy target, and ground truth counts
                    loss = criterion(preds, dens_or_target, counts)
                    total_loss += loss.item()
                    loss_batches += 1

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

    # Safety: if nothing collected, return just loss (or NaN)
    if len(all_pred_counts) == 0 or len(all_gt_counts) == 0:
        return {
            'loss': (total_loss / loss_batches) if loss_batches > 0 else float('nan'),
        }

    # Round to integers for count metrics
    preds_int = [int(round(float(x))) for x in all_pred_counts]
    gts_int   = [int(round(float(x))) for x in all_gt_counts]

    metrics = calculate_counting_metrics(preds_int, gts_int, list(thresholds))
    metrics['loss'] = (total_loss / loss_batches) if loss_batches > 0 else float('nan')
    return metrics


def train(model, train_dataset, val_dataset, train_cfg, device=DEVICE, optuna=False, trial=None):
    """
    The main training loop orchestrator.

    Handles data loading, optimizer/loss setup, epoch iteration, logging,
    checkpointing, and final result visualization.

    Args:
        model (nn.Module): The model to train.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        train_cfg (dict): The training configuration.
        device (str or torch.device): The device to use for training.
        optuna (bool): If True, runs in Optuna hyperparameter optimization mode.
        trial (optuna.trial.Trial or None): The Optuna trial object if optuna=True.

    Returns:
        float: If optuna=True, returns the best validation accuracy @ threshold 3.
               Otherwise, returns None.
    """

    if optuna:
        # Update config with Optuna suggested parameters
        train_cfg = train_cfg_for_optuna(trial, train_cfg)
        print("lr: ", train_cfg['learning_rate'])

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=train_cfg['num_workers'], pin_memory=True if device == 'cuda' else False) # num_workers=train_cfg['num_workers']
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=train_cfg['num_workers'], pin_memory=True if device == 'cuda' else False) # num_workers=train_cfg['num_workers'], 

    # Select Optimizer
    if train_cfg['optimizer_name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer_name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer_name'] == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=train_cfg['learning_rate'])

    # total_training_steps = len(train_loader) * train_cfg['num_epochs']
    # warmup_steps = int(0.1 * total_training_steps)  # 10% warm-up

    # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    # Select Loss Function
    if MODEL_NAME != 'Mask_R_CNN_ResNet50':
        criterion = select_loss(train_cfg)
    else:
        criterion=None # Mask R-CNN uses its own internal loss functions

    # Setup logging and output directory
    logger, log_file_path, output_dir = setup_logging(train_cfg, TIME, model_args, optuna=optuna, run_exp=RUN_EXP)

    since = time.time()
    epochs = train_cfg['num_epochs']
    train_losses = []
    val_losses = []
    val_metrics_history = [] 
    best_val_loss = float('inf') # not used for checkpointing in this version
    best_val_accuracy = -1.0 # Tracks accuracy @ threshold 3 for best model selection
    # best_checkpoint = {}
    best_epoch        = None
    best_model_cpu    = None
    best_val_metrics = {}

    for epoch in range(epochs):
        start_epoch = time.time()
        logger.warning('Epoch {}/{}'.format(epoch+1, epochs))
        logger.warning('-' * 10)

        train_loss = train_epoch(model, optimizer, criterion, train_loader, device) #, scheduler
        train_losses.append(train_loss)

        # Validate and calculate metrics
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        # Get current learning rate from optimizer
        lr = optimizer.param_groups[0]['lr']
        logger.warning(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")
        if DEBUG and optuna:
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}") # debug

        # Print detailed metrics every 2 epochs
        if val_metrics and (epoch + 1) % 2 == 0:
            logger.warning("Detailed Counting Metrics:")
            logger.warning(f"  Mean GT: {val_metrics['mean_gt']:.1f}, Mean Pred: {val_metrics['mean_pred']:.1f}")
            for thresh in [0, 1, 3, 5, 10, 20]:
                acc = val_metrics[f'acc_thresh_{thresh}']
                logger.warning(f"  Accuracy @ threshold {thresh}: {acc:.1f}%")

        # Checkpoint if validation accuracy @ threshold 3 improves
        if val_metrics[f'acc_thresh_3'] > best_val_accuracy:
            # best_val_accuracy = val_metrics[f'acc_thresh_3']
            # best_val_metrics = val_metrics
            # # Save checkpoint
            # best_checkpoint = {
            #     'epoch': epoch + 1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     #'scheduler_state_dict': scheduler.state_dict(),
            #     'train_loss': train_loss,
            #     'val_loss': val_loss
            # }
            best_val_accuracy = val_metrics[f'acc_thresh_3']
            best_val_metrics  = copy.deepcopy(val_metrics)
            best_epoch        = epoch + 1

            best_model_cpu = _clone_state_dict_to_cpu(model.state_dict())

        logger.warning('Epoch complete in {:.0f}h {:.0f}m {:.0f}s'.format((time.time() - start_epoch) // 3600, ((time.time() - start_epoch) % 3600) // 60, (time.time() - start_epoch) % 60))

    if SAVE_MODEL and not optuna and RUN_EXP:
        #save_model(best_checkpoint, train_cfg, output_dir)
        best_checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": best_model_cpu,
            #"optimizer_state_dict": best_optim_cpu,  # or omit if you don't need resume-from-best
            "train_cfg": copy.deepcopy(train_cfg),
            "metrics": best_val_metrics,
        }
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
        # Print results directly for Optuna output parsing
        print("------  The best Statistics  ------")
        print(f"  Mean GT: {best_val_metrics['mean_gt']:.1f}, Mean Pred: {best_val_metrics['mean_pred']:.1f}")
        for thresh in [0, 1, 3, 5, 10, 20]:
            acc = best_val_metrics[f'acc_thresh_{thresh}']
            print(f"  Accuracy @ threshold {thresh}: {acc:.1f}%")
        
        return best_val_accuracy


def main():
    """
    Main function to setup and run the training process.
    """

    try:
        # Initialize model and move to device
        model = get_model()
        model.to(DEVICE)
        # Load datasets
        train_dataset = load_LiveCellDataSet(mode='train')
        val_dataset   = load_LiveCellDataSet(mode='val')
        # Start training
        train(model, train_dataset, val_dataset, train_cfg[MODEL_NAME], DEVICE)

    finally:
        # Ensure GPU memory is freed after training finishes or encounters an error
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("Freed GPU memory")


if __name__ == "__main__":
    main()