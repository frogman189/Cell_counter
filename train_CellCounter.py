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
    if MODEL_NAME == 'Mask_R_CNN_ResNet50':
        images, targets, cell_counts = zip(*batch)   # each: image [3,H,W], target dict
        cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
        return list(images), list(targets), cell_counts
    elif MODEL_NAME == 'Unet':
        images, class_maps, cell_counts = zip(*batch)              # image [3,H,W], target [H,W] (long)
        images = torch.stack(images, dim=0)           # [B,3,H,W]
        class_maps = torch.stack(class_maps, dim=0)   # [B,H,W]
        cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
        return images, class_maps, cell_counts
    else:
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
    
    for batch_idx, (images, targets, cell_counts) in enumerate(train_loader):
        optimizer.zero_grad()

        if MODEL_NAME == 'Mask_R_CNN_ResNet50':
            # images: list[tensor], targets: list[dict], cell_counts: None
            images = [img.to(device).float() for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
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
            loss = criterion(preds, targets, None)

            if DEBUG and batch_idx % 50 == 0:
                print(f"Batch {batch_idx} UNet logits range: "
                      f"{preds.min().item():.3f}..{preds.max().item():.3f}")

        else:
            # density/count models: images [B,3,H,W], targets=density [B,1,H,W], cell_counts [B]
            images = images.to(device).float()
            targets = targets.to(device).float()
            cell_counts = None if cell_counts is None else cell_counts.to(device).float()

            preds = model(images)                   # density map [B,1,H,W] OR counts [B]
            loss = criterion(preds, targets, cell_counts)

            if DEBUG and batch_idx % 50 == 0:
                if preds.dim() == 4:
                    # density branch
                    print(f"\nBatch {batch_idx}:")
                    print(f"Pred density range: {preds.min().item():.3f} - {preds.max().item():.3f}")
                    print(f"True density range: {targets.min().item():.3f} - {targets.max().item():.3f}")
                    print(f"Pred counts: {preds.sum(dim=(1,2,3)).detach().cpu().tolist()}")
                    print(f"True counts: {targets.sum(dim=(1,2,3)).detach().cpu().tolist()}")
                    if cell_counts is not None:
                        print(f"GT counts: {cell_counts.detach().cpu().tolist()}")
                else:
                    # direct count regressors
                    print(f"\nBatch {batch_idx}:")
                    print(f"Pred counts: {preds.detach().cpu().tolist()}")
                    if cell_counts is not None:
                        print(f"GT counts: {cell_counts.detach().cpu().tolist()}")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device, *,
                   det_score_thresh: float = 0.5,  # for Mask R-CNN counting
                   seg_bin_thresh: float = 0.5,    # for UNet foreground threshold on cell class prob
                   thresholds = (0, 1, 3, 5, 10, 20)):
    model.eval()
    total_loss = 0.0
    loss_batches = 0

    all_pred_counts = []
    all_gt_counts = []

    with torch.no_grad():
        for images, targets, cell_counts in val_loader:
            if MODEL_NAME == 'Mask_R_CNN_ResNet50':
                # images: list[tensor], targets: list[dict], cell_counts: None
                images = [img.to(device).float() for img in images]

                # predictions (eval path)
                outputs = model(images)  # list of dicts
                for out in outputs:
                    scores = out.get('scores', None)
                    if scores is not None:
                        pred_c = int((scores >= det_score_thresh).sum().item())
                    else:
                        pred_c = int(len(out.get('boxes', [])))
                    all_pred_counts.append(pred_c)

                # GT counts from targets' instance labels
                # for t in targets:
                #     all_gt_counts.append(int(t['labels'].shape[0]))
                if counts is not None:
                    all_gt_counts.extend(counts.detach().cpu().numpy().tolist())

                # no validation loss here for detection (native losses require train-mode forward)

            elif MODEL_NAME == 'Unet':
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
                cell_probs = probs[:, 1] if logits.shape[1] > 1 else torch.sigmoid(logits[:, 0])
                bin_masks = (cell_probs >= seg_bin_thresh).to(torch.uint8).cpu().numpy()

                for bm in bin_masks:
                    # Subtract background (label 0)
                    n, _ = cv2.connectedComponents(bm, connectivity=8)
                    all_pred_counts.append(int(max(n - 1, 0)))

                # GT count from class map (same rule)
                # cm_np = class_maps.cpu().numpy().astype(np.uint8)
                # for cm in cm_np:
                #     n, _ = cv2.connectedComponents((cm > 0).astype(np.uint8), connectivity=8)
                #     all_gt_counts.append(int(max(n - 1, 0)))
                if counts is not None:
                    all_gt_counts.extend(counts.detach().cpu().numpy().tolist())

            else:
                # Density / count models
                # images: [B,3,H,W], targets: density [B,1,H,W] (or dummy), cell_counts: [B] or None
                images = images.to(device)
                dens_or_target = targets.to(device) if targets is not None else None
                counts = cell_counts.to(device) if cell_counts is not None else None

                preds = model(images)  # density map [B,1,H,W] OR counts [B]

                if criterion is not None:
                    loss = criterion(preds, dens_or_target, counts)
                    total_loss += loss.item()
                    loss_batches += 1

                if preds.dim() == 4:  # density -> integrate to counts
                    pred_counts = preds.sum(dim=(1, 2, 3)).detach().cpu().numpy()
                else:                  # direct regressor
                    pred_counts = preds.detach().cpu().numpy()

                all_pred_counts.extend(pred_counts.tolist())

                if counts is not None:
                    all_gt_counts.extend(counts.detach().cpu().numpy().tolist())
                elif dens_or_target is not None:
                    all_gt_counts.extend(dens_or_target.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist())
                else:
                    # no GT available; skip (keeps lengths aligned for other branches)
                    pass

    # Safety: if nothing collected, return just loss (or NaN)
    if len(all_pred_counts) == 0 or len(all_gt_counts) == 0:
        return {
            'loss': (total_loss / loss_batches) if loss_batches > 0 else float('nan'),
        }

    # Round to integers for count metrics (matches your previous behavior)
    preds_int = [int(round(float(x))) for x in all_pred_counts]
    gts_int   = [int(round(float(x))) for x in all_gt_counts]

    metrics = calculate_counting_metrics(preds_int, gts_int, list(thresholds))
    metrics['loss'] = (total_loss / loss_batches) if loss_batches > 0 else float('nan')
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

    if MODEL_NAME != 'Mask_R_CNN_ResNet50':
        criterion = select_loss(train_cfg)
    else:
        criterion=None

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


    try:
        model = get_model()
        model.to(DEVICE)
        train_dataset = load_LiveCellDataSet(mode='train')
        val_dataset   = load_LiveCellDataSet(mode='val')
        train(model, train_dataset, val_dataset, train_cfg[MODEL_NAME], DEVICE)

    finally:
        # Free GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("Freed GPU memory")



if __name__ == "__main__":
    main()
