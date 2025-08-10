import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision
import os
from preprocess import prepare_dataset, LiveCellDataset
import numpy as np
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import time
from scipy.ndimage import label as connected_components
from torch.nn import SmoothL1Loss

from utils.logger_utils import setup_logging
from utils.constants import TIME, model_args, train_cfg, dataset_paths, SAVE_MODEL, DEVICE, RUN_EXP, MODEL_NAME
from models import get_model, save_model
from utils.metrics import count_predictions, calculate_counting_metrics, plot_training_results, count_from_mask

# --- ADD: a tiny factory for the image-level regression loss ---
def make_regression_loss(loss_type: str = "huber", huber_delta: float = 5.0):
    huber_delta = float(huber_delta)  # <— add this
    if loss_type == "poisson":
        def _poisson(pred, target):
            return F.poisson_nll_loss(pred, target, log_input=False, full=True)
        return _poisson
    def _huber(pred, target):
        return F.smooth_l1_loss(pred, target, beta=huber_delta)
    return _huber

#     self.plot_training_history(save_dir)

# def plot_training_history(self, save_dir: str):
#     """Plot and save training history"""
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(self.train_losses, label='Train Loss')
#     if self.val_losses:
#         plt.plot(self.val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training History')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(1, 2, 2)
#     plt.plot([self.scheduler.get_last_lr()[0]] * len(self.train_losses))
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rate Schedule')
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'training_history.png'))
#     plt.close()


class CountMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_counts, true_counts):
        return nn.functional.mse_loss(pred_counts.float(), true_counts.float())
    

class FocalLoss(nn.Module):
    """
    Multi-class focal loss on logits (CrossEntropy with modulating factor).
    targets: LongTensor of shape (N, H, W) with class indices [0..C-1].
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); targets: (N, H, W)
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        # pt = exp(-CE)
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            # exclude ignored from mean
            if self.ignore_index >= 0:
                mask = (targets != self.ignore_index).float()
                return (focal * mask).sum() / (mask.sum().clamp_min(1.0))
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


class SoftDiceLoss(nn.Module):
    """
    Multi-class soft Dice on probabilities (softmax inside).
    By default averages over all classes; you can choose to exclude background.
    targets: LongTensor (N, H, W)
    """
    def __init__(self, smooth: float = 1e-6, exclude_bg: bool = False, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.exclude_bg = exclude_bg
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); targets: (N, H, W)
        N, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        # One-hot target with ignore support
        with torch.no_grad():
            # targets_one_hot: (N, H, W, C) -> (N, C, H, W)
            targets_clamped = targets.clamp_min(0)  # keep ignore as-is (will mask out below)
            oh = F.one_hot(targets_clamped, num_classes=C).permute(0, 3, 1, 2).to(probs.dtype)  # (N,C,H,W)

            if self.ignore_index >= 0:
                valid = (targets != self.ignore_index).unsqueeze(1)  # (N,1,H,W)
                oh = oh * valid  # zero out ignored in one-hot
                probs = probs * valid  # exclude ignored from prediction mass as well

        dims = (0, 2, 3)  # sum over N,H,W per class
        if self.exclude_bg and C > 1:
            probs = probs[:, 1:, :, :]
            oh    = oh[:, 1:, :, :]

        intersection = (probs * oh).sum(dims)
        denom = (probs * probs).sum(dims) + (oh * oh).sum(dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class CompoundSegLoss(nn.Module):
    """
    Focal(γ=2) + Dice (1:1 by default).
    """
    def __init__(self,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25,
                 dice_exclude_bg: bool = False,
                 w_focal: float = 1.0,
                 w_dice: float = 1.0,
                 ignore_index: int = -100):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(exclude_bg=dice_exclude_bg, ignore_index=ignore_index)
        self.wf = w_focal
        self.wd = w_dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.wf * self.focal(logits, targets) + self.wd * self.dice(logits, targets)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))  # Cosine decay
    return LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    images, masks, cell_counts = zip(*batch)        # masks are tensors, not dicts
    images = torch.stack(images, 0)                 # (N, C, H, W)
    masks  = torch.stack(masks, 0).long()           # (N, H, W)
    cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
    return images, masks, cell_counts


def train_cfg_for_optuna(trial, train_cfg):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # log=True, will use log scale to interplolate between lr
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_cfg['learning_rate'] = lr
    train_cfg['batch_size'] = batch_size
    return train_cfg



def get_predicted_counts(model, images, device, threshold=0.5):
    if MODEL_NAME == 'Mask_R_CNN_ResNet50':
        outputs = model(images)
        return torch.tensor([count_predictions(p, threshold) for p in outputs], dtype=torch.float32, device=device)
    else:
        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            return torch.tensor(
                [count_from_mask(p.cpu().numpy(), threshold) for p in probs],
                dtype=torch.float32, device=device
            )


class SoftCountWrapper(nn.Module):
    """
    Wraps a segmentation model that outputs logits (N,1,H,W).
    Produces a scalar count per image using soft-count.

    Optionally learns a global scalar alpha multiplying the soft area
    (equivalent to learning 1/avg_cell_area).
    """
    def __init__(self, seg_model: nn.Module, avg_cell_area: float, learn_alpha: bool = False):
        super().__init__()
        self.seg_model = seg_model
        if learn_alpha:
            # initialize alpha ~ 1/avg_cell_area
            self.alpha = nn.Parameter(torch.tensor([1.0/float(avg_cell_area)], dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor([1.0/float(avg_cell_area)], dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.seg_model(images)        # (N,1,H,W)
        probs  = torch.sigmoid(logits)         # (N,1,H,W)
        soft_area = probs.sum(dim=(2,3))       # (N,1)
        pred_counts = soft_area * self.alpha   # (N,1)
        return pred_counts.squeeze(1)          # (N,)


def train_epoch(model, optimizer, criterion, train_loader, device): #, scheduler
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    is_regressor = MODEL_NAME in ("ViT_Count", "ConvNeXt_Count")
    
    for i, (images, targets, gt_counts) in enumerate(train_loader):
        images = images.to(device)
        #targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
        targets = targets.to(device)
        gt_counts = gt_counts.to(device)
        
        optimizer.zero_grad()
        
        if is_regressor:
            # IMAGE -> SCALAR
            pred_counts = model(images)              # (N,)
            loss = criterion(pred_counts, gt_counts) # regression loss
        else:
            # SEGMENTATION path (unchanged)
            logits = model(images)
            loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        #print(f"Batch Loss: {loss.item():.4f}")
        #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch with counting metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Counting metrics
    all_predictions = []
    all_ground_truths = []
    thresholds = [0, 1, 3, 5, 10, 20]

    is_regressor = MODEL_NAME in ("ViT_Count", "ConvNeXt_Count")
    
    with torch.no_grad():
        for images, targets, gt_counts in val_loader:
            images = images.to(device)
            targets = targets.to(device)  # harmless for regressor
            gt_counts = gt_counts.to(device)

            if is_regressor:
                # ---- image-level regression path ----
                pred_counts = model(images)                 # (N,)
                loss = criterion(pred_counts, gt_counts)    # scalar loss

                # for counting metrics, use integers
                all_predictions.extend([int(round(x)) for x in pred_counts.cpu().tolist()])
                all_ground_truths.extend([int(x) for x in gt_counts.cpu().tolist()])

            else:
                # ---- segmentation path (unchanged) ----
                logits = model(images)                      # (N,C,H,W)
                loss = criterion(logits, targets)

                probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # (N,H,W)
                for p in probs:
                    pred_count = count_from_mask(p.cpu().numpy())
                    all_predictions.append(int(pred_count))
                all_ground_truths.extend([int(x) for x in gt_counts.cpu().tolist()])
            
            total_loss += loss.item()
            #all_predictions.extend(logits.tolist())
            #all_ground_truths.extend(gt_counts.tolist())
            #all_ground_truths.extend([int(x) for x in gt_counts.cpu().tolist()])
            
            num_batches += 1
    
    # Calculate final metrics
    metrics = calculate_counting_metrics(all_predictions, all_ground_truths, thresholds)
    metrics['loss'] = (total_loss / num_batches) if num_batches > 0 else 0.0
    
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
    warmup_steps = int(0.01 * total_training_steps)  # 10% warm-up

    #scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    #criterion = CountMSELoss()
    if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count"):
        loss_type = "huber" #model_args.get("loss_type", "huber")   # 'huber' or 'poisson'
        huber_delta = 5.0 #model_args.get("huber_delta", 5.0)
        criterion = make_regression_loss(loss_type=loss_type, huber_delta=huber_delta)
    else:
        criterion = CompoundSegLoss(
            focal_gamma=2.0, focal_alpha=0.25,
            dice_exclude_bg=False,
            w_focal=1.0, w_dice=1.0,
            ignore_index=255
        )

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

    if SAVE_MODEL and not optuna and not RUN_EXP:
        save_model(best_checkpoint, output_dir)

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
    # path_to_original_dataset = "/home/meidanzehavi/livecell"
    # path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
    # path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    train_dataset = LiveCellDataset(dataset_dict['train'], img_size=img_size)
    val_dataset   = LiveCellDataset(dataset_dict['val'],   img_size=img_size)

    model = get_model()
    model.to(DEVICE)

    train(model, train_dataset, val_dataset, train_cfg, DEVICE)

if __name__ == "__main__":
    main()
