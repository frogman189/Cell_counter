import torch
import torch.nn as nn
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

from utils.logger_utils import setup_logging
from utils.constants import TIME, model_args, train_cfg, dataset_paths, SAVE_MODEL, DEVICE, RUN_EXP
from models import get_model, save_model
from utils.metrics import count_predictions, calculate_counting_metrics, plot_training_results



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

        

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))  # Cosine decay
    return LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    """Custom collate function for handling variable-sized targets"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


def train_cfg_for_optuna(trial, train_cfg):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # log=True, will use log scale to interplolate between lr
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_cfg['learning_rate'] = lr
    train_cfg['batch_size'] = batch_size
    return train_cfg


def train_epoch(model, optimizer, scheduler, train_loader, device, batch_size=16, accumulation_steps = 4):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
        
        optimizer.zero_grad()
        
        # Forward pass
        losses = model(images, targets)
        
        if isinstance(losses, dict):
            # Mask R-CNN returns dict of losses
            loss = sum(losses.values())
        else:
            loss = losses
        # loss = loss / accumulation_steps

        # loss.backward()

        # if (i + 1) % accumulation_steps == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        #print(f"Batch Loss: {loss.item():.4f}")
        #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Handle remaining gradients (if total batches % accumulation_steps != 0)
    # if (i + 1) % accumulation_steps != 0:
    #     optimizer.step()
    #     optimizer.zero_grad()
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, device):
    """Validate for one epoch with counting metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Counting metrics
    all_predictions = []
    all_ground_truths = []
    thresholds = [0, 1, 3, 5, 10, 20]
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
            
            # Calculate loss (need train mode for Mask R-CNN)
            model.train()
            losses = model(images, targets)
            model.eval()
            
            if isinstance(losses, dict):
                loss = sum(losses.values())
            else:
                loss = losses
            
            total_loss += loss.item()
            
            # Get predictions for counting
            predictions = model(images, None)
            
            # Count objects in predictions and ground truth
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Ground truth count
                gt_count = len(target['labels'])
                
                # Predicted count (filter by confidence score)
                pred_count = count_predictions(pred, confidence_threshold=0.5)
                
                all_predictions.append(pred_count)
                all_ground_truths.append(gt_count)
            
            num_batches += 1
    
    # Calculate final metrics
    metrics = calculate_counting_metrics(all_predictions, all_ground_truths, thresholds)
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    
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

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    logger, log_file_path, output_dir = setup_logging(train_cfg, TIME, model_args, optuna=optuna, run_exp=RUN_EXP)

    since = time.time()
    epochs = train_cfg['num_epochs']
    train_losses = []
    val_losses = []
    val_metrics_history = [] 
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_checkpoint = {}
    best_val_metrics = {}

    for epoch in range(epochs):
        start_epoch = time.time()
        logger.warning('Epoch {}/{}'.format(epoch+1, epochs))
        logger.warning('-' * 10)

        train_loss = train_epoch(model, optimizer, scheduler, train_loader, device)
        train_losses.append(train_loss)

        val_metrics = validate_epoch(model, val_loader, device)
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

        if val_metrics[f'acc_thresh_3'] < best_val_accuracy:
            best_val_accuracy = val_metrics[f'acc_thresh_3']
            best_val_metrics = val_metrics
            # Save checkpoint
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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

    if not optuna and not RUN_EXP:
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
    train_dataset = LiveCellDataset(dataset_dict['train'])
    val_dataset = LiveCellDataset(dataset_dict['val'])

    model = get_model()
    model.to(DEVICE)

    train(model, train_dataset, val_dataset, train_cfg, DEVICE)

if __name__ == "__main__":
    main()
