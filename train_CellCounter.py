import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision
import os
from preprocess import prepare_dataset, LiveCellDataset
import numpy as np
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from typing import Dict, Any, List, Tuple, Optional
import time

from utils.logger_utils import setup_logging
from utils.constants import TIME, model_args, train_cfg, dataset_paths, SAVE_MODEL
from models import get_model, save_model



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





def collate_fn(batch):
    """Custom collate function for handling variable-sized targets"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


def count_predictions(prediction: Dict, confidence_threshold: float = 0.5):
    """Count predicted objects above confidence threshold"""
    if 'scores' in prediction and len(prediction['scores']) > 0:
        valid_predictions = prediction['scores'] >= confidence_threshold
        return valid_predictions.sum().item()
    return 0
    
def calculate_counting_metrics(predictions: List[int], ground_truths: List[int], thresholds: List[int]):
    """Calculate MSE and threshold-based accuracy metrics"""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # MSE
    mse = np.mean((predictions - ground_truths) ** 2)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predictions - ground_truths))
    
    # Threshold-based accuracies
    metrics = {
        'mse': mse,
        'mae': mae,
        'mean_pred': np.mean(predictions),
        'mean_gt': np.mean(ground_truths)
    }
    
    for threshold in thresholds:
        correct_predictions = np.abs(predictions - ground_truths) <= threshold
        accuracy = np.mean(correct_predictions) * 100  # Convert to percentage
        metrics[f'acc_thresh_{threshold}'] = accuracy
    
    return metrics


def train_epoch(model, optimizer, train_loader, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, targets in train_loader:
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
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        print(f"Batch Loss: {loss.item():.4f}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
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


def train(model, train_dataset, val_dataset, train_cfg, optimizer, scheduler, device):

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn, pin_memory=True if device == 'cuda' else False) # num_workers=train_cfg['num_workers']
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_fn, pin_memory=True if device == 'cuda' else False) # num_workers=train_cfg['num_workers'], 

    logger, log_file_path, output_dir = setup_logging(train_cfg, TIME, model_args, optuna=False, run_exp=True)

    since = time.time()
    epochs = train_cfg['num_epochs']
    train_losses = []
    val_losses = []
    val_metrics_history = [] 
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_checkpoint = {}

    for epoch in range(epochs):
        start_epoch = time.time()
        logger.warning('Epoch {}/{}'.format(epoch+1, epochs))
        logger.warning('-' * 10)

        train_loss = train_epoch(model, optimizer, train_loader, device)
        train_losses.append(train_loss)

        val_metrics = validate_epoch(model, val_loader, device)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        logger.warning(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

                    # Print detailed metrics every 5 epochs
        if val_metrics and (epoch + 1) % 2 == 0:
            logger.warning("Detailed Counting Metrics:")
            logger.warning(f"  Mean GT: {val_metrics['mean_gt']:.1f}, Mean Pred: {val_metrics['mean_pred']:.1f}")
            for thresh in [0, 1, 3, 5, 10, 20]:
                acc = val_metrics[f'acc_thresh_{thresh}']
                logger.warning(f"  Accuracy @ threshold {thresh}: {acc:.1f}%")

        if val_metrics[f'acc_thresh_0'] < best_val_accuracy:
            best_val_accuracy = val_metrics[f'acc_thresh_0']
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

    if SAVE_MODEL:
        save_model(best_checkpoint, output_dir)

    time_elapsed = time.time() - since
    logger.warning('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        




def main():
    # path_to_original_dataset = "/home/meidanzehavi/livecell"
    # path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
    # path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    train_dataset = LiveCellDataset(dataset_dict['train'])
    val_dataset = LiveCellDataset(dataset_dict['val'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.to(device)

    if train_cfg['optimizer_name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer_name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer_name'] == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=train_cfg['learning_rate'])

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train(model, train_dataset, val_dataset, train_cfg, optimizer, scheduler, device)

if __name__ == "__main__":
    main()
