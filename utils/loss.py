import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.constants import MODEL_NAME, DEVICE

class CountLoss(nn.Module):
    """
    A loss function for direct count regression models, combining MAE and MSE.
    """
    def __init__(self):
        """
        Initializes the CountLoss module with Mean Absolute Error (MAE) and 
        Mean Squared Error (MSE) components.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target, gt_count):
        """
        Calculates the weighted sum of MAE and MSE between predicted counts (pred) 
        and ground truth counts (gt_count).

        Args:
            pred (torch.Tensor): Predicted cell counts, shape (N,).
            target (torch.Tensor): Dummy target (density map/None), not used in this loss.
            gt_count (torch.Tensor): Ground truth cell counts, shape (N,).

        Returns:
            torch.Tensor: The final scalar loss value.
        """
        # Weighted combination of MAE (0.7) and MSE (0.3)
        return 0.7 * self.mae(pred, gt_count) + 0.3 * self.mse(pred, gt_count)


class CellCountingLoss(nn.Module):
    """
    A compound loss for density map prediction models, combining:
    1. Density Map MSE.
    2. Density Map SSIM (Structural Similarity).
    3. Count Consistency L1 Loss.
    """
    def __init__(self, w_density=1.0, w_count=0.5, w_ssim=0.5):
        """
        Initializes the CellCountingLoss.

        Args:
            w_density (float): Weight for the combined density map loss (MSE + SSIM).
            w_count (float): Weight for the count consistency L1 loss.
            w_ssim (float): Weight for the SSIM component within the density map loss.
        """
        super().__init__()
        self.w_density = w_density
        self.w_count = w_count
        self.w_ssim = w_ssim
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # Use full class name, data_range=1.0 assuming normalized maps
        
    def forward(self, pred, target, gt_count=None):
        """
        Calculates the weighted compound loss.

        Args:
            pred (torch.Tensor): Predicted density map, shape (N, 1, H, W).
            target (torch.Tensor): Ground truth density map, shape (N, H, W) or (N, 1, H, W).
            gt_count (torch.Tensor, optional): Ground truth total cell count, shape (N,). 
                                               If None, count is derived from summing `target`.

        Returns:
            torch.Tensor: The final scalar loss value.
        """
        # Ensure proper shapes (N,1,H,W) for density maps
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        # 1. Density map MSE
        mse_loss = F.mse_loss(pred, target)
        
        # 2. Structural similarity
        ssim_loss = 1 - self.ssim(pred, target)  # SSIM is usually in [0, 1]. We maximize similarity by minimizing 1 - SSIM.
        
        # 3. Count consistency
        pred_count = pred.sum(dim=(1,2,3)) # Integrate predicted density map to get count
        # Get true count either from explicit gt_count or by integrating the density map target
        true_count = gt_count if gt_count is not None else target.sum(dim=(1,2,3))
        # Use L1 loss for count consistency
        count_loss = F.l1_loss(pred_count, true_count.to(pred_count))
        
        # Combine weighted loss components
        return (
            self.w_density * (mse_loss + self.w_ssim * ssim_loss) +
            self.w_count * count_loss
        )


class DensityLoss(nn.Module):
    """
    A loss function for density map prediction models, combining:
    1. Density Map MSE.
    2. Count Consistency L1 Loss (without SSIM component).
    """
    def __init__(self, w_density=1.0, w_count=0.1):
        """
        Initializes the DensityLoss.

        Args:
            w_density (float): Weight for the density map MSE loss component.
            w_count (float): Weight for the count consistency L1 loss component.
        """
        super().__init__()
        self.w_density = w_density
        self.w_count = w_count
        
    def forward(self, pred, target, gt_count=None):
        """
        Calculates the weighted combination of MSE on the density map and L1 on the count.

        Args:
            pred (torch.Tensor): Predicted density map, shape (N, 1, H, W).
            target (torch.Tensor): Ground truth density map, shape (N, H, W) or (N, 1, H, W).
            gt_count (torch.Tensor, optional): Ground truth total cell count, shape (N,). 
                                               If None, count is derived from summing `target`.

        Returns:
            torch.Tensor: The final scalar loss value.
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (N,H,W) -> (N,1,H,W)

        # Density map MSE loss
        loss_density = F.mse_loss(pred, target)

        # Count consistency L1 loss
        pred_count = pred.sum(dim=(1,2,3))
        if gt_count is not None:
            # Use explicit GT count if available
            gt_count = gt_count.to(pred_count.dtype).to(pred_count.device)
            loss_count = F.l1_loss(pred_count, gt_count)
        else:
            # Otherwise, use count integrated from the density map target
            true_count_from_map = target.sum(dim=(1,2,3))
            loss_count = F.l1_loss(pred_count, true_count_from_map)

        return self.w_density * loss_density + self.w_count * loss_count
    

class RegressionLoss(nn.Module):
    """
    A loss function for direct count regression models, supporting Huber (Smooth L1) 
    and Poisson Negative Log-Likelihood losses.
    """
    def __init__(self, loss_type: str = "huber", huber_delta: float = 5.0, reduction: str = "mean"):
        """
        Initializes the RegressionLoss.

        Args:
            loss_type (str): The type of loss to use: "huber" (default) or "poisson".
            huber_delta (float): The delta parameter ($\delta$) for SmoothL1Loss (Huber loss).
            reduction (str): Specifies the reduction to apply to the output: 
                             'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.huber_delta = float(huber_delta)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, gt_count= None) -> torch.Tensor:
        """
        Calculates the loss between predicted counts (pred) and ground truth counts (gt_count).

        Args:
            pred (torch.Tensor): Predicted cell counts, shape (N,).
            target (torch.Tensor): Dummy target (density map/None), not used in this loss.
            gt_count (torch.Tensor): Ground truth cell counts, shape (N,).

        Returns:
            torch.Tensor: The final scalar loss value.
        """
        # Ensure ground truth count has the correct dtype for calculation
        gt_count = gt_count.to(pred.dtype)

        if self.loss_type == "poisson":
            # Poisson Negative Log-Likelihood Loss
            return F.poisson_nll_loss(
                pred, gt_count,
                log_input=False,  # we pass $\lambda$ (the count prediction), not log $\lambda$
                full=True,        # includes Stirling term for better numerical stability with large counts
                reduction=self.reduction
            )

        # Default: Smooth L1 (Huber) Loss
        return F.smooth_l1_loss(
            pred, gt_count,
            beta=self.huber_delta,
            reduction=self.reduction
        )
    

class FocalLoss(nn.Module):
    """
    Multi-class focal loss, implemented on logits, equivalent to modulated CrossEntropy.

    Args:
        logits (torch.Tensor): Predicted logits, shape (N, C, H, W).
        targets (torch.Tensor): Ground truth class indices, shape (N, H, W) with class indices [0..C-1].
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", ignore_index: int = -100):
        """
        Initializes the FocalLoss.

        Args:
            gamma (float): Modulating factor exponent ($\gamma$).
            alpha (float): Weighting factor ($\alpha$).
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); targets: (N, H, W)
        # Standard Cross-Entropy loss calculated element-wise
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        # pt = $\text{exp}(-\text{CE})$ is the probability of the correct class
        pt = torch.exp(-ce)
        # Focal loss formula: $\alpha * (1 - pt)^\gamma * \text{CE}$
        focal = self.alpha * (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            # exclude ignored from mean calculation
            if self.ignore_index >= 0:
                # Create a mask to identify non-ignored elements
                mask = (targets != self.ignore_index).float()
                # Sum of focal loss in non-ignored regions divided by the number of non-ignored elements
                return (focal * mask).sum() / (mask.sum().clamp_min(1.0))
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


class SoftDiceLoss(nn.Module):
    """
    Multi-class soft Dice loss, applied on probabilities (softmax is performed internally on logits).
    Averages over all classes by default, with an option to exclude the background class.

    Args:
        logits (torch.Tensor): Predicted logits, shape (N, C, H, W).
        targets (torch.Tensor): Ground truth class indices, shape (N, H, W).
    """
    def __init__(self, smooth: float = 1e-6, exclude_bg: bool = False, ignore_index: int = -100):
        """
        Initializes the SoftDiceLoss.

        Args:
            smooth (float): Smoothing factor ($\epsilon$) to prevent division by zero.
            exclude_bg (bool): If True, the loss is calculated only on foreground classes (1 to C-1).
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
        """
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
            # Clamp targets before one-hot to handle negative ignore_index for F.one_hot
            targets_clamped = targets.clamp_min(0) 
            # Convert targets to one-hot encoding: (N, H, W, C) -> (N, C, H, W)
            oh = F.one_hot(targets_clamped, num_classes=C).permute(0, 3, 1, 2).to(probs.dtype) 

            if self.ignore_index >= 0:
                # Create mask for valid pixels (not ignored)
                valid = (targets != self.ignore_index).unsqueeze(1)  # (N,1,H,W)
                oh = oh * valid  # Zero out ignored pixels in one-hot target
                probs = probs * valid  # Zero out prediction mass for ignored pixels

        dims = (0, 2, 3)  # Sum over Batch, Height, and Width for Dice calculation (per class)
        if self.exclude_bg and C > 1:
            # Exclude the background class (channel 0)
            probs = probs[:, 1:, :, :]
            oh    = oh[:, 1:, :, :]

        # Dice calculation: (2 * intersection) / (pred_sq_sum + target_sq_sum)
        intersection = (probs * oh).sum(dims)
        # Using (probs * probs) and (oh * oh) for denominator simplifies to a standard soft Dice
        denom = (probs * probs).sum(dims) + (oh * oh).sum(dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        # Dice loss is 1 - Dice Score, averaged over classes
        loss = 1.0 - dice.mean()
        return loss


class CompoundSegLoss(nn.Module):
    """
    A compound loss for semantic segmentation, typically Focal Loss + Soft Dice Loss.
    """
    def __init__(self,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25,
                 dice_exclude_bg: bool = False,
                 w_focal: float = 1.0,
                 w_dice: float = 1.0,
                 ignore_index: int = -100):
        """
        Initializes the CompoundSegLoss.

        Args:
            focal_gamma (float): $\gamma$ parameter for Focal Loss.
            focal_alpha (float): $\alpha$ parameter for Focal Loss.
            dice_exclude_bg (bool): If True, excludes background class from Dice Loss calculation.
            w_focal (float): Weight for the Focal Loss component.
            w_dice (float): Weight for the Soft Dice Loss component.
            ignore_index (int): Specifies a target value that is ignored in both loss components.
        """
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(exclude_bg=dice_exclude_bg, ignore_index=ignore_index)
        self.wf = w_focal
        self.wd = w_dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, gt_count=None) -> torch.Tensor:
        """
        Calculates the weighted sum of Focal Loss and Soft Dice Loss.

        Args:
            logits (torch.Tensor): Predicted logits, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth class indices, shape (N, H, W).
            gt_count (torch.Tensor, optional): Not used in this segmentation loss.

        Returns:
            torch.Tensor: The final scalar loss value.
        """
        return self.wf * self.focal(logits, targets) + self.wd * self.dice(logits, targets)
    

def select_loss(train_cfg):
    """
    Selects and initializes the appropriate loss function based on the global MODEL_NAME 
    and model-specific configurations in train_cfg.

    Args:
        train_cfg (dict): The training configuration dictionary containing 
                          model-specific hyperparameters (e.g., 'huber_delta', 'w_density').

    Returns:
        nn.Module: The initialized loss function module.
    """
    criterion = None # Initialize criterion to None

    if MODEL_NAME == "Unet":
        # Segmentation loss for UNet
        criterion = CompoundSegLoss()
    elif MODEL_NAME == "ViT_Count":
        # Regression loss for direct count prediction
        criterion = RegressionLoss(huber_delta=train_cfg['huber_delta'])
    elif MODEL_NAME == "ConvNeXt_Count":
        # Regression loss for direct count prediction
        criterion = RegressionLoss(huber_delta=train_cfg['huber_delta'])
    elif MODEL_NAME == "UNetDensity":
        # Compound density and count loss for density map prediction models
        criterion = CellCountingLoss(w_density=train_cfg['w_density'], w_ssim=train_cfg['w_ssim']).to(DEVICE)
    elif MODEL_NAME == 'DeepLabDensity':
        # Compound density and count loss
        criterion = CellCountingLoss(w_density=train_cfg['w_density'], w_ssim=train_cfg['w_ssim']).to(DEVICE)
    elif MODEL_NAME == "MicroCellUNet":
        # Compound density and count loss
        criterion = CellCountingLoss(w_density=train_cfg['w_density'], w_ssim=train_cfg['w_ssim']).to(DEVICE)
    elif MODEL_NAME == 'CNNTransformerCounter':
        # Simple count regression loss
        criterion = CountLoss()

    return criterion