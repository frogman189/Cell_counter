import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.constants import MODEL_NAME

class CountLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target, gt_count):
        return 0.7 * self.mae(pred, gt_count) + 0.3 * self.mse(pred, gt_count)


class CellCountingLoss(nn.Module):
    def __init__(self, w_density=1.0, w_count=0.3, w_ssim=0.5):
        super().__init__()
        self.w_density = w_density
        self.w_count = w_count
        self.w_ssim = w_ssim
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # Use full class name
        
    def forward(self, pred, target, gt_count=None):
        # Ensure proper shapes (N,1,H,W)
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        # 1. Density map MSE
        mse_loss = F.mse_loss(pred, target)
        
        # 2. Structural similarity
        ssim_loss = 1 - self.ssim(pred, target)  # 0=perfect, 1=worst
        
        # 3. Count consistency
        pred_count = pred.sum(dim=(1,2,3))
        true_count = gt_count if gt_count is not None else target.sum(dim=(1,2,3))
        count_loss = F.l1_loss(pred_count, true_count.to(pred_count))
        
        return (
            self.w_density * (mse_loss + self.w_ssim * ssim_loss) +
            self.w_count * count_loss
        )


class DensityLoss(nn.Module):
    def __init__(self, w_density=1.0, w_count=0.1):
        super().__init__()
        self.w_density = w_density
        self.w_count = w_count
        
    def forward(self, pred, target, gt_count=None):
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (N,H,W) -> (N,1,H,W)

        loss_density = F.mse_loss(pred, target)

        pred_count = pred.sum(dim=(1,2,3))
        if gt_count is not None:
            gt_count = gt_count.to(pred_count.dtype).to(pred_count.device)
            loss_count = F.l1_loss(pred_count, gt_count)
        else:
            true_count_from_map = target.sum(dim=(1,2,3))
            loss_count = F.l1_loss(pred_count, true_count_from_map)

        return self.w_density * loss_density + self.w_count * loss_count
    


class RegressionLoss(nn.Module):
    def __init__(self, loss_type: str = "huber", huber_delta: float = 5.0, reduction: str = "mean"):
        """
        loss_type: "huber" or "poisson"
        huber_delta: δ parameter for SmoothL1Loss
        reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.huber_delta = float(huber_delta)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, gt_count= None) -> torch.Tensor:
        target = target.to(pred.dtype)

        if self.loss_type == "poisson":
            # Expect pred to be positive λ (e.g., Softplus in the model)
            return F.poisson_nll_loss(
                pred, target,
                log_input=False,  # we pass λ, not log λ
                full=True,        # includes Stirling term for large counts
                reduction=self.reduction
            )

        # Default: Smooth L1 (Huber)
        return F.smooth_l1_loss(
            pred, target,
            beta=self.huber_delta,
            reduction=self.reduction
        )
    


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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, gt_count=None) -> torch.Tensor:
        return self.wf * self.focal(logits, targets) + self.wd * self.dice(logits, targets)
    

def select_loss(train_cfg):
    if MODEL_NAME == "Unet":
        criterion = CompoundSegLoss()
    elif MODEL_NAME == "ViT_Count":
        criterion = RegressionLoss()
    elif MODEL_NAME == "ConvNeXt_Count":
        criterion = RegressionLoss()
    elif MODEL_NAME == "UNetDensity":
        criterion = CellCountingLoss()
    elif MODEL_NAME == 'DeepLabDensity':
        criterion = CellCountingLoss()
    elif MODEL_NAME == "MicroCellUNet":
        criterion = CellCountingLoss()
    elif MODEL_NAME == 'CNNTransformerCounter':
        criterion = CountLoss()

    return criterion