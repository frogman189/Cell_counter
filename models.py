import torchvision
import os
import torch
from torch import nn
import torch.nn.functional as F
import json

from utils.constants import MODEL_NAME, TIME
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from segmentation_models_pytorch import Unet
from ultralytics import YOLO
from instanseg import InstanSeg
#from detectron2 import model_zoo
#from detectron2.config import get_cfg
#from detectron2.modeling import build_model
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision

def set_parameter_requires_grad(model, feature_extracting=True):
    # approach 1
    if feature_extracting:
        # frozen model
        model.requires_grad_(False)
    else:
        # fine-tuning
        model.requires_grad_(True)


def calculate_trainable(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"model size: {size_all_mb:.2f} MB") 

    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"model trainable params: {num_trainable_params}")



# def load_maskrcnn_ResNet50_model(num_classes):
#     # Load a pre-trained Mask R-CNN and adapt for custom classes
#     weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # or DEFAULT
#     model = maskrcnn_resnet50_fpn(weights=weights)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

#     return model
def load_maskrcnn_ResNet50_model(num_classes: int):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    # Smaller anchors help nuclei/cell scales; tune sizes to your pixel scale
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128),),           # try (8,16,32,64) if cells are tiny
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.05,               # keep low during training; raise at eval
        trainable_backbone_layers=3          # 3–5; more layers = more finetuning, more compute
    )

    # Replace heads for your num_classes (incl. background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

# def load_unet_model(num_classes=2, pretrained=True):
#     if pretrained:
#         weights = DeepLabV3_ResNet50_Weights.DEFAULT
#         model = deeplabv3_resnet50(weights=weights)
#     else:
#         model = deeplabv3_resnet50(weights=None)

#     # Replace the classifier head
#     model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

#     # Optional: Freeze backbone for fine-tuning
#     if pretrained:
#         for param in model.backbone.parameters():
#             param.requires_grad = False

#     return model

def load_unet_model(num_classes: int = 2,
               in_channels: int = 3,
               encoder_name: str = "resnet34",
               encoder_weights: str = "imagenet"):
    model = Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # return logits
    )

    # Safety: if you ever need to 'replace the last layer' explicitly:
    # (Not strictly necessary because classes=num_classes already sets it.)
    head = getattr(model, "segmentation_head", None)
    if isinstance(head, nn.Sequential) and isinstance(head[0], nn.Conv2d):
        in_ch = head[0].in_channels
        head[0] = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)

    return model


# --- ADD: imports at top of models.py ---
from torchvision.models import vit_b_16, ViT_B_16_Weights, convnext_small, ConvNeXt_Small_Weights
import torch.nn.functional as F
import torch.nn as nn

# --- ADD: image-level count regressor ---
class ImageCountRegressor(nn.Module):
    """
    Image -> scalar count.
    backbone: 'vit_b_16' or 'convnext_small'
    loss_type only affects forward’s activation (poisson uses softplus, huber uses relu).
    """
    def __init__(self, backbone: str = "vit_b_16", loss_type: str = "huber"):
        super().__init__()
        self.loss_type = loss_type

        if backbone == "vit_b_16":
            m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            d = m.heads.head.in_features
            m.heads.head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(d // 2, 1),
            )
            self.backbone = m

        elif backbone == "convnext_small":
            m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            in_dim = m.classifier[-1].in_features
            m.classifier = nn.Sequential(
                m.classifier[0],                 # LayerNorm
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(in_dim // 2, 1),
            )
            self.backbone = m
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        y = self.backbone(x).squeeze(1)  # (N,)
        if self.loss_type == "poisson":
            # positive rate for Poisson
            y = F.softplus(y) + 1e-6
        else:
            # keep non-negative counts for regression
            y = F.relu(y)
        return y

#Fine tune Yolo guide: https://docs.ultralytics.com/tasks/segment/ - maybe aviad can take this personly, can teach alot.
#Important: expected input size: (B, 3, 640, 640)
def load_yolov8_seg_model(num_classes=1, model_size="yolov8s-seg.pt"):
    # Load pre-trained YOLOv8 segmentation model
    model = YOLO(model_size)  # 'yolov8n-seg.pt', 'yolov8s-seg.pt', etc.
    
    # Update class count (1 class: "cell")
    model.model.nc = num_classes
    
    # Optional: Freeze backbone (YOLOv8-specific)
    for name, param in model.model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
    
    return model

# Intro to detctron2 - https://detectron2.readthedocs.io/en/latest/
# def load_detectron2(model_name="mask_rcnn_R_50_FPN_3x.yaml", num_classes=1):
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{model_name}"))
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{model_name}")
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 1 class (cell)
    
#     # Freeze backbone (optional)
#     cfg.MODEL.BACKBONE.FREEZE_AT = 2
    
#     model = build_model(cfg)
#     return model, cfg


def get_model():
    if MODEL_NAME == "Mask_R_CNN_ResNet50":
        model = load_maskrcnn_ResNet50_model(num_classes=2)
    elif MODEL_NAME == "Unet":
        model = load_unet_model(num_classes=2)
    elif MODEL_NAME == "YOLOv8":
        model = load_yolov8_seg_model(num_classes=1)
    elif MODEL_NAME == "ViT_Count":
        return ImageCountRegressor(backbone="vit_b_16", loss_type="huber") #model_args.get("loss_type", "huber"))
    elif MODEL_NAME == "ConvNeXt_Count":
        return ImageCountRegressor(backbone="convnext_small", loss_type="huber") #model_args.get("loss_type", "huber"))


    set_parameter_requires_grad(model, feature_extracting=False)
    calculate_trainable(model)

    return model


def save_model(checkpoint, output_dir): #, model_config, log_file_path
    timestamp = TIME
    model_weights_path = os.path.join(output_dir, 'model_weights')

    os.makedirs(model_weights_path, exist_ok=True)

    checkpoint_path = os.path.join(model_weights_path, f"best_model_{timestamp}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # # Save the model's configuration as well
    # config_save_path = os.path.join(model_weights_path, 'config.json')
    # with open(config_save_path, 'w') as f:
    #     json.dump(model_config, f, indent=4)

    print(f"Model and configuration saved at {model_weights_path}")
    print("Timestamp: ", timestamp)


def load_model(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Loads model, optimizer, scheduler, and training state from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', None)
    val_loss = checkpoint.get('val_loss', None)

    print(f"Loaded model from {checkpoint_path} (epoch {epoch})")

    return model, optimizer, scheduler, epoch, train_loss, val_loss