import torchvision
import os
import torch
from torch import nn
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



def load_maskrcnn_ResNet50_model(num_classes):
    # Load a pre-trained Mask R-CNN and adapt for custom classes
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # or DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def load_unet_model(num_classes=2, pretrained=True):
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    else:
        model = deeplabv3_resnet50(weights=None)

    # Replace the classifier head
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # Optional: Freeze backbone for fine-tuning
    if pretrained:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model

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