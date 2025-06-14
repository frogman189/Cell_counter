import numpy as np
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision
from utils.constants import MODEL_NAME, TIME
import os
import torch
import json

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



def get_ResNet50_model(num_classes):
    # Load a pre-trained Mask R-CNN and adapt for custom classes
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # or DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_model():
    if MODEL_NAME == "ResNet50":
        model = get_ResNet50_model(num_classes=2)

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