import os
import json
import torch
from torch import nn
import torchvision
import torch.nn.functional as F


from utils.constants import MODEL_NAME, TIME
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from segmentation_models_pytorch import Unet
from ultralytics import YOLO
from instanseg import InstanSeg
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.models.segmentation as segmentation
from torchvision.models import vit_b_16, ViT_B_16_Weights, convnext_small, ConvNeXt_Small_Weights
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet50
from transformers import ViTModel
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder



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

### ---------------------------------- Density regression models -------------------------------- ###

class UNetDensity(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = segmentation.fcn_resnet50(weights=segmentation.FCN_ResNet50_Weights.DEFAULT)
        self.unet.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        
        # Learnable scalar to adjust output scale
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        density = F.relu(self.unet(x)['out'])  # (N,1,H,W)
        return density * self.scale  # Allow model to learn scale



class DeepLabDensity(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_resnet101(weights='DEFAULT')
        # Modify classifier
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        out = self.model(x)['out']
        return F.relu(out) * self.scale
    



class MicroAttentionBlock(nn.Module):
    """Multi-scale attention block for micro-cell features"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * channel_att * spatial_att



class MicroCellUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Encoder with ResNet50 (unchanged)
        self.encoder = get_encoder(
            name='resnet50',
            in_channels=3,
            depth=5,
            weights='imagenet'
        )
        
        # 2. High-resolution branch (unchanged)
        self.hr_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MicroAttentionBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 3. Fixed Decoder Initialization
        self.decoder = self._create_decoder()
        
        # 4. Original prediction heads (unchanged)
        self.density_head = nn.Sequential(
            nn.Conv2d(16 + 128, 64, 3, padding=1),
            MicroAttentionBlock(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
        
        self.count_refiner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _create_decoder(self):
        """Version-compatible decoder creation"""
        
        
        # Get encoder channels from the encoder
        encoder_channels = self.encoder.out_channels
        
        # Try modern SMP version first
        try:
            return UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                attention_type='scse'
            )
        # Fallback for older versions
        except TypeError:
            return UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                center=False,
                attention_type='scse'
            )

    def forward(self, x):
        # Original forward pass (unchanged)
        hr_features = self.hr_branch(x)
        enc_features = self.encoder(x)
        dec_features = self.decoder(*enc_features)
        
        dec_features = F.interpolate(dec_features, scale_factor=2, mode='bilinear')
        fused = torch.cat([dec_features, hr_features], dim=1)
        
        density = F.relu(self.density_head(fused))
        count_adjust = self.count_refiner(enc_features[-1]).sigmoid()
        
        return density * self.scale * (1 + count_adjust)






    


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



    


### ---------------------------------  Direct count regression models  --------------------------------------------- ###

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
    




class CNNTransformerCounter(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) CNN backbone
        self.cnn = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2]
        )
        # 2) Project to ViT dim
        self.projection = nn.Conv2d(2048, 768, 1)

        # 3) ViT encoder (keep embeddings, but we won't call patch_embeddings)
        self.transformer = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # IMPORTANT: remove this line, it breaks the embedding pipeline:
        # self.transformer.embeddings.patch_embeddings = nn.Identity()

        # 4) Counting head
        self.count_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.token_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x):
        # CNN features: [B, 2048, H/32, W/32]
        feats = self.cnn(x)

        # Project to ViT dim: [B, 768, H/32, W/32]
        feats = self.projection(feats)

        feats = self.token_pool(feats)       # [B, 768, 14, 14]

        # Flatten to tokens: [B, N, 768], where N = (H/32)*(W/32)
        B, C, H, W = feats.shape
        tokens = feats.view(B, C, H * W).permute(0, 2, 1)  # [B, N, 768]
        N = tokens.size(1)

        vit = self.transformer
        # Prepare CLS token
        cls_token = vit.embeddings.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x_seq = torch.cat([cls_token, tokens], dim=1)           # [B, N+1, 768]

        # Positional embeddings: ViT position table is [1, 197, 768] for 224x224 (14x14+1)
        # We slice to the needed length (N+1). Ensure N+1 <= pos_embed_len.
        pos_embed = vit.embeddings.position_embeddings  # [1, P, 768]
        if x_seq.size(1) > pos_embed.size(1):
            # If your token count exceeds ViT’s pos-embed length, interpolate (rare here).
            # Simple fallback: interpolate in 2D grid space (recommended to implement if needed).
            raise ValueError(f"Token count {x_seq.size(1)} exceeds ViT position table {pos_embed.size(1)}")
        x_seq = x_seq + pos_embed[:, : x_seq.size(1), :]

        # Dropout from embeddings block
        x_seq = vit.embeddings.dropout(x_seq)

        # Encoder + final LN
        enc_out = vit.encoder(
            hidden_states=x_seq,            # using encoder directly, this arg is supported
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state               # [B, N+1, 768]

        x_seq = vit.layernorm(enc_out)    # [B, N+1, 768]

        # Pool (you can also use the CLS token: x_seq[:, 0])
        global_feature = x_seq.mean(dim=1)   # [B, 768]
        count = self.count_head(global_feature).squeeze(1)  # [B]
        return count



def get_model():
    if MODEL_NAME == "Mask_R_CNN_ResNet50":
        model = load_maskrcnn_ResNet50_model(num_classes=2)
    elif MODEL_NAME == "Unet":
        model = load_unet_model(num_classes=2)
    elif MODEL_NAME == "YOLOv8":
        model = load_yolov8_seg_model(num_classes=1)
    elif MODEL_NAME == "ViT_Count":
        model = ImageCountRegressor(backbone="vit_b_16", loss_type="huber") 
    elif MODEL_NAME == "ConvNeXt_Count":
        model = ImageCountRegressor(backbone="convnext_small", loss_type="huber")
    elif MODEL_NAME == "UNetDensity":
        model = UNetDensity()
    elif MODEL_NAME == 'DeepLabDensity':
        model = DeepLabDensity()
    elif MODEL_NAME == "MicroCellUNet":
        model = MicroCellUNet()
    elif MODEL_NAME == 'CNNTransformerCounter':
        model = CNNTransformerCounter()


    set_parameter_requires_grad(model, feature_extracting=False)
    calculate_trainable(model)

    return model


def save_model(checkpoint, model_config, output_dir):
    timestamp = TIME
    model_weights_path = os.path.join(output_dir, 'model_weights')

    os.makedirs(model_weights_path, exist_ok=True)

    checkpoint_path = os.path.join(model_weights_path, f"best_model_{timestamp}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save the model's configuration as well
    config_save_path = os.path.join(model_weights_path, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    print(f"Model and configuration saved at {model_weights_path}")
    print("Timestamp: ", timestamp)



def load_model(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load model weights (strict=True by default; keep for safety)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if missing or unexpected:
        print("State dict mismatch:",
              f"\n  missing: {missing}\n  unexpected: {unexpected}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    train_loss = ckpt.get("train_loss")
    val_loss = ckpt.get("val_loss")

    print(f"Loaded model from {checkpoint_path} (epoch {epoch})")
    return model, optimizer, scheduler, epoch, train_loss, val_loss