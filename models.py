import os
import json
import torch
from torch import nn
import torchvision
import torch.nn.functional as F


from utils.constants import MODEL_NAME, TIME
from segmentation_models_pytorch import Unet
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.models.segmentation as segmentation
from torchvision.models import vit_b_16, ViT_B_16_Weights, convnext_small, ConvNeXt_Small_Weights
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet50
from transformers import ViTModel
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder



def set_parameter_requires_grad(model, feature_extracting=True):
    """
    Sets the `requires_grad` attribute for all parameters in the model.

    This function is used to control whether the model is trained in a 
    feature extraction mode (freezing the base weights) or a fine-tuning mode.

    Args:
        model (nn.Module): The model whose parameters are to be set.
        feature_extracting (bool): If True, freeze all parameters 
            (i.e., set requires_grad to False). If False, unfreeze all 
            parameters for full fine-tuning.
    """
    # approach 1
    if feature_extracting:
        # frozen model
        model.requires_grad_(False)
    else:
        # fine-tuning
        model.requires_grad_(True)



def calculate_trainable(model):
    """
    Calculates and prints the total size (in MB) and the number of 
    trainable parameters of the given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
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






# -------------------------------------------------------------------------------- #

def load_maskrcnn_ResNet50_model(num_classes: int):
    """
    Loads and customizes a pre-trained Mask R-CNN model with a ResNet50-FPN backbone 
    for instance segmentation.

    Architecture Explanation:
    Mask R-CNN is a two-stage instance segmentation model:
    1. **Stage 1 (RPN):** A Region Proposal Network (RPN) is used to scan the image 
       and propose candidate object bounding boxes.
    2. **Stage 2 (Heads):** The proposed boxes are sent to three parallel heads:
       - **Box Head:** Predicts the final class label (classification) and precise 
         bounding box (regression).
       - **Mask Head:** Predicts a high-resolution mask for each instance.
    The backbone is a pre-trained **ResNet50** paired with a **Feature Pyramid Network (FPN)** for multi-scale feature extraction.

    Customization:
    - **Anchor Generator:** Custom anchor sizes are set post-hoc to better match 
      the expected size of objects (e.g., cells) in the target dataset.
    - **Box Predictor:** The final classification and box regression layer 
      (`FastRCNNPredictor`) is replaced to match the desired `num_classes` (including background).
    - **Mask Predictor:** The final mask segmentation head (`MaskRCNNPredictor`) is 
      also replaced to match `num_classes`.

    Args:
        num_classes (int): The number of output classes (e.g., 2 for cell/background).

    Returns:
        nn.Module: The customized Mask R-CNN model.
    """
    # Load model with pre-trained weights from COCO
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    # --- set smaller anchors post-hoc (5 FPN levels expected) ---
    anchor_generator = AnchorGenerator(
        # Specify one anchor size (8 to 128) for each of the 5 FPN levels
        sizes=((8,), (16,), (32,), (64,), (128,)), 			# one size per FPN level
        # Use 3 aspect ratios (0.5, 1.0, 2.0) at every FPN level
        aspect_ratios=((0.5, 1.0, 2.0),) * 5 			# replicate per level
    )
    model.rpn.anchor_generator = anchor_generator

    # Replace heads for your num_classes (incl. background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the final classification/regression layer
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the final mask prediction layer
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def load_unet_model(num_classes: int = 2,
               in_channels: int = 3,
               encoder_name: str = "resnet34",
               encoder_weights: str = "imagenet"):
    """
    Loads a customizable U-Net model from the `segmentation_models_pytorch` library.

    Architecture Explanation:
    U-Net is a classic architecture for semantic segmentation, characterized by 
    its U-shaped structure consisting of a contracting path (encoder) and an 
    expanding path (decoder) with skip connections.
    - **Encoder:** Downsamples the input image to capture context. Here, it uses a 
      pre-trained **ResNet34** backbone.
    - **Decoder:** Upsamples the features to recover spatial resolution.
    - **Skip Connections:** Concatenate encoder features with decoder features at 
      corresponding levels to help the decoder utilize fine-grained spatial information.
    The output is a segmentation map with `num_classes` channels (logits).

    Args:
        num_classes (int): The number of output channels/classes.
        in_channels (int): The number of input channels (e.g., 3 for RGB).
        encoder_name (str): The name of the backbone network (e.g., 'resnet34').
        encoder_weights (str): Pre-trained weights for the encoder (e.g., 'imagenet').

    Returns:
        nn.Module: The configured U-Net model.
    """
    model = Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # return logits (raw scores) for use with standard loss functions
    )

    # Safety: if you ever need to 'replace the last layer' explicitly:
    # (Not strictly necessary because classes=num_classes already sets it.)
    head = getattr(model, "segmentation_head", None)
    if isinstance(head, nn.Sequential) and isinstance(head[0], nn.Conv2d):
        in_ch = head[0].in_channels
        # Explicitly ensures the final convolution matches the number of classes
        head[0] = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)

    return model

# -------------------------------------------------------------------------------- #
### ---------------------------------- Density regression models -------------------------------- ###
# -------------------------------------------------------------------------------- #

class UNetDensity(nn.Module):
    """
    A custom model for density map regression based on a ResNet-FCN architecture.
    
    Architecture Explanation:
    The model uses a pre-trained **FCN (Fully Convolutional Network)** with a **ResNet50** backbone, which is structurally similar to a U-Net but often simpler on the decoder side. 
    It is typically used for semantic segmentation.
    Customization:
    - **Output Layer:** The final classification layer of the FCN is modified from 
      its default number of classes to output a single channel (`1`), representing the 
      cell density map.
    - **Activation:** A ReLU is applied to the output to ensure non-negative density values.
    - **Learnable Scale:** A simple learnable scalar is added to the output to allow 
      the network to fine-tune the overall magnitude of the predicted density map.
    """
    def __init__(self):
        """Initializes the FCN-based density model."""
        super().__init__()
        # Load FCN with ResNet50 backbone, pre-trained on ImageNet/COCO
        self.unet = segmentation.fcn_resnet50(weights=segmentation.FCN_ResNet50_Weights.DEFAULT)
        # Modify the final output layer (classifier[4]) to output 1 channel (density)
        self.unet.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        
        # Learnable scalar to adjust output scale
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """
        Performs the forward pass to predict the density map.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Predicted density map, scaled and non-negative.
        """
        # Apply ReLU to ensure non-negative density output
        density = F.relu(self.unet(x)['out'])  # (N,1,H,W)
        return density * self.scale  # Allow model to learn scale


class DeepLabDensity(nn.Module):
    """
    A custom model for density map regression based on DeepLabV3 with a ResNet101 backbone.
    
    Architecture Explanation:
    **DeepLabV3** is a state-of-the-art semantic segmentation model designed for 
    high-resolution output. Key components are:
    - **ResNet101:** Used as a strong backbone for feature extraction.
    - **ASPP (Atrous Spatial Pyramid Pooling):** Captures multi-scale contextual 
      information by using parallel atrous (dilated) convolutions with different rates.
    Customization:
    - **Output Layer:** The final classification layer is modified to output a single 
      channel (`1`), representing the cell density map.
    - **Activation:** A ReLU is applied to the output to ensure non-negative density values.
    - **Learnable Scale:** A simple learnable scalar is added to the output.
    """
    def __init__(self):
        """Initializes the DeepLabV3-based density model."""
        super().__init__()
        # Load DeepLabV3 with ResNet101 backbone
        self.model = deeplabv3_resnet101(weights='DEFAULT')
        # Modify classifier's final convolution layer to output 1 channel
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """
        Performs the forward pass to predict the density map.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Predicted density map, scaled and non-negative.
        """
        out = self.model(x)['out']
        # Apply ReLU to ensure non-negative density output
        return F.relu(out) * self.scale
    

class MicroAttentionBlock(nn.Module):
    """
    A lightweight block combining Channel and Spatial Attention.

    Architecture Explanation:
    This block implements a form of attention mechanism to refine feature maps:
    1. **Channel Attention:** Uses global average pooling followed by two 1x1 convolutions 
       and a sigmoid to learn an attention vector that weights the channels.
    2. **Spatial Attention:** Uses a 1x1 convolution and a sigmoid to learn an attention 
       map that weights different spatial locations.
    The input features are scaled by the element-wise product of both attention maps.
    """
    def __init__(self, channels):
        """
        Initializes the Micro Attention Block.
        
        Args:
            channels (int): Number of input/output feature channels.
        """
        super().__init__()
        # Channel attention branch: Global Avg Pool -> Conv -> ReLU -> Conv -> Sigmoid
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # Reduces channels for computational efficiency
            nn.Conv2d(channels, max(1, channels // 8), 1),
            nn.ReLU(),
            # Restores channels
            nn.Conv2d(max(1, channels // 8), channels, 1),
            nn.Sigmoid()
        )
        # Spatial attention branch: Conv (1x1) -> Sigmoid
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        """
        Applies both channel and spatial attention to the input feature map.
        
        Args:
            x (torch.Tensor): Input feature tensor.
            
        Returns:
            torch.Tensor: Attended feature tensor.
        """
        # Element-wise multiplication of input with channel and spatial attention masks
        return x * self.channel_att(x) * self.spatial_att(x)

class MicroCellUNet(nn.Module):
    """
    A custom U-Net based model for cell density prediction, incorporating a high-resolution 
    branch and an attention block for enhanced feature fusion.
    
    Architecture Explanation:
    This model extends the standard U-Net concept for cell counting:
    1. **Encoder:** A pre-trained **ResNet50** is used to extract deep, multi-scale features.
    2. **High-Resolution (HR) Branch:** A separate shallow CNN branch processes the input 
       image directly, incorporating a `MicroAttentionBlock` early on, to preserve fine-grained 
       spatial details, which are critical for small cells.
    3. **Decoder:** A standard U-Net decoder, with 'scse' attention type, upsamples the 
       deep features.
    4. **Fusion:** The output of the decoder's final block is bilinearly interpolated and 
       **concatenated** with the features from the HR branch.
    5. **Density Head:** A final CNN block processes the fused features to predict the 
       single-channel density map.
    6. **Count Refiner:** The final feature map from the ResNet50 encoder's last stage is 
       passed through a small MLP (Multi-Layer Perceptron) to predict a scalar sigmoid-activated 
       adjustment factor, which is then used to modulate the final density map.
    """
    def __init__(self):
        """Initializes the MicroCellUNet model."""
        super().__init__()
        self.encoder = get_encoder(
            name='resnet50',
            in_channels=3,
            depth=5,
            weights='imagenet'
        )

        # High-Resolution (HR) branch to retain fine-grained details
        self.hr_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MicroAttentionBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> H/2, W/2 resolution
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.decoder = self._create_decoder()

        # Density prediction head, processes fused features
        self.density_head = nn.Sequential(
            # Input channels are from decoder's final block (16) + HR branch (128)
            nn.Conv2d(16 + 128, 64, 3, padding=1),
            MicroAttentionBlock(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

        # Global feature refiner for count adjustment
        self.count_refiner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512), # 2048 is the channel count of ResNet50's final stage
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.scale = nn.Parameter(torch.tensor(1.0))

    def _create_decoder(self):
        """
        Instantiates the U-Net decoder, handling potential API changes across library versions.
        
        Returns:
            UnetDecoder: The configured decoder module.
        """
        encoder_channels = self.encoder.out_channels
        try:
            return UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                attention_type='scse'
            )
        except TypeError:
            # Fallback for older segmentation_models_pytorch versions
            return UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                center=False,
                attention_type='scse'
            )

    def _decode(self, enc_features):
        """
        Call decoder compatibly across SMP versions.
        
        Args:
            enc_features (tuple/list): Encoder feature maps.
            
        Returns:
            torch.Tensor: Decoded feature map.
        """
        try:
            # Standard call for newer SMP versions
            return self.decoder(*enc_features)
        except TypeError:
            # Fallback for older SMP versions
            return self.decoder(enc_features)

    def forward(self, x):
        """
        Performs the forward pass, combining high-resolution features with deep U-Net features.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Final density map, modulated by the count refiner.
        """
        hr_features = self.hr_branch(x)          # [B,128,H/2,W/2]
        enc_features = self.encoder(x)

        # decoder (API-robust call)
        try:
            # Unpacks the feature tuple/list for the decoder
            dec_features = self.decoder(*enc_features)
        except TypeError:
            # Passes the entire feature list/tuple to the decoder
            dec_features = self.decoder(enc_features)

        if dec_features.shape[-2:] != hr_features.shape[-2:]:
            # Resample decoder output if resolution doesn't match HR branch output (H/2, W/2)
            dec_features = F.interpolate(dec_features, size=hr_features.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate features along the channel dimension
        fused = torch.cat([dec_features, hr_features], dim=1)   # [B,16+128,H/2,W/2]
        density = F.relu(self.density_head(fused))              # [B,1,H/2,W/2]
        if density.shape[-2:] != x.shape[-2:]:
            # Upsample the final density map to match the input image resolution (H, W)
            density = F.interpolate(density, size=x.shape[-2:], mode='bilinear', align_corners=False)  # [B,1,H,W]

        # Use the deepest encoder feature for global count refinement
        count_adjust = self.count_refiner(enc_features[-1]).sigmoid()   # [B,1]
        # Reshape to (B, 1, 1, 1) for element-wise multiplication
        count_adjust = count_adjust.unsqueeze(-1).unsqueeze(-1)         # [B,1,1,1]
        # (optional) keep dtypes aligned:
        count_adjust = count_adjust.to(density.dtype)

        # Modulate the density map by a scale factor and the learned count adjustment
        return density * self.scale * (1 + count_adjust)

    
# -------------------------------------------------------------------------------- #
### ---------------------------------  Direct count regression models  --------------------------------------------- ###
# -------------------------------------------------------------------------------- #

class ImageCountRegressor(nn.Module):
    """
    Directly predicts a scalar count from an image using a Vision Transformer (ViT) 
    or ConvNeXt backbone.

    Architecture Explanation:
    This model treats the counting task as a pure image-to-scalar regression:
    - **Backbone (Encoder):** Uses a pre-trained image classification model 
      (**ViT-B/16** or **ConvNeXt-Small**) as a feature extractor.
    - **Head:** The original classification head is replaced with a custom regression 
      head, typically consisting of one or two linear layers with ReLU activation and 
      Dropout, outputting a single value.
    - **Output Activation:** Applies Softplus (for Poisson loss) or ReLU (for Huber loss) 
      to ensure the predicted count is non-negative.
    """
    def __init__(self, backbone: str = "vit_b_16", loss_type: str = "huber"):
        """
        Initializes the Image Count Regressor.
        
        Args:
            backbone (str): The CNN or Transformer backbone to use ('vit_b_16' or 'convnext_small').
            loss_type (str): Specifies the loss function to be used ('poisson' or 'huber'), 
                which dictates the final output activation function.
        """
        super().__init__()
        self.loss_type = loss_type

        if backbone == "vit_b_16":
            # Load pre-trained ViT
            m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # Input dimension of the final classification head
            d = m.heads.head.in_features
            # Replace the classification head with a regression head
            m.heads.head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(d // 2, 1), # Final output is a single count value
            )
            self.backbone = m

        elif backbone == "convnext_small":
            # Load pre-trained ConvNeXt
            m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            # Input dimension of the final classification head
            in_dim = m.classifier[-1].in_features
            # Replace the classification head with a regression head
            m.classifier = nn.Sequential(
                nn.Flatten(1),                    # Must flatten the feature map before the final Linear layer
                nn.LayerNorm(in_dim, eps=1e-6),
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(in_dim // 2, 1), # Final output is a single count value
            )
            self.backbone = m
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        """
        Performs the forward pass and applies non-negative activation.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Predicted count (scalar).
        """
        y = self.backbone(x).squeeze(1)  # Output is (N, 1), squeeze to (N,)
        if self.loss_type == "poisson":
            # Softplus ensures positive rate for Poisson loss
            y = F.softplus(y) + 1e-6
        else:
            # ReLU ensures non-negative counts for regression losses (e.g., Huber)
            y = F.relu(y)
        return y
    

class CNNTransformerCounter(nn.Module):
    """
    A hybrid model that uses a CNN for initial feature extraction and a Vision Transformer 
    (ViT) encoder for global context aggregation before predicting the count.

    Architecture Explanation:
    This model is a combination designed to leverage the best of both worlds:
    1. **CNN Backbone:** A pre-trained **ResNet50** (excluding the final pooling/head) 
       is used as a powerful feature extractor, outputting a high-dimensional feature map.
    2. **Projection:** A 1x1 convolution reduces the ResNet's 2048 channels to 768, 
       matching the ViT's embedding dimension.
    3. **Tokenization:** The 2D feature map is reshaped and flattened into a sequence 
       of tokens, similar to ViT's patch embedding process.
    4. **ViT Encoder:** A pre-trained **ViT Encoder** is applied to the token sequence 
       (with added CLS and positional tokens) to capture long-range, global dependencies.
    5. **Counting Head:** The resulting global feature (mean-pooled across all tokens) 
       is passed through a small MLP to regress the final scalar count.
    """
    def __init__(self):
        """Initializes the CNN-Transformer hybrid model."""
        super().__init__()
        # 1) CNN backbone: ResNet50 up to the final convolutional block (before AvgPool)
        self.cnn = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2]
        )
        # 2) Project to ViT dim (768)
        self.projection = nn.Conv2d(2048, 768, 1)

        # 3) ViT encoder (pretrained on Google's model)
        self.transformer = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # IMPORTANT: remove this line, it breaks the embedding pipeline:
        # self.transformer.embeddings.patch_embeddings = nn.Identity()

        # 4) Counting head: MLP for final count regression
        self.count_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        # Adaptive pooling to ensure a consistent feature map size (e.g., 14x14) 
        # for tokenization, regardless of input size, matching ViT's base design.
        self.token_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x):
        """
        Performs the forward pass: CNN -> Projection -> Transformer -> Regression.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Predicted count (scalar).
        """
        # CNN features: [B, 2048, H/32, W/32]
        feats = self.cnn(x)

        # Project to ViT dim: [B, 768, H/32, W/32]
        feats = self.projection(feats)

        feats = self.token_pool(feats)       # Pool to a fixed size (e.g., 14x14)

        # Flatten to tokens: [B, N, 768], where N = 14*14 (196)
        B, C, H, W = feats.shape
        # Reshape: (B, C, H*W) -> (B, H*W, C)
        tokens = feats.view(B, C, H * W).permute(0, 2, 1)  # [B, N, 768]
        N = tokens.size(1)

        vit = self.transformer
        # Prepare CLS token: Replicate the learned CLS token for the batch size
        cls_token = vit.embeddings.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x_seq = torch.cat([cls_token, tokens], dim=1)           # [B, N+1, 768]

        # Positional embeddings: Add positional embeddings to the token sequence
        pos_embed = vit.embeddings.position_embeddings  # [1, P, 768]
        if x_seq.size(1) > pos_embed.size(1):
            # Raise an error if the number of tokens exceeds the max positional embedding length
            raise ValueError(f"Token count {x_seq.size(1)} exceeds ViT position table {pos_embed.size(1)}")
        # Slice the positional embeddings to match the current token sequence length
        x_seq = x_seq + pos_embed[:, : x_seq.size(1), :]

        # Dropout from embeddings block
        x_seq = vit.embeddings.dropout(x_seq)

        # Encoder + final LN
        # Pass the prepared sequence through the ViT encoder layers
        enc_out = vit.encoder(
            hidden_states=x_seq,            # using encoder directly, this arg is supported
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state               # [B, N+1, 768]

        # Apply the final Layer Normalization
        x_seq = vit.layernorm(enc_out)    # [B, N+1, 768]

        # Pool: Aggregate all token features to get a global feature vector
        # Using mean pooling across all tokens (including CLS)
        global_feature = x_seq.mean(dim=1)   # [B, 768]
        # Pass the global feature to the regression head
        count = self.count_head(global_feature).squeeze(1)  # [B]
        return count


# -------------------------------------------------------------------------------- #

def get_model():
    """
    Selects and initializes the PyTorch model based on the global MODEL_NAME constant.

    Initializes all parameters to be trainable by default, then calculates and prints
    the model size and parameter count.

    Returns:
        nn.Module: The initialized model.
    """
    if MODEL_NAME == "Mask_R_CNN_ResNet50":
        model = load_maskrcnn_ResNet50_model(num_classes=2)
    elif MODEL_NAME == "Unet":
        model = load_unet_model(num_classes=2)
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
    else:
        # Note: Added for robustness, original script assumed valid MODEL_NAME
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")


    # Set all parameters to be trainable (feature_extracting=False)
    set_parameter_requires_grad(model, feature_extracting=False)
    calculate_trainable(model)

    return model


def save_model(checkpoint, model_config, output_dir):
    """
    Saves the model checkpoint (state dict) and its configuration to disk.
    
    Args:
        checkpoint (dict): Dictionary containing the model state, optimizer state, etc.
        model_config (dict): Dictionary containing the model's training configuration.
        output_dir (str): The base directory where model weights and config should be saved.
    """
    timestamp = TIME
    model_weights_path = os.path.join(output_dir, 'model_weights')

    # Creates the directory if it doesn't already exist
    os.makedirs(model_weights_path, exist_ok=True)

    # Saves the model checkpoint with a unique timestamp
    checkpoint_path = os.path.join(model_weights_path, f"best_model_{timestamp}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save the model's configuration as well
    config_save_path = os.path.join(model_weights_path, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    print(f"Model and configuration saved at {model_weights_path}")
    print("Timestamp: ", timestamp)


def load_model(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads model weights, optimizer, and scheduler states from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the saved PyTorch checkpoint file (.pt).
        model (nn.Module): The model instance to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load the state into. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler instance to load the state into. Defaults to None.
        device (str, optional): The device (e.g., "cuda:0" or "cpu") to map the loaded tensor to. Defaults to "cpu".

    Returns:
        tuple: (model, optimizer, scheduler, epoch, train_loss, val_loss)
    """
    # Load the checkpoint file, mapping tensors to the specified device
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load model weights (strict=True by default; keep for safety)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if missing or unexpected:
        # Print a warning if the loaded state dictionary does not perfectly match the model
        print("State dict mismatch:",
              f"\n  missing: {missing}\n  unexpected: {unexpected}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Extract training progress metrics (epoch and losses), defaulting to 0 if not present
    epoch = ckpt.get("epoch", 0)
    train_loss = ckpt.get("train_loss")
    val_loss = ckpt.get("val_loss")

    print(f"Loaded model from {checkpoint_path} (epoch {epoch})")
    return model, optimizer, scheduler, epoch, train_loss, val_loss