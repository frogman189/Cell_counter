import os, glob, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from cellpose import models
import cellpose
from importlib.metadata import version as _pkgver
try:
    # Try to get the version of the installed 'cellpose' package
    _CPVER = _pkgver("cellpose")
except Exception:
    _CPVER = "0.0.0"
# Extract the major version number for API compatibility checks
_CP_MAJOR = int(_CPVER.split(".")[0]) if _CPVER and _CPVER[0].isdigit() else 0


from utils.metrics import calculate_counting_metrics, print_metrics
from preprocess import prepare_dataset, LiveCellDataset
from utils.constants import MODEL_NAME, dataset_paths, DEVICE

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the 'centermask2' directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "centermask2"))

from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from centermask2.centermask.config import get_cfg

# ================== CONFIG ==================
EVAL_MODEL = "lacss"        # "lacss" or "cellpose" or "centermask2"
TEST = True                 # True => evaluate on test split, else val

# LACSS settings
LACSS_ID_OR_PATH = os.path.join(BASE_DIR, "benchmark_models", "lacss3-small-l")
LACSS_RESHAPE_TO = None
LACSS_MIN_AREA = 1.0
LACSS_SCORE_THR = 0.001
LACSS_SEG_THR = 0.001

# Cellpose settings
CP_MODE = "grayscale"       # "grayscale" or "rgb"
CP_MODEL_TYPE = "cyto"      # "cyto","nuclei","cyto2","cyto3","cpsam"
CP_DIAMETER = None          # e.g., 20 (pixels) or None for auto
CP_USE_GPU = True
BATCH_VERBOSE = False

# CenterMask2 (LIVECell-trained) settings
CM2_CONFIG_PATH = os.path.join(BASE_DIR, "benchmark_models", "livecell_anchor_free_config.yaml")
CM2_WEIGHTS_PATH = os.path.join(BASE_DIR, "benchmark_models", "LIVECell_anchor_free_model.pth")
CM2_SCORE_THR = 0.05            # test-time score threshold
CM2_MAX_DETS = 3000             # VERY IMPORTANT for LIVECell density


# ================== DATALOADER HELPERS ==================
def collate_fn(batch):
    """
    Custom collate function for the DataLoader to process batches from LiveCellDataset.

    It stacks images and density maps into single tensors and converts cell counts to a float tensor.

    Args:
        batch (List[Tuple]): A list of tuples, where each tuple contains 
                             (image, density_map, cell_count) for a single sample.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The batched images, density maps, and cell counts.
    """
    images, density_maps, cell_counts = zip(*batch)
    images = torch.stack(images, 0)
    density_maps = torch.stack(density_maps, 0)  # Density maps are kept as float tensors
    cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
    return images, density_maps, cell_counts


def make_loader(test_split: bool):
    """
    Creates and returns a DataLoader for the validation or test split of the LiveCell dataset.

    Args:
        test_split (bool): If True, creates the DataLoader for the 'test' split; 
                           otherwise, creates it for the 'val' split.

    Returns:
        torch.utils.data.DataLoader: The configured DataLoader instance.
    """
    # Load and process dataset metadata
    dataset_dict = prepare_dataset(
        dataset_paths['path_to_original_dataset'],
        dataset_paths['path_to_livecell_images'],
        dataset_paths['path_to_labels'],
    )
    split_name = 'test' if test_split else 'val'
    # Set image size based on the model (224 for ViT/ConvNeXt, 512 otherwise)
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    
    # Create the dataset instance
    dataset = LiveCellDataset(dataset_dict[split_name], img_size=img_size, normalize=False)
    
    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False, # Shuffle is set to False for consistent evaluation
        collate_fn=collate_fn,
        num_workers=8,
        # Pin memory only if a CUDA device is available to speed up data transfer
        pin_memory=True if DEVICE == 'cuda' else False,
    )
    return loader


# ================== IMAGE UTILS ==================
def to_hw_or_hwc(img_t: torch.Tensor,
                 to_float01: bool = False,
                 ensure_3ch: bool | None = None,
                 debug: bool = False) -> np.ndarray:
    """
    Converts a [C,H,W] PyTorch tensor image (in [0,1] or [0,255]) to a NumPy array 
    in the required (H,W) or (H,W,C) format and range.

    Args:
        img_t (torch.Tensor): Input image tensor of shape [C, H, W].
        to_float01 (bool, optional): If True, output is float32 in [0,1]. 
                                     If False, output is uint8 in [0,255]. Defaults to False.
        ensure_3ch (bool | None, optional): If True, a 1-channel image is repeated to 3 channels (H,W,3).
                                            If None/False, shape is (H,W) for 1-channel. Defaults to None.
        debug (bool, optional): If True, prints debug information about the output array. Defaults to False.

    Returns:
        np.ndarray: The converted image array of shape (H,W) or (H,W,C).
    """
    x = img_t.detach().cpu().to(torch.float32)      # [C,H,W] float32
    C, H, W = x.shape

    # --- scale to desired range ---
    x_min, x_max = float(x.min()), float(x.max())
    # Heuristic to detect [0, 255] vs [0, 1] range based on max value
    is_0_255_range = x_max > 1.0 + 1e-6 

    if to_float01:
        # Target: [0,1] float32
        if is_0_255_range:
            x = x / 255.0
        x = x.clamp_(0.0, 1.0) # Clamp ensures bounds
    else:
        # Target: [0,255] uint8
        if not is_0_255_range:
            x = x * 255.0
        x = x.clamp_(0.0, 255.0) # Clamp ensures bounds

    # --- channel handling ---
    if ensure_3ch:
        if C == 1:
            x = x.repeat(3, 1, 1)        # Make 3-channel by repeating the single channel
            C = 3
        elif C > 3:
            x = x[:3]                    # Drop extra channels if more than 3

    if C == 1:
        out = x[0]                      # (H,W)
    else:
        # Permute from [C,H,W] to [H,W,C] for NumPy/Matplotlib compatibility
        out = x[:3].permute(1, 2, 0)      # (H,W,3)

    # --- dtype cast for output ---
    if to_float01:
        out_np = out.numpy().astype(np.float32)      # [0,1]
    else:
        out_np = out.numpy().astype(np.uint8)        # [0,255]

    if debug:
        mn, mx = float(out_np.min()), float(out_np.max())
        print(f"[to_hw_or_hwc] shape={out_np.shape}, dtype={out_np.dtype}, range=({mn:.4f},{mx:.4f})")

    return out_np


def _to_cellpose_img_and_channels(img_t: torch.Tensor, mode: str):
    """
    Converts a [C,H,W] PyTorch tensor to the NumPy format and channel arguments 
    required by the Cellpose `eval` function.

    Args:
        img_t (torch.Tensor): Input image tensor of shape [C, H, W].
        mode (str): The Cellpose processing mode: 'grayscale' or 'rgb'.

    Returns:
        Tuple[np.ndarray, List[int]]: A tuple containing the converted NumPy image 
                                      ((H,W) or (H,W,3)) and the channels tuple/list for Cellpose.
    
    Raises:
        ValueError: If `mode` is neither 'grayscale' nor 'rgb'.
    """
    img = img_t.detach().cpu().float()
    # Scale tensor to [0, 255] if it appears to be in [0, 1]
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.clamp(0, 255)

    if mode == 'grayscale':
        if img.ndim == 3 and img.shape[0] > 1:
            # Convert multi-channel image to grayscale using standard ITU-R BT.601 luminance formula
            r, g, b = img[0], img[1], img[2]
            img = 0.299*r + 0.587*g + 0.114*b
        else:
            # Handle single channel (already or only one channel available)
            img = img[0] if img.ndim == 3 else img
        np_img = img.numpy().astype(np.uint8)    # (H,W)
        channels = [0, 0]                        # Cytoplasm from channel 0, no nucleus channel (Cellpose convention)
    elif mode == 'rgb':
        if img.shape[0] == 1:
            # Repeat single channel to 3 channels for RGB
            img = img.repeat(3, 1, 1)
        # Permute to (H,W,C) for NumPy and cast to uint8
        np_img = img.numpy().transpose(1, 2, 0).astype(np.uint8)  # (H,W,3)
        channels = [2, 1]  # (R as cytoplasm, G as nucleus) — a common choice for Cellpose
    else:
        raise ValueError("mode must be 'grayscale' or 'rgb'")
    return np_img, channels


def _rgb_to_bgr(np_img):
    """
    Converts a NumPy image from RGB (H,W,C) to BGR (H,W,C) format 
    (required by Detectron2/CenterMask2 when using an RGB input).

    Args:
        np_img (np.ndarray): Input image array (H,W) or (H,W,3).

    Returns:
        np.ndarray: The converted image in BGR format, or the original image if grayscale.
    """
    # Detectron2 expects BGR uint8 (OpenCV style)
    if np_img.ndim == 2:
        return np_img  # grayscale is acceptable as-is
    # Reverse the channel order (RGB -> BGR)
    return np_img[:, :, ::-1]


# ================== ROBUST LACSS IMPORTS/BUILDER ==================
def _import_lacss_predictor_cls():
    """
    Attempts to robustly import the LACSS Predictor/Inferer class, 
    checking multiple possible locations due to potential API changes across versions.

    Returns:
        Tuple[Type, str]: The imported predictor class and a string indicating its origin/flavor.

    Raises:
        ImportError: If no compatible LACSS predictor class can be found.
    """
    import importlib
    tried = []

    # Try lacss.deploy.Predictor
    try:
        mod = importlib.import_module("lacss.deploy")
        if hasattr(mod, "Predictor"):
            return getattr(mod, "Predictor"), "deploy.Predictor"
        tried.append("lacss.deploy.Predictor")
    except Exception as e:
        tried.append(f"lacss.deploy.Predictor ! {type(e).__name__}")

    # Try lacss.deploy.predict.Predictor
    try:
        mod = importlib.import_module("lacss.deploy.predict")
        if hasattr(mod, "Predictor"):
            return getattr(mod, "Predictor"), "deploy.predict.Predictor"
        tried.append("lacss.deploy.predict.Predictor")
    except Exception as e:
        tried.append(f"lacss.deploy.predict.Predictor ! {type(e).__name__}")

    # Try lacss.deploy.inference.Inferer
    try:
        mod = importlib.import_module("lacss.deploy.inference")
        if hasattr(mod, "Inferer"):
            return getattr(mod, "Inferer"), "deploy.inference.Inferer"
        tried.append("lacss.deploy.inference.Inferer")
    except Exception as e:
        tried.append(f"lacss.deploy.inference.Inferer ! {type(e).__name__}")

    raise ImportError("Could not find a LACSS Predictor class. Tried: " + " | ".join(tried))


def build_lacss_predictor(id_or_path: str, **kwargs):
    """
    Constructs a LACSS predictor instance, trying multiple ways to load a model 
    (alias, direct path, or path to a directory).

    Args:
        id_or_path (str): Can be a model alias (e.g., 'lacss3-small-l'), a path to a model file, 
                          or a path to a directory containing a model file.
        **kwargs: Additional keyword arguments passed to the predictor's constructor.

    Returns:
        Object: The initialized LACSS predictor instance.

    Raises:
        ValueError: If the predictor instance cannot be successfully constructed 
                    from the provided path/ID.
    """
    Predictor, flavor = _import_lacss_predictor_cls()

    def _try_ctor_for(path, **ctor_kwargs):
        """Helper to try various constructor signatures for the Predictor class."""
        # Try common constructor signatures with different keyword names for the path argument
        for sig in [
            {"model_path": path, **ctor_kwargs},
            {"ckpt_path": path, **ctor_kwargs},
            {"weights": path, **ctor_kwargs},
            {"path": path, **ctor_kwargs},
            {"url": path, **ctor_kwargs},
        ]:
            try:
                return Predictor(**sig)
            except TypeError:
                continue # Skip to next signature if keywords don't match
            except Exception:
                # If it's not clearly a signature issue, keep trying others
                continue
        # Try positional-only path
        try:
            return Predictor(path, **ctor_kwargs)
        except Exception:
            return None

    # 1) Try as given (might be an alias or direct file path)
    pred = _try_ctor_for(id_or_path, **kwargs)
    if pred is not None:
        return pred

    # 2) If it's a directory, try files inside with common extensions
    if os.path.isdir(id_or_path):
        candidates = []
        # Check for common model file extensions
        for pat in ("*.npz", "*.ckpt", "*.pt", "*.pth", "*"):
            candidates.extend(glob.glob(os.path.join(id_or_path, pat)))
        for c in candidates:
            pred = _try_ctor_for(c, **kwargs)
            if pred is not None:
                return pred

    # 3) Try appending common extensions to the ID/base path
    base = id_or_path.rstrip("/\\")
    for ext in (".npz", ".ckpt", ".pt", ".pth"):
        path = base + ext
        if os.path.exists(path):
            pred = _try_ctor_for(path, **kwargs)
            if pred is not None:
                return pred

    # 4) Final attempt with minimal arguments
    pred = _try_ctor_for(id_or_path)
    if pred is not None:
        return pred

    raise ValueError(f"Could not construct LACSS predictor for '{id_or_path}' using API {flavor}")


# ================== LACSS INFERENCE WRAPPERS ==================
def _predict_single(predictor, img_np: np.ndarray,
                    output_type: str = "label",
                    reshape_to=None,
                    min_area: float = 0.0,
                    score_threshold: float = 0.5,
                    segmentation_threshold: float = 0.5):
    """
    Runs prediction on a single image using the LACSS predictor, trying different 
    predict methods/signatures across LACSS versions for compatibility.

    Args:
        predictor (Object): The initialized LACSS predictor instance.
        img_np (np.ndarray): The input image array.
        output_type (str, optional): Desired output type from the predictor (e.g., "label"). 
                                     Defaults to "label".
        reshape_to (Any, optional): Reshape argument for the predictor. Defaults to None.
        min_area (float, optional): Minimum area threshold for detected objects. Defaults to 0.0.
        score_threshold (float, optional): Confidence score threshold for detected objects. Defaults to 0.5.
        segmentation_threshold (float, optional): Pixel-level segmentation threshold. Defaults to 0.5.

    Returns:
        Dict | np.ndarray: The raw output from the LACSS predictor.
        
    Raises:
        RuntimeError: If no compatible prediction method is found on the predictor instance.
    """
    # Try the most common `predict` signature with all kwargs
    try:
        return predictor.predict(
            img_np,
            output_type=output_type,
            reshape_to=reshape_to,
            min_area=min_area,
            score_threshold=score_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_iou=1.0, # Pass default value
            normalize=True, # Pass default value
        )
    except AttributeError:
        pass
    except TypeError:
        # Try simpler signature (only image and output_type)
        try:
            return predictor.predict(img_np, output_type=output_type)
        except Exception:
            pass

    # Try `predict_on_large_image` for large input mode
    try:
        return predictor.predict_on_large_image(
            img_np,
            output_type=output_type,
            reshape_to=reshape_to,
            min_area=min_area,
            score_threshold=score_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_iou=0.0, # Pass default value
        )
    except AttributeError:
        pass
    except TypeError:
        # Try simpler signature (only image and output_type)
        try:
            return predictor.predict_on_large_image(img_np, output_type=output_type)
        except Exception:
            pass

    # Try __call__ (if the predictor is callable)
    try:
        return predictor(img_np) 
    except Exception:
        # Try .infer() (if an Inferer class was imported)
        try:
            return predictor.infer(img_np)
        except Exception:
            pass

    raise RuntimeError("No compatible LACSS prediction method found on this predictor instance.")


# ================== ROBUST CenterMask2 IMPORTS/BUILDER ==================

def build_centermask2_predictor(cfg_path: str, weights_path: str, score_thr: float = 0.05, max_dets: int = 3000, device: str = DEVICE):
    """
    Configures and initializes a Detectron2 DefaultPredictor for CenterMask2 
    using a specified configuration and weights file.

    Args:
        cfg_path (str): Path to the CenterMask2 configuration YAML file.
        weights_path (str): Path to the model checkpoint weights file (.pth).
        score_thr (float, optional): Test-time confidence score threshold. Defaults to 0.05.
        max_dets (int, optional): Maximum number of detections per image. Defaults to 3000.
        device (str, optional): The torch device ('cuda' or 'cpu') to run the model on. 
                                Defaults to the global DEVICE.

    Returns:
        detectron2.engine.DefaultPredictor: The initialized Detectron2 predictor.
    """
    cfg = get_cfg()                      # Get default configuration from CenterMask2
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thr) # Set score threshold
    cfg.INPUT.MASK_FORMAT = "bitmask" # Standard mask format
    cfg.TEST.DETECTIONS_PER_IMAGE = int(max_dets) # Set max detections
    # Determine the actual device string for Detectron2 configuration
    cfg.MODEL.DEVICE = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    
    return DefaultPredictor(cfg)


@torch.no_grad()
def evaluate_centermask2(val_loader,
                        cfg_path=CM2_CONFIG_PATH,
                        weights_path=CM2_WEIGHTS_PATH,
                        score_thr=CM2_SCORE_THR,
                        max_dets=CM2_MAX_DETS,
                        batch_verbose=False):
    """
    Evaluates CenterMask2 (LIVECell-trained model) on the dataset split 
    and computes counting metrics.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for the evaluation split.
        cfg_path (str, optional): CenterMask2 config path. Defaults to CM2_CONFIG_PATH.
        weights_path (str, optional): CenterMask2 weights path. Defaults to CM2_WEIGHTS_PATH.
        score_thr (float, optional): Test-time score threshold. Defaults to CM2_SCORE_THR.
        max_dets (int, optional): Max detections per image. Defaults to CM2_MAX_DETS.
        batch_verbose (bool, optional): If True, prints count results for each image. Defaults to False.

    Returns:
        Dict: A dictionary containing the calculated counting metrics.
    """
    predictor = build_centermask2_predictor(cfg_path, weights_path, score_thr, max_dets, device=DEVICE)

    all_pred, all_gt = [], []
    for batch in val_loader:
        # Handle batches with or without density maps (depending on LiveCellDataset configuration)
        if len(batch) == 3:
            images, _, gt_counts = batch
        else:
            images, gt_counts = batch

        for i in range(images.size(0)):
            # Convert image tensor to NumPy array in float[0,1] format, then BGR for Detectron2
            np_img = to_hw_or_hwc(images[i], to_float01=True)      # float32 (H,W) or (H,W,C) in [0,1]
            np_img = _rgb_to_bgr(np_img)           # D2 expects BGR
            
            # Predict instances using the Detectron2 predictor
            outputs = predictor(np_img)
            # Extract detected instances and move to CPU
            inst: Instances = outputs["instances"].to("cpu")
            # The predicted count is the number of detected instances
            pred_count = int(len(inst))            
            
            all_pred.append(pred_count)
            all_gt.append(int(gt_counts[i].item()))
            if batch_verbose:
                print(f"[CM2] Pred {pred_count} | GT {int(gt_counts[i].item())}")

    # Calculate final counting metrics
    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics


# ================== COUNTING UTILS ==================
def lacss_count_from_pred(pred) -> int:
    """
    Extracts the predicted cell count from the raw output of a LACSS predictor.

    LACSS predictors can return a dictionary (with masks, labels, or scores) or 
    a single label image. This function tries to robustly extract the count.

    Args:
        pred (Dict | np.ndarray): The raw output from the LACSS predictor.

    Returns:
        int: The extracted predicted count.
    """
    if isinstance(pred, dict):
        # Case 1: Predicted label map (each object has a unique pixel value 1..N)
        if "pred_label" in pred and pred["pred_label"] is not None:
            lab = pred["pred_label"]
            # Count is the maximum value in the label map
            return int(np.max(lab)) if getattr(lab, "size", 0) else 0
        for key in ("segmentation", "label"):
            if key in pred and pred[key] is not None:
                lab = pred[key]
                return int(np.max(lab)) if getattr(lab, "size", 0) else 0
        # Case 2: List/array of instance masks (N x H x W)
        for key in ("pred_masks", "instances", "masks"):
            if key in pred and pred[key] is not None:
                m = pred[key]
                if isinstance(m, np.ndarray):  # NxHxW
                    return int(m.shape[0]) # Count is the batch dimension N
                if isinstance(m, (list, tuple)):
                    return int(len(m))
    # Case 3: Single label map array
    if isinstance(pred, np.ndarray):
        return int(np.max(pred)) # Count is the maximum label value
    return 0


# ================== EVALUATORS ==================
@torch.no_grad()
def evaluate_lacss(val_loader,
                   id_or_path: str,
                   reshape_to=None,
                   min_area: float = 0.0,
                   score_threshold: float = 0.5,
                   segmentation_threshold: float = 0.5,
                   batch_verbose: bool = False):
    """
    Evaluates a LACSS model on the dataset split and computes counting metrics.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for the evaluation split.
        id_or_path (str): Model ID or path for the LACSS predictor.
        reshape_to (Any, optional): Reshape argument for the predictor. Defaults to None.
        min_area (float, optional): Minimum area threshold. Defaults to 0.0.
        score_threshold (float, optional): Confidence score threshold. Defaults to 0.5.
        segmentation_threshold (float, optional): Segmentation threshold. Defaults to 0.5.
        batch_verbose (bool, optional): If True, prints count results for each image. Defaults to False.

    Returns:
        Dict: A dictionary containing the calculated counting metrics.
    """
    predictor = build_lacss_predictor(id_or_path)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, _, gt_counts = batch
            else:
                images, gt_counts = batch

            for i in range(images.size(0)):
                
                # Convert image tensor to NumPy array in uint8 [0,255] format with 3 channels
                img_np = to_hw_or_hwc(images[i], to_float01=False, ensure_3ch=True, debug=False)
                
                # Run prediction on the single image
                pred = _predict_single(
                    predictor,
                    img_np,
                    output_type="label", # Request a label map for easy counting
                    reshape_to=reshape_to,
                    min_area=min_area,
                    score_threshold=score_threshold,
                    segmentation_threshold=segmentation_threshold,
                )
                
                # Extract count from the predictor's output
                pred_count = lacss_count_from_pred(pred)
                all_pred.append(int(pred_count))
                all_gt.append(int(gt_counts[i].item()))
                if batch_verbose:
                    print(f"Pred {pred_count} | GT {int(gt_counts[i].item())}")

    # Calculate final counting metrics
    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics


# ================== CELLPOSE ==================


def _build_cellpose(model_type: str, use_gpu: bool):
    """
    Constructs a Cellpose model instance, handling API differences between 
    older (Cellpose) and newer (CellposeModel) versions.

    Args:
        model_type (str): The pre-trained model type (e.g., "cyto", "nuclei").
        use_gpu (bool): If True, attempts to use the GPU for inference.

    Returns:
        cellpose.models.Cellpose | cellpose.models.CellposeModel: The initialized Cellpose model.
    """
    # Check if GPU is available and requested
    gpu_flag = (use_gpu and torch.cuda.is_available())
    try:
        return models.Cellpose(model_type=model_type, gpu=gpu_flag)        # <= 2.x API
    except AttributeError:
        return models.CellposeModel(model_type=model_type, gpu=gpu_flag) # >= 3.x API


def _shape_with_channels(arr):
    """Returns the shape of an image array, including a channel dimension of 1 for 2D arrays."""
    return arr.shape if arr.ndim == 3 else (arr.shape[0], arr.shape[1], 1)


def _cellpose_eval(cp, np_img, channels, diameter, _debug_once=[]):
    """
    Runs the Cellpose model's evaluation function with version compatibility handling.

    Args:
        cp (Object): The initialized Cellpose model instance.
        np_img (np.ndarray): The input image array (H,W) or (H,W,C).
        channels (List[int]): The channel arguments for Cellpose (required for v2/v3).
        diameter (float | None): The diameter argument for Cellpose.
        _debug_once (List, optional): Used internally to print debug info only once.

    Returns:
        np.ndarray | None: The resulting label mask array (H, W).
    """
    # one-time debug print
    if not _debug_once:
        print(f"[Cellpose] v{_CPVER} | np_img shape={_shape_with_channels(np_img)} "
              f"| dtype={np_img.dtype} | min={np_img.min()} | max={np_img.max()}")
        print(f"[Cellpose] channels arg will be {'USED' if _CP_MAJOR<4 else 'IGNORED'}: {channels}")
        _debug_once.append(True)

    # v4+: do NOT pass channels; v2/v3: pass it
    kwargs = dict(diameter=diameter, augment=False, batch_size=1)
    if _CP_MAJOR < 4:
        kwargs["channels"] = channels # Only pass 'channels' argument for older versions

    # The result structure changes depending on the Cellpose version
    result = cp.eval(np_img, **kwargs)

    # Coerce result to the mask array
    if isinstance(result, dict):
        masks = result.get("masks", result.get("labels"))
    elif isinstance(result, (list, tuple)):
        # Handle batched output from eval
        masks = result[0] if len(result) else None
        if isinstance(masks, (list, tuple)):  # handle potential nested list if batch_size > 1 was used
            masks = masks[0] if len(masks) else None
    else:
        masks = result
    
    return masks


@torch.no_grad()
def evaluate_cellpose(val_loader, mode='grayscale', model_type='cyto', diameter=None,
                      use_gpu=True, batch_verbose=False):
    """
    Evaluates a Cellpose model on the dataset split and computes counting metrics.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for the evaluation split.
        mode (str, optional): Image processing mode ('grayscale' or 'rgb'). Defaults to 'grayscale'.
        model_type (str, optional): Cellpose pre-trained model type. Defaults to 'cyto'.
        diameter (float | None, optional): Expected cell diameter. Defaults to None (auto-detection).
        use_gpu (bool, optional): Whether to use GPU for Cellpose inference. Defaults to True.
        batch_verbose (bool, optional): If True, prints count results for each image. Defaults to False.

    Returns:
        Dict: A dictionary containing the calculated counting metrics.
    """
    cp = _build_cellpose(model_type=model_type, use_gpu=use_gpu)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                images, _, gt_counts = batch
            else:
                images, gt_counts = batch
            B = images.size(0)

            for i in range(B):
                # Convert tensor to Cellpose-compatible NumPy array and get channel info
                np_img, channels = _to_cellpose_img_and_channels(images[i], mode)
                
                # Run Cellpose inference
                masks = _cellpose_eval(cp, np_img, channels, diameter)
                
                # Count is the maximum label value in the resulting label map
                pred_count = int(masks.max()) if masks is not None and masks.size > 0 else 0

                all_pred.append(pred_count)
                all_gt.append(int(gt_counts[i].item()))
                if batch_verbose:
                    print(f"Pred {pred_count} | GT {int(gt_counts[i].item())}")

    # Calculate final counting metrics
    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics


# ================== MAIN ==================
if __name__ == "__main__":
    loader = make_loader(TEST)
    print(f"Started evaluation | Model={EVAL_MODEL} | Split={'test' if TEST else 'val'}")

    if EVAL_MODEL.lower() == "lacss":
        metrics = evaluate_lacss(
            loader,
            id_or_path=LACSS_ID_OR_PATH,
            reshape_to=LACSS_RESHAPE_TO,
            min_area=LACSS_MIN_AREA,
            score_threshold=LACSS_SCORE_THR,
            segmentation_threshold=LACSS_SEG_THR,
            batch_verbose=BATCH_VERBOSE,
        )
    elif EVAL_MODEL.lower() == "cellpose":
        metrics = evaluate_cellpose(
            loader,
            mode=CP_MODE,
            model_type=CP_MODEL_TYPE,
            diameter=CP_DIAMETER,
            use_gpu=CP_USE_GPU,
            batch_verbose=BATCH_VERBOSE,
        )
    elif EVAL_MODEL.lower() == "centermask2":
        metrics = evaluate_centermask2(
            loader,
            cfg_path=CM2_CONFIG_PATH,
            weights_path=CM2_WEIGHTS_PATH,
            score_thr=CM2_SCORE_THR,
            max_dets=CM2_MAX_DETS,
            batch_verbose=BATCH_VERBOSE,
        )
    else:
        raise ValueError("EVAL_MODEL must be 'lacss', 'cellpose', or 'centermask2'.")

    print_metrics(metrics, split="test" if TEST else "val",
                  model=EVAL_MODEL.upper(),
                  n_images=len(loader.dataset),
                  show_rmse=True)