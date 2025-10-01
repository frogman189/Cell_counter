import os, glob, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from cellpose import models
import cellpose
from importlib.metadata import version as _pkgver
try:
    _CPVER = _pkgver("cellpose")
except Exception:
    _CPVER = "0.0.0"
_CP_MAJOR = int(_CPVER.split(".")[0]) if _CPVER and _CPVER[0].isdigit() else 0


from utils.metrics import calculate_counting_metrics, print_metrics
from preprocess import prepare_dataset, LiveCellDataset
from utils.constants import MODEL_NAME, dataset_paths, DEVICE

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(os.path.dirname(__file__), "centermask2"))

from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from centermask2.centermask.config import get_cfg

# ================== CONFIG ==================
EVAL_MODEL = "lacss"      # "lacss" or "cellpose" or "centermask2"
TEST = True              # True => evaluate on test split, else val

# LACSS settings
LACSS_ID_OR_PATH = os.path.join(BASE_DIR, "benchmark_models", "lacss3-small-l")
LACSS_RESHAPE_TO = None
LACSS_MIN_AREA = 1.0
LACSS_SCORE_THR = 0.001
LACSS_SEG_THR = 0.001

# Cellpose settings
CP_MODE = "grayscale"     # "grayscale" or "rgb"
CP_MODEL_TYPE = "cyto"    # "cyto","nuclei","cyto2","cyto3","cpsam"
CP_DIAMETER = None        # e.g., 20 (pixels) or None for auto
CP_USE_GPU = True
BATCH_VERBOSE = False

# CenterMask2 (LIVECell-trained) settings
CM2_CONFIG_PATH = os.path.join(BASE_DIR, "benchmark_models", "livecell_anchor_free_config.yaml")
CM2_WEIGHTS_PATH = os.path.join(BASE_DIR, "benchmark_models", "LIVECell_anchor_free_model.pth")
CM2_SCORE_THR = 0.05          # test-time score threshold
CM2_MAX_DETS = 3000           # VERY IMPORTANT for LIVECell density


# ================== DATALOADER HELPERS ==================
def collate_fn(batch):
    images, density_maps, cell_counts = zip(*batch)
    images = torch.stack(images, 0)
    density_maps = torch.stack(density_maps, 0)  # keep float
    cell_counts = torch.tensor(cell_counts, dtype=torch.float32)
    return images, density_maps, cell_counts


def make_loader(test_split: bool):
    dataset_dict = prepare_dataset(
        dataset_paths['path_to_original_dataset'],
        dataset_paths['path_to_livecell_images'],
        dataset_paths['path_to_labels'],
    )
    split_name = 'test' if test_split else 'val'
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    
    

    dataset = LiveCellDataset(dataset_dict[split_name], img_size=img_size, normalize=False)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True if DEVICE == 'cuda' else False,
    )
    return loader


# ================== IMAGE UTILS ==================
# def to_hw_or_hwc(img_t: torch.Tensor) -> np.ndarray:
#     """
#     Accepts [C,H,W] float tensor in [0,1] or [0,255]; returns (H,W) or (H,W,3) uint8.
#     """
#     eval_name = EVAL_MODEL.lower()
#     normalize = False if eval_name in ("cellpose") else True

#     x = img_t.detach().cpu().float()
#     y=x.numpy()
#     print("images[i] dtype/range:", y.dtype, np.min(y), np.max(y))
#     if x.max() <= 1.0:
#         x = x * 255.0
#     x = x.clamp(0, 255)
#     if x.shape[0] == 1:
#         print("fuck why")
#         return x[0].numpy().astype(np.uint8)            # grayscale
#     return x.permute(1, 2, 0).numpy().astype(np.uint8)  # RGB


def to_hw_or_hwc(img_t: torch.Tensor,
                 to_float01: bool = False,
                 ensure_3ch: bool | None = None,
                 debug: bool = False) -> np.ndarray:
    """
    Convert [C,H,W] tensor (values in [0,1] or [0,255]) to a NumPy image.

    - If to_float01=True  -> returns float32 in [0,1]
    - If to_float01=False -> returns uint8   in [0,255]

    Output shape:
      * (H,W)   if single-channel and ensure_3ch is False/None
      * (H,W,3) if 3 channels, or if ensure_3ch=True (grayscale is repeated)
    """
    x = img_t.detach().cpu().to(torch.float32)          # [C,H,W] float32
    C, H, W = x.shape

    # --- scale to desired range ---
    x_min, x_max = float(x.min()), float(x.max())

    if to_float01:
        # want [0,1]
        if x_max > 1.0 + 1e-6:           # looks like [0,255]
            x = x / 255.0
        x = x.clamp_(0.0, 1.0)
    else:
        # want uint8 [0,255]
        if x_max <= 1.0 + 1e-6:          # looks like [0,1]
            x = x * 255.0
        x = x.clamp_(0.0, 255.0)

    # --- channel handling ---
    if ensure_3ch:
        if C == 1:
            x = x.repeat(3, 1, 1)        # make 3-ch
            C = 3
        elif C > 3:
            x = x[:3]                    # drop extras if any

    if C == 1:
        out = x[0]                        # (H,W)
    else:
        out = x[:3].permute(1, 2, 0)      # (H,W,3)

    # --- dtype cast for output ---
    if to_float01:
        out_np = out.numpy().astype(np.float32)         # [0,1]
    else:
        out_np = out.numpy().astype(np.uint8)           # [0,255]

    if debug:
        mn, mx = float(out_np.min()), float(out_np.max())
        print(f"[to_hw_or_hwc] shape={out_np.shape}, dtype={out_np.dtype}, range=({mn:.4f},{mx:.4f})")

    return out_np


def _to_cellpose_img_and_channels(img_t: torch.Tensor, mode: str):
    """
    img_t: [C,H,W] float tensor in [0,1] or [0,255]
    mode: 'grayscale' or 'rgb'
    returns: np_img (H,W) or (H,W,3), channels tuple/list for Cellpose
    """
    img = img_t.detach().cpu().float()
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.clamp(0, 255)

    if mode == 'grayscale':
        if img.ndim == 3 and img.shape[0] > 1:
            r, g, b = img[0], img[1], img[2]
            img = 0.299*r + 0.587*g + 0.114*b
        else:
            img = img[0] if img.ndim == 3 else img
        np_img = img.numpy().astype(np.uint8)    # (H,W)
        channels = [0, 0]                        # cytoplasm from channel 0, no nuclei channel
    elif mode == 'rgb':
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        np_img = img.numpy().transpose(1, 2, 0).astype(np.uint8)  # (H,W,3)
        channels = [2, 1]  # (R as cytoplasm, G as nucleus) — common choice
    else:
        raise ValueError("mode must be 'grayscale' or 'rgb'")
    return np_img, channels


def _rgb_to_bgr(np_img):
    # Detectron2 expects BGR uint8 (OpenCV style)
    if np_img.ndim == 2:
        return np_img  # grayscale ok
    return np_img[:, :, ::-1]


# ================== ROBUST LACSS IMPORTS/BUILDER ==================
def _import_lacss_predictor_cls():
    """
    Return (PredictorClass, flavor) trying several LACSS APIs.
    """
    import importlib
    tried = []

    try:
        mod = importlib.import_module("lacss.deploy")
        if hasattr(mod, "Predictor"):
            return getattr(mod, "Predictor"), "deploy.Predictor"
        tried.append("lacss.deploy.Predictor")
    except Exception as e:
        tried.append(f"lacss.deploy.Predictor ! {type(e).__name__}")

    try:
        mod = importlib.import_module("lacss.deploy.predict")
        if hasattr(mod, "Predictor"):
            return getattr(mod, "Predictor"), "deploy.predict.Predictor"
        tried.append("lacss.deploy.predict.Predictor")
    except Exception as e:
        tried.append(f"lacss.deploy.predict.Predictor ! {type(e).__name__}")

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
    id_or_path can be:
      - an alias (e.g., 'small-2dL', 'lacss3-small-l'), or
      - a local file, or
      - a directory containing a model file.
    kwargs are passed to the predictor ctor if supported (silently ignored if not).
    """
    Predictor, flavor = _import_lacss_predictor_cls()

    def _try_ctor_for(path, **ctor_kwargs):
        # Try common constructor signatures with different kw names
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
                continue
            except Exception:
                # If it's not clearly a signature issue, keep trying others
                continue
        # Try positional-only path
        try:
            return Predictor(path, **ctor_kwargs)
        except Exception:
            return None

    # 1) Try as given
    pred = _try_ctor_for(id_or_path, **kwargs)
    if pred is not None:
        return pred

    # 2) If it's a directory, try files inside with common extensions
    if os.path.isdir(id_or_path):
        candidates = []
        for pat in ("*.npz", "*.ckpt", "*.pt", "*.pth", "*"):
            candidates.extend(glob.glob(os.path.join(id_or_path, pat)))
        for c in candidates:
            pred = _try_ctor_for(c, **kwargs)
            if pred is not None:
                return pred

    # 3) Try appending common extensions
    base = id_or_path.rstrip("/\\")
    for ext in (".npz", ".ckpt", ".pt", ".pth"):
        path = base + ext
        if os.path.exists(path):
            pred = _try_ctor_for(path, **kwargs)
            if pred is not None:
                return pred

    # 4) Last attempt
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
    Try the available predict methods across LACSS versions.
    """
    try:
        return predictor.predict(
            img_np,
            output_type=output_type,
            reshape_to=reshape_to,
            min_area=min_area,
            score_threshold=score_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_iou=1.0,
            normalize=True,
        )
    except AttributeError:
        pass
    except TypeError:
        try:
            return predictor.predict(img_np, output_type=output_type)
        except Exception:
            pass

    try:
        return predictor.predict_on_large_image(
            img_np,
            output_type=output_type,
            reshape_to=reshape_to,
            min_area=min_area,
            score_threshold=score_threshold,
            segmentation_threshold=segmentation_threshold,
            nms_iou=0.0,
        )
    except AttributeError:
        pass
    except TypeError:
        try:
            return predictor.predict_on_large_image(img_np, output_type=output_type)
        except Exception:
            pass

    try:
        return predictor(img_np)  # __call__
    except Exception:
        try:
            return predictor.infer(img_np)
        except Exception:
            pass

    raise RuntimeError("No compatible LACSS prediction method found on this predictor instance.")


# ================== ROBUST CenterMask2 IMPORTS/BUILDER ==================

def build_centermask2_predictor(cfg_path, weights_path, score_thr=0.05, max_dets=3000, device=DEVICE):
    cfg = get_cfg()                      # <-- from centermask.config
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thr)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.TEST.DETECTIONS_PER_IMAGE = int(max_dets)
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
    Runs CenterMask2 (LIVECell-trained) and computes counting metrics.
    """
    predictor = build_centermask2_predictor(cfg_path, weights_path, score_thr, max_dets, device=DEVICE)

    all_pred, all_gt = [], []
    for batch in val_loader:
        if len(batch) == 3:
            images, _, gt_counts = batch
        else:
            images, gt_counts = batch

        for i in range(images.size(0)):
            np_img = to_hw_or_hwc(images[i], to_float01=True)     # uint8 (H,W) or (H,W,3)
            np_img = _rgb_to_bgr(np_img)         # D2 expects BGR
            outputs = predictor(np_img)
            inst: Instances = outputs["instances"].to("cpu")
            pred_count = int(len(inst))          # number of instances
            all_pred.append(pred_count)
            all_gt.append(int(gt_counts[i].item()))
            if batch_verbose:
                print(f"[CM2] Pred {pred_count} | GT {int(gt_counts[i].item())}")

    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics


# ================== COUNTING UTILS ==================
def lacss_count_from_pred(pred) -> int:
    if isinstance(pred, dict):
        if "pred_label" in pred and pred["pred_label"] is not None:
            lab = pred["pred_label"]
            return int(np.max(lab)) if getattr(lab, "size", 0) else 0
        for key in ("segmentation", "label"):
            if key in pred and pred[key] is not None:
                lab = pred[key]
                return int(np.max(lab)) if getattr(lab, "size", 0) else 0
        for key in ("pred_masks", "instances", "masks"):
            if key in pred and pred[key] is not None:
                m = pred[key]
                if isinstance(m, np.ndarray):  # NxHxW
                    return int(m.shape[0])
                if isinstance(m, (list, tuple)):
                    return int(len(m))
    if isinstance(pred, np.ndarray):
        return int(np.max(pred))
    return 0


# ================== EVALUATORS ==================
def evaluate_lacss(val_loader,
                   id_or_path: str,
                   reshape_to=None,
                   min_area: float = 0.0,
                   score_threshold: float = 0.5,
                   segmentation_threshold: float = 0.5,
                   batch_verbose: bool = False):
    """
    Runs LACSS inference and computes your counting metrics.
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
                
                img_np = to_hw_or_hwc(images[i], to_float01=False, ensure_3ch=True, debug=False)
                pred = _predict_single(
                    predictor,
                    img_np,
                    output_type="label",
                    reshape_to=reshape_to,
                    min_area=min_area,
                    score_threshold=score_threshold,
                    segmentation_threshold=segmentation_threshold,
                )
                
                pred_count = lacss_count_from_pred(pred)
                all_pred.append(int(pred_count))
                all_gt.append(int(gt_counts[i].item()))
                if batch_verbose:
                    print(f"Pred {pred_count} | GT {int(gt_counts[i].item())}")

    metrics = calculate_counting_metrics(
        [int(x) for x in all_pred],
        [int(x) for x in all_gt],
        thresholds=[0, 1, 3, 5, 10, 20]
    )
    return metrics



# ================== CELLPOSE ==================


def _build_cellpose(model_type: str, use_gpu: bool):
    """Construct Cellpose model for both old (Cellpose) and new (CellposeModel) versions."""
    gpu_flag = (use_gpu and torch.cuda.is_available())
    try:
        return models.Cellpose(model_type=model_type, gpu=gpu_flag)      # <= 2.x
    except AttributeError:
        return models.CellposeModel(model_type=model_type, gpu=gpu_flag) # >= 3.x


def _shape_with_channels(arr):
    return arr.shape if arr.ndim == 3 else (arr.shape[0], arr.shape[1], 1)


def _cellpose_eval(cp, np_img, channels, diameter, _debug_once=[]):
    # one-time debug print
    if not _debug_once:
        print(f"[Cellpose] v{_CPVER} | np_img shape={_shape_with_channels(np_img)} "
              f"| dtype={np_img.dtype} | min={np_img.min()} | max={np_img.max()}")
        print(f"[Cellpose] channels arg will be {'USED' if _CP_MAJOR<4 else 'IGNORED'}: {channels}")
        _debug_once.append(True)

    # v4+: do NOT pass channels; v2/v3: pass it
    kwargs = dict(diameter=diameter, augment=False, batch_size=1)
    if _CP_MAJOR < 4:
        kwargs["channels"] = channels

    result = cp.eval(np_img, **kwargs)

    # coerce result to masks
    if isinstance(result, dict):
        masks = result.get("masks", result.get("labels"))
    elif isinstance(result, (list, tuple)):
        masks = result[0] if len(result) else None
        if isinstance(masks, (list, tuple)):  # batched
            masks = masks[0] if len(masks) else None
    else:
        masks = result
    return masks


def evaluate_cellpose(val_loader, mode='grayscale', model_type='cyto', diameter=None,
                      use_gpu=True, batch_verbose=False):
    """
    Runs Cellpose inference and computes your counting metrics.
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
                np_img, channels = _to_cellpose_img_and_channels(images[i], mode)
                masks = _cellpose_eval(cp, np_img, channels, diameter)
                pred_count = int(masks.max()) if masks is not None else 0

                all_pred.append(pred_count)
                all_gt.append(int(gt_counts[i].item()))
                if batch_verbose:
                    print(f"Pred {pred_count} | GT {int(gt_counts[i].item())}")

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
