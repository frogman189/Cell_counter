import os
import zipfile
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from pycocotools import coco as pycocotools_coco
from pycocotools import mask as maskUtils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy.ndimage import gaussian_filter
from utils.constants import MODEL_NAME, dataset_paths

# -----------------------------------------
# unzip jsons 
# -----------------------------------------
def unzip_jsons(path_to_dataset_dir):
    """
    Unzips COCO-style annotation JSON files for the LiveCell dataset splits (train, val, test) 
    if the zip archives exist.

    Args:
        path_to_dataset_dir (str): The directory containing the zipped annotation files.
    """
    json_zips = [
        "livecell_annotations_train.json.zip",
        "livecell_annotations_val.json.zip",
        "livecell_annotations_test.json.zip",
    ]
    for json_zip in json_zips:
        zip_path = os.path.join(path_to_dataset_dir, json_zip)
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extracts the contents (the .json file) into the same directory
                zip_ref.extractall(path_to_dataset_dir)
                print(f"Extracted {json_zip}")

# -----------------------------------------
# load COCO jsons 
# -----------------------------------------
def load_annotations(path_to_dataset_dir):
    """
    Loads COCO-style annotation JSON files for each dataset split into COCO objects.

    Args:
        path_to_dataset_dir (str): The directory containing the unzipped annotation JSON files.

    Returns:
        dict: A dictionary mapping split names ('train', 'val', 'test') to COCO objects.

    Raises:
        FileNotFoundError: If any required JSON file is missing.
    """
    coco_anns = {}
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(path_to_dataset_dir, f'livecell_annotations_{split}.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing {json_path}")
        # Initialize the COCO API object with the annotation file
        coco_anns[split] = pycocotools_coco.COCO(json_path)
    return coco_anns

# -----------------------------------------
# load CSVs 
# -----------------------------------------
def load_labels_csv(path_to_labels_dir):
    """
    Loads CSV files containing image filenames and their corresponding cell counts.

    Args:
        path_to_labels_dir (str): The directory containing the CSV files (e.g., 'train_data.csv').

    Returns:
        dict: A dictionary mapping split names ('train', 'val', 'test') to pandas DataFrames.

    Raises:
        FileNotFoundError: If any required CSV file is missing.
        ValueError: If a loaded CSV file does not contain 'filename' and 'cell_count' columns.
    """
    labels = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(path_to_labels_dir, f"{split}_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        if "filename" not in df.columns or "cell_count" not in df.columns:
            raise ValueError(f"{csv_path} must contain 'filename' and 'cell_count' columns")
        df["cell_count"] = df["cell_count"].astype(int)
        labels[split] = df
    return labels

# -----------------------------------------
# build entries 
# -----------------------------------------
def build_data_entries(coco_anns, labels, path_to_images_root):
    """
    Combines COCO annotations and CSV labels into a structured list of data entries per split.

    Annotations are stored directly in the entry; centroids are computed and cached lazily 
    during the first call to __getitem__.

    Args:
        coco_anns (dict): Dictionary of COCO objects loaded via `load_annotations`.
        labels (dict): Dictionary of pandas DataFrames loaded via `load_labels_csv`.
        path_to_images_root (str): The base directory for the images (e.g., /images).

    Returns:
        dict: A dictionary mapping split names to lists of data entry dictionaries.
    """
    dataset = {}
    for split in ['train', 'val', 'test']:
        coco = coco_anns[split]
        label_df = labels[split]
        entries = []

        for _, row in label_df.iterrows():
            filename_from_csv = row['filename']
            # Take just the image basename (e.g., 'A182_1.tif') for matching
            filename_only = os.path.basename(filename_from_csv)

            # match by basename
            matched = [
                img for img in coco.imgs.values()
                if os.path.basename(img['file_name']) == filename_only
            ]
            if not matched:
                # If no matching image is found in COCO annotations, skip this CSV entry
                continue

            img_info = matched[0]
            img_id = img_info['id']
            # Get annotation IDs and load the full annotations list for this image
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            entry = {
                # Constructs the full, absolute path to the image file
                'filename': os.path.join(path_to_images_root, split, filename_from_csv),  # original path rule
                'cell_count': int(row['cell_count']),
                'width': int(img_info['width']),
                'height': int(img_info['height']),
                'annotations': anns,         # Stores raw COCO annotations
                'centroids': None,           # <-- will be filled on first __getitem__ (lazy computation)
            }
            entries.append(entry)

        dataset[split] = entries
    return dataset

def prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels):
    """
    A convenience function to run the full preparation pipeline.

    Args:
        path_to_original_dataset (str): Directory containing the zipped JSON annotations.
        path_to_livecell_images (str): Root directory containing split image folders (train/val/test).
        path_to_labels (str): Directory containing the cell count CSV files.

    Returns:
        dict: The structured data entries dictionary.
    """
    unzip_jsons(path_to_original_dataset)
    coco_anns = load_annotations(path_to_original_dataset)
    label_dfs = load_labels_csv(path_to_labels)
    return build_data_entries(coco_anns, label_dfs, path_to_livecell_images)

# -----------------------------------------
# density utils 
# -----------------------------------------
def points_to_density(points, img_shape, sigma=8):
    """
    Generates a smoothed density map from a list of point coordinates.

    This implements a Gaussian kernel density estimation approach where a spike 
    of '1.0' is placed at each point, and the resulting map is convolved with a 
    Gaussian filter.

    Args:
        points (list): List of (x, y) coordinates.
        img_shape (tuple): (Height, Width) of the density map grid.
        sigma (float): Standard deviation of the Gaussian kernel for smoothing.

    Returns:
        np.ndarray: The smoothed density map of shape (H, W).
    """
    H, W = img_shape
    density = np.zeros((H, W), dtype=np.float32)
    for x, y in points:
        # Clip coordinates to be within bounds [0, W-eps) and [0, H-eps)
        x = np.clip(x, 0, W - 1e-6)
        y = np.clip(y, 0, H - 1e-6)
        # Convert to integer index by taking the floor
        ix = int(np.floor(x)); iy = int(np.floor(y))
        # Place a unit spike at the nearest pixel coordinate
        density[iy, ix] += 1.0
    # Apply Gaussian smoothing to spread the count over the area
    density = gaussian_filter(density, sigma=sigma)
    return density

# -----------------------------------------
# helper: compute centroids from full mask 
# -----------------------------------------
def _compute_centroids_from_annotations(anns, H, W):
    """
    Calculates the centroid (mean x, mean y) for each instance mask in the COCO annotations.
    If segmentation data is unavailable, it falls back to the center of the bounding box.

    Args:
        anns (list): List of COCO annotation dictionaries.
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        list: List of (x, y) centroid coordinates (floats).
    """
    pts = []
    for ann in anns:
        seg = ann.get('segmentation')
        m = None
        if isinstance(seg, list) and len(seg) > 0:
            # Handle list of polygons: convert to RLE and decode to mask
            rles = maskUtils.frPyObjects(seg, H, W)
            rle = maskUtils.merge(rles)
            m = maskUtils.decode(rle)
        elif isinstance(seg, dict) and 'counts' in seg:
            # Handle RLE format: decode directly to mask
            m = maskUtils.decode(seg)
        
        if m is not None:
            # If a mask exists, compute the mean (centroid) of non-zero pixels
            ys, xs = np.where(m > 0)
            if xs.size:
                pts.append([float(xs.mean()), float(ys.mean())])
        elif 'bbox' in ann:
            # fallback to bounding box center if no segmentation data is available
            x, y, w, h = ann['bbox']
            pts.append([x + w/2.0, y + h/2.0])
    return pts

# -----------------------------------------
# dataset 
# -----------------------------------------
class LiveCellDataset(Dataset):
    """
    PyTorch Dataset for cell counting using the density map regression approach.

    Returns: (image_tensor, density_map_tensor, scalar_count_tensor)
    """
    def __init__(self, data_entries, img_size=512,
                 normalize: bool = True,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        """
        Initializes the LiveCellDataset for density map regression.

        Args:
            data_entries (list): List of data entry dictionaries (from `build_data_entries`).
            img_size (int): The target size (height and width) for image resizing.
            normalize (bool): If True, applies ImageNet mean/std normalization.
            mean (tuple): Mean values for normalization.
            std (tuple): Standard deviation values for normalization.
        """
        self.data = data_entries
        self.img_size = int(img_size)

        tfms = [A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR)]
        if normalize:
            # Add normalization based on the flag
            tfms.append(A.Normalize(mean=mean, std=std))
        tfms.append(ToTensorV2())

        self.transforms = A.Compose(tfms)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample (image, density map, and count) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, density_tensor, count_tensor)
        
        Raises:
            FileNotFoundError: If the image path in the entry is invalid.
        """
        entry = self.data[idx]

        # Check for file existence before reading
        if not os.path.exists(entry['filename']):
            raise FileNotFoundError(f"Image not found: {entry['filename']}")

        # read image and convert to RGB format
        image = np.array(Image.open(entry['filename']).convert("RGB"))

        # calculate scaling factors from original size to resized size
        sx = self.img_size / float(entry['width'])
        sy = self.img_size / float(entry['height'])

        # LAZY + CACHED centroid computation (correct full-mask centroid)
        if entry['centroids'] is None:
            # Computes and caches the centroids on first access
            entry['centroids'] = _compute_centroids_from_annotations(
                entry['annotations'],
                entry['height'],
                entry['width']
            )
        points = entry['centroids']

        # scale points to resized grid (important for density map generation)
        scaled_points = [(px * sx, py * sy) for (px, py) in points]

        # density on resized grid (H x W)
        density = points_to_density(scaled_points, (self.img_size, self.img_size), sigma=8)

        # apply image transforms (resizing, normalization, ToTensor)
        transformed = self.transforms(image=image)
        image_tensor = transformed['image'].float()
        # Convert density map to tensor and add a channel dimension [1, H, W]
        density_tensor = torch.from_numpy(density).float().unsqueeze(0)

        return image_tensor, density_tensor, torch.tensor(entry['cell_count'], dtype=torch.float32)

# -----------------------------------------
# minimal debug to confirm shapes
# -----------------------------------------
def debug_count_dataset(ds, k=2):
    """
    Prints information about the dataset and the first k samples, including shapes 
    and ground truth values.

    Args:
        ds (Dataset): The dataset instance to inspect.
        k (int): Number of samples to inspect and print details for.
    """
    print("=== COUNT DATASET DEBUG ===")
    print(f"size: {len(ds)}")
    for i in range(min(k, len(ds))):
        img, dens, cnt = ds[i]
        # Print tensor shapes and ground truth values
        print(f"[{i}] img {tuple(img.shape)} dens {tuple(dens.shape)} gt {int(cnt)} sum {float(dens.sum()):.1f}")


# ---------- shared helpers for masks/boxes (do not affect old code) ----------
def _decode_ann_masks(anns, H, W):
    """
    Decodes COCO annotations (polygons or RLE) into a stack of binary instance masks.

    Args:
        anns (list): List of COCO annotation dictionaries.
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        np.ndarray: Stack of binary masks of shape [N, H, W], where N is the number of instances.
    """
    ms = []
    for ann in anns:
        seg = ann.get('segmentation')
        m = None
        if isinstance(seg, list) and len(seg) > 0:
            # Handle multipolygon format
            rles = maskUtils.frPyObjects(seg, H, W)
            rle  = maskUtils.merge(rles)
            m    = maskUtils.decode(rle)
        elif isinstance(seg, dict) and 'counts' in seg:
            # Handle RLE format
            m = maskUtils.decode(seg)
        else:
            m = None
        if m is not None:
            ms.append(m.astype(np.uint8))
    if len(ms) == 0:
        # Return an empty stack if no masks were found
        return np.zeros((0, H, W), dtype=np.uint8)
    return np.stack(ms, axis=0)

def _masks_to_boxes(msk):
    """
    Computes axis-aligned bounding boxes [x1, y1, x2, y2) for each binary mask in a stack.

    The format is half-open: [min_x, min_y, max_x + 1, max_y + 1).

    Args:
        msk (np.ndarray): Stack of binary masks of shape [N, H, W] (uint8).

    Returns:
        np.ndarray: Array of bounding boxes of shape [N, 4] (float32).
    """
    if msk.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    N, H, W = msk.shape
    boxes = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        ys, xs = np.where(msk[i] > 0)
        if xs.size == 0:
            boxes[i] = 0  # set box to [0, 0, 0, 0] if mask is empty
        else:
            x1, y1 = xs.min(), ys.min()
            # Calculate max coordinates and add 1 for half-open interval [x1, y1, x2, y2)
            x2, y2 = xs.max() + 1, ys.max() + 1
            # clip to image bounds (should not be necessary if masks are correctly sized)
            if x2 > W: x2 = W
            if y2 > H: y2 = H
            boxes[i] = [x1, y1, x2, y2]
    return boxes

def _instance_to_semantic(masks, num_classes=2):
    """
    Converts a stack of instance masks into a single semantic segmentation map.
    Foreground (cell) pixels are marked as 1, and background as 0.

    Args:
        masks (np.ndarray): Stack of instance masks of shape [N, H, W].
        num_classes (int): The number of classes (expected to be >= 2 for bg/cell).

    Returns:
        np.ndarray: Semantic map of shape [H, W] in {0, 1} (uint8).
    """
    if masks.shape[0] == 0:
        # If no instances, return an all-zero background map
        return np.zeros(masks.shape[1:], dtype=np.uint8)
    # Sum the masks along the instance dimension (axis=0) and check where the sum > 0
    fg = (masks.sum(axis=0) > 0).astype(np.uint8)
    return fg  # 0/1 map

# ---------------------- Mask R-CNN dataset ----------------------
class LiveCellMaskRCNNDataset(Dataset):
    """
    PyTorch Dataset tailored for Instance Segmentation using torchvision's Mask R-CNN models.

    Returns: (image_tensor, target_dict, scalar_count_tensor)
    - image: [3,H,W] float tensor (usually ImageNet normalized)
    - target: dict with keys: boxes [N,4], labels [N], masks [N,H,W] uint8, etc.
    """
    def __init__(self, data_entries, img_size=512, imagenet_norm=True):
        """
        Initializes the LiveCellMaskRCNNDataset.

        Args:
            data_entries (list): List of data entry dictionaries.
            img_size (int): The target size (H and W) for image and mask resizing.
            imagenet_norm (bool): If True, applies ImageNet mean/std normalization.
        """
        self.data = data_entries
        self.img_size = int(img_size)

        norm = [A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])] if imagenet_norm else []
        # Compose transforms: Resize image linearly, normalize (if required), and convert to tensor
        self.tf = A.Compose([A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR)] + norm + [ToTensorV2()])

    def __len__(self): 
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (image, Mask R-CNN target, and count).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, target_dict, count_tensor)
        
        Raises:
            FileNotFoundError: If the image path is invalid.
        """
        entry = self.data[idx]
        img_path = entry['filename']
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        H0, W0 = entry['height'], entry['width']
        assert image.shape[:2] == (H0, W0), "Image size mismatch with annotations"

        # Image transform (resizing, normalization, ToTensor)
        img_t = self.tf(image=image)['image'].float()

        # decode and resize instance masks with NEAREST interpolation
        masks0 = _decode_ann_masks(entry['annotations'], H0, W0)  # [N,H0,W0]
        if masks0.shape[0] > 0:
            masks_rs = np.stack([
                # Masks must be resized using NEAREST interpolation to preserve binary values
                cv2.resize(m.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                for m in masks0
            ], axis=0)
        else:
            masks_rs = masks0

        boxes = _masks_to_boxes(masks_rs)                                 # [N,4] (xyxy)
        # All instances are the same class (cell), labeled as 1 (background is 0 implicitly)
        labels = np.ones((boxes.shape[0],), dtype=np.int64)                # 1 = "cell"
        iscrowd = np.zeros_like(labels, dtype=np.int64)                    # LiveCell annotations don't use 'iscrowd'
        # Calculate area from the computed bounding boxes
        area = (boxes[:,2]-boxes[:,0]).clip(min=0) * (boxes[:,3]-boxes[:,1]).clip(min=0)

        # Structure target dictionary according to torchvision format
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks_rs, dtype=torch.uint8),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return img_t, target, torch.tensor(entry['cell_count'], dtype=torch.float32)

def maskrcnn_collate_fn(batch):
    """
    Collate function for Mask R-CNN dataloader.

    The model expects images and targets as lists, so this function unzips the batch 
    and returns them as lists.

    Args:
        batch (list): A list of tuples, where each tuple is a sample (image, target_dict, count).

    Returns:
        tuple: (list of image tensors, list of target dictionaries)
    """
    # Unzip the batch: [(img1, target1, c1), (img2, target2, c2), ...] -> ([img1, img2, ...], [target1, target2, ...], [c1, c2, ...])
    # The count (c1, c2, ...) is discarded here as it's not used by the standard Mask R-CNN forward pass
    imgs, targets, counts = zip(*batch)
    return list(imgs), list(targets)

# ---------------------- UNet semantic segmentation dataset ----------------------
class LiveCellSemSegDataset(Dataset):
    """
    PyTorch Dataset tailored for Semantic Segmentation using models like UNet (2-class output: background/cell).

    Returns: (image_tensor, class_map_tensor, scalar_count_tensor)
    - image: [3,H,W] float tensor (usually ImageNet normalized)
    - class_map: [H,W] long tensor in {0,1} (0=background, 1=cell)
    """
    def __init__(self, data_entries, img_size=512, imagenet_norm=True, num_classes=2):
        """
        Initializes the LiveCellSemSegDataset.

        Args:
            data_entries (list): List of data entry dictionaries.
            img_size (int): The target size (H and W) for image and mask resizing.
            imagenet_norm (bool): If True, applies ImageNet mean/std normalization.
            num_classes (int): The number of output classes (must be >= 2).
        """
        assert num_classes >= 2, "For semantic UNet use num_classes >= 2 (bg/cell)."
        self.data = data_entries
        self.img_size = int(img_size)
        self.num_classes = num_classes

        norm = [A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])] if imagenet_norm else []
        # Compose transforms: Resize image linearly, normalize (if required), and convert to tensor
        self.tf = A.Compose([A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR)] + norm + [ToTensorV2()])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (image, semantic map, and count).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, target_tensor, count_tensor)
        
        Raises:
            FileNotFoundError: If the image path is invalid.
        """
        entry = self.data[idx]
        img_path = entry['filename']
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        H0, W0 = entry['height'], entry['width']

        # Image transform (resizing, normalization, ToTensor)
        img_t = self.tf(image=image)['image'].float()  # [3,H,W]

        # decode instances and collapse to class map
        masks0 = _decode_ann_masks(entry['annotations'], H0, W0)  # [N,H0,W0]
        # Convert the stack of instance masks to a single semantic map
        class_map = _instance_to_semantic(masks0, num_classes=self.num_classes)  # [H0,W0] in {0,1}

        # resize with NEAREST interpolation to keep labels intact
        class_map_rs = cv2.resize(class_map, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        # Convert to LongTensor, as required for PyTorch's cross-entropy loss target
        target = torch.from_numpy(class_map_rs).long()  # [H,W]
        return img_t, target, torch.tensor(entry['cell_count'], dtype=torch.float32)
    

def load_LiveCellDataSet(mode='train'):
    """
    Initializes and returns the appropriate LiveCell Dataset class based on the global MODEL_NAME.

    Args:
        mode (str): The dataset split to load ('train', 'val', or 'test').

    Returns:
        Dataset: An instance of LiveCellMaskRCNNDataset, LiveCellSemSegDataset, or LiveCellDataset.

    Raises:
        ValueError: If the provided mode is invalid.
    """
    if mode not in ['train', 'val', 'test']:
        raise ValueError(f"Dataset mode '{mode}' is invalid. "
            "Supported modes are: 'train', 'val' or 'test'.")
    
    # Load all data entries once
    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    
    # Set image size based on model type (e.g., smaller size for ViT/ConvNeXt classification models)
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    
    if MODEL_NAME == 'Mask_R_CNN_ResNet50':
        # Instance segmentation requires a specific target format
        dataset = LiveCellMaskRCNNDataset(dataset_dict[mode], img_size=img_size)
    elif MODEL_NAME == 'Unet':
        # Semantic segmentation requires a class map target
        dataset = LiveCellSemSegDataset(dataset_dict[mode], img_size=img_size)
    else:
        # Default to the density regression/direct count dataset
        dataset = LiveCellDataset(dataset_dict[mode], img_size=img_size)
    
    return dataset

# -----------------------------------------
# example
# -----------------------------------------
if __name__ == "__main__":
    # Example paths (placeholder - actual paths would need to be valid)
    path_to_original_dataset = "/home/meidanzehavi/livecell"
    # /home/meidanzehavi/livecell_temp
    path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
    path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

    dataset = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)

    print(f"Train images: {len(dataset['train'])}")
    print(f"Val images:   {len(dataset['val'])}")
    print(f"Test images:  {len(dataset['test'])}")

    # Example initialization of the default LiveCellDataset
    train_dataset = LiveCellDataset(dataset['train'], img_size=512, normalize=True)
    debug_count_dataset(train_dataset, k=2)