import os
import zipfile
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from pycocotools.coco import COCO
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
    json_zips = [
        "livecell_annotations_train.json.zip",
        "livecell_annotations_val.json.zip",
        "livecell_annotations_test.json.zip",
    ]
    for json_zip in json_zips:
        zip_path = os.path.join(path_to_dataset_dir, json_zip)
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path_to_dataset_dir)
                print(f"Extracted {json_zip}")

# -----------------------------------------
# load COCO jsons 
# -----------------------------------------
def load_annotations(path_to_dataset_dir):
    coco_anns = {}
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(path_to_dataset_dir, f'livecell_annotations_{split}.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing {json_path}")
        coco_anns[split] = COCO(json_path)
    return coco_anns

# -----------------------------------------
# load CSVs 
# -----------------------------------------
def load_labels_csv(path_to_labels_dir):
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
    Fast: we keep annotations in each entry; centroids are computed lazily later.
    """
    dataset = {}
    for split in ['train', 'val', 'test']:
        coco = coco_anns[split]
        label_df = labels[split]
        entries = []

        for _, row in label_df.iterrows():
            filename_from_csv = row['filename']
            filename_only = os.path.basename(filename_from_csv)

            # match by basename (your original approach)
            matched = [
                img for img in coco.imgs.values()
                if os.path.basename(img['file_name']) == filename_only
            ]
            if not matched:
                # keep running; maybe some CSV rows don’t exist in this split
                # (this matches your previous behavior)
                continue

            img_info = matched[0]
            img_id = img_info['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            entry = {
                'filename': os.path.join(path_to_images_root, split, filename_from_csv),  # original path rule
                'cell_count': int(row['cell_count']),
                'width': int(img_info['width']),
                'height': int(img_info['height']),
                'annotations': anns,         # we’ll compute centroids from this lazily
                'centroids': None,           # <-- will be filled on first __getitem__
            }
            entries.append(entry)

        dataset[split] = entries
    return dataset

def prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels):
    unzip_jsons(path_to_original_dataset)
    coco_anns = load_annotations(path_to_original_dataset)
    label_dfs = load_labels_csv(path_to_labels)
    return build_data_entries(coco_anns, label_dfs, path_to_livecell_images)

# -----------------------------------------
# density utils 
# -----------------------------------------
def points_to_density(points, img_shape, sigma=8):
    H, W = img_shape
    density = np.zeros((H, W), dtype=np.float32)
    for x, y in points:
        x = np.clip(x, 0, W - 1e-6)
        y = np.clip(y, 0, H - 1e-6)
        ix = int(np.floor(x)); iy = int(np.floor(y))
        density[iy, ix] += 1.0
    density = gaussian_filter(density, sigma=sigma)
    return density

# -----------------------------------------
# helper: compute centroids from full mask 
# -----------------------------------------
def _compute_centroids_from_annotations(anns, H, W):
    pts = []
    for ann in anns:
        seg = ann.get('segmentation')
        m = None
        if isinstance(seg, list) and len(seg) > 0:
            # multipolygon → RLE → decode once
            rles = maskUtils.frPyObjects(seg, H, W)
            rle = maskUtils.merge(rles)
            m = maskUtils.decode(rle)
        elif isinstance(seg, dict) and 'counts' in seg:
            m = maskUtils.decode(seg)
        # fallback to bbox center if no seg
        if m is not None:
            ys, xs = np.where(m > 0)
            if xs.size:
                pts.append([float(xs.mean()), float(ys.mean())])
        elif 'bbox' in ann:
            x, y, w, h = ann['bbox']
            pts.append([x + w/2.0, y + h/2.0])
    return pts

# -----------------------------------------
# dataset 
# -----------------------------------------
class LiveCellDataset(Dataset):
    def __init__(self, data_entries, img_size=512, imagenet_norm=True):
        self.data = data_entries
        self.img_size = img_size
        if imagenet_norm:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]

        # fast fail if path wrong
        if not os.path.exists(entry['filename']):
            raise FileNotFoundError(f"Image not found: {entry['filename']}")

        # read image
        image = np.array(Image.open(entry['filename']).convert("RGB"))

        # scale factors (original -> resized)
        sx = self.img_size / float(entry['width'])
        sy = self.img_size / float(entry['height'])

        # LAZY + CACHED centroid computation (correct full-mask centroid)
        if entry['centroids'] is None:
            entry['centroids'] = _compute_centroids_from_annotations(
                entry['annotations'],
                entry['height'],
                entry['width']
            )
        points = entry['centroids']

        # scale points to resized grid
        scaled_points = [(px * sx, py * sy) for (px, py) in points]

        # density on resized grid
        density = points_to_density(scaled_points, (self.img_size, self.img_size), sigma=8)

        # transforms
        transformed = self.transforms(image=image)
        image_tensor = transformed['image'].float()
        density_tensor = torch.from_numpy(density).float().unsqueeze(0)

        return image_tensor, density_tensor, torch.tensor(entry['cell_count'], dtype=torch.float32)

# -----------------------------------------
# minimal debug to confirm shapes
# -----------------------------------------
def debug_count_dataset(ds, k=2):
    print("=== COUNT DATASET DEBUG ===")
    print(f"size: {len(ds)}")
    for i in range(min(k, len(ds))):
        img, dens, cnt = ds[i]
        print(f"[{i}] img {tuple(img.shape)} dens {tuple(dens.shape)} gt {int(cnt)} sum {float(dens.sum()):.1f}")


# ---------- shared helpers for masks/boxes (do not affect old code) ----------
def _decode_ann_masks(anns, H, W):
    """Return list/stack of binary masks [N,H,W] from COCO anns."""
    ms = []
    for ann in anns:
        seg = ann.get('segmentation')
        if isinstance(seg, list) and len(seg) > 0:
            rles = maskUtils.frPyObjects(seg, H, W)
            rle  = maskUtils.merge(rles)
            m    = maskUtils.decode(rle)
        elif isinstance(seg, dict) and 'counts' in seg:
            m = maskUtils.decode(seg)
        else:
            m = None
        if m is not None:
            ms.append(m.astype(np.uint8))
    if len(ms) == 0:
        return np.zeros((0, H, W), dtype=np.uint8)
    return np.stack(ms, axis=0)

def _masks_to_boxes(msk):
    """Compute xyxy boxes from binary masks. msk: [N,H,W] uint8."""
    if msk.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    N, H, W = msk.shape
    boxes = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        ys, xs = np.where(msk[i] > 0)
        if xs.size == 0:
            boxes[i] = 0
        else:
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            boxes[i] = [x1, y1, x2, y2]
    return boxes

def _instance_to_semantic(masks, num_classes=2):
    """
    Collapse instances to a class map.
    For num_classes=2 (bg/cell) => {0,1}.
    """
    if masks.shape[0] == 0:
        return np.zeros(masks.shape[1:], dtype=np.uint8)
    fg = (masks.sum(axis=0) > 0).astype(np.uint8)
    return fg  # 0/1 map

# ---------------------- Mask R-CNN dataset ----------------------
class LiveCellMaskRCNNDataset(Dataset):
    """
    Returns (image_tensor, target_dict) for torchvision Mask R-CNN.
    - image: [3,H,W] float tensor (ImageNet norm)
    - target: dict with keys: boxes [N,4], labels [N], masks [N,H,W] uint8, image_id, area, iscrowd
    """
    def __init__(self, data_entries, img_size=512, imagenet_norm=True):
        self.data = data_entries
        self.img_size = int(img_size)

        norm = [A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])] if imagenet_norm else []
        self.tf = A.Compose([A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR)] + norm + [ToTensorV2()])

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry['filename']
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        H0, W0 = entry['height'], entry['width']
        assert image.shape[:2] == (H0, W0), "Image size mismatch with annotations"

        # image transform to [3,H,W]
        img_t = self.tf(image=image)['image'].float()

        # decode and resize instance masks with NEAREST
        masks0 = _decode_ann_masks(entry['annotations'], H0, W0)  # [N,H0,W0]
        if masks0.shape[0] > 0:
            masks_rs = np.stack([
                cv2.resize(m.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                for m in masks0
            ], axis=0)
        else:
            masks_rs = masks0

        boxes = _masks_to_boxes(masks_rs)                             # [N,4]
        labels = np.ones((boxes.shape[0],), dtype=np.int64)           # 1 = "cell"
        iscrowd = np.zeros_like(labels, dtype=np.int64)
        area = (boxes[:,2]-boxes[:,0]).clip(min=0) * (boxes[:,3]-boxes[:,1]).clip(min=0)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks_rs, dtype=torch.uint8),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return img_t, target

def maskrcnn_collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

# ---------------------- UNet semantic segmentation dataset ----------------------
class LiveCellSemSegDataset(Dataset):
    """
    Returns (image_tensor, class_map) for semantic segmentation UNet with 2 logits (bg/cell).
    - image: [3,H,W] float tensor (ImageNet norm)
    - class_map: [H,W] long in {0,1}
    """
    def __init__(self, data_entries, img_size=512, imagenet_norm=True, num_classes=2):
        assert num_classes >= 2, "For semantic UNet use num_classes >= 2 (bg/cell)."
        self.data = data_entries
        self.img_size = int(img_size)
        self.num_classes = num_classes

        norm = [A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])] if imagenet_norm else []
        self.tf = A.Compose([A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR)] + norm + [ToTensorV2()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry['filename']
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        H0, W0 = entry['height'], entry['width']

        img_t = self.tf(image=image)['image'].float()  # [3,H,W]

        # decode instances and collapse to class map
        masks0 = _decode_ann_masks(entry['annotations'], H0, W0)  # [N,H0,W0]
        class_map = _instance_to_semantic(masks0, num_classes=self.num_classes)  # [H0,W0] in {0,1}

        # resize with NEAREST to keep labels intact
        class_map_rs = cv2.resize(class_map, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        target = torch.from_numpy(class_map_rs).long()  # [H,W]
        return img_t, target
    

def load_LiveCellDataSet(mode='train'):
    if mode not in ['train', 'val', 'test']:
        raise ValueError(f"Dataset mode '{mode}' is invalid. "
            "Supported modes are: 'train', 'val' or 'test'.")
    
    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    img_size = 224 if MODEL_NAME in ("ViT_Count", "ConvNeXt_Count") else 512
    if MODEL_NAME == 'Mask_R_CNN_ResNet50':
        dataset = LiveCellMaskRCNNDataset(dataset_dict[mode], img_size=img_size)
    elif MODEL_NAME == 'Unet':
        dataset = LiveCellSemSegDataset(dataset_dict[mode], img_size=img_size)
    else:
        dataset = LiveCellDataset(dataset_dict[mode], img_size=img_size)
    
    return dataset

# -----------------------------------------
# example
# -----------------------------------------
if __name__ == "__main__":
    path_to_original_dataset = "/home/meidanzehavi/livecell"
    # /home/meidanzehavi/livecell_temp
    path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
    path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

    dataset = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)

    print(f"Train images: {len(dataset['train'])}")
    print(f"Val images:   {len(dataset['val'])}")
    print(f"Test images:  {len(dataset['test'])}")

    train_dataset = LiveCellDataset(dataset['train'], img_size=512, imagenet_norm=True)
    debug_count_dataset(train_dataset, k=2)
