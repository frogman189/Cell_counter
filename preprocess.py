# import os
# import json
# import zipfile
# import pandas as pd
# from PIL import Image
# from pycocotools.coco import COCO
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from pycocotools import mask as maskUtils
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
# from scipy.ndimage import gaussian_filter
# from pycocotools.coco import COCO


# def unzip_jsons(path_to_dataset_dir):
#     json_zips = [
#         "livecell_annotations_train.json.zip",
#         "livecell_annotations_val.json.zip",
#         "livecell_annotations_test.json.zip",
#     ]
#     for json_zip in json_zips:
#         zip_path = os.path.join(path_to_dataset_dir, json_zip)
#         if os.path.exists(zip_path):
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(path_to_dataset_dir)
#                 print(f"Extracted {json_zip}")


# def load_annotations(path_to_dataset_dir):
#     # Assumes unzipped json files are present in the same dir
#     coco_anns = {}
#     for split in ['train', 'val', 'test']:
#         json_path = os.path.join(path_to_dataset_dir, f'livecell_annotations_{split}.json')
#         coco = COCO(json_path)
#         coco_anns[split] = coco
#     return coco_anns


# def load_labels_csv(path_to_labels_dir):
#     labels = {}
#     for split in ['train', 'val', 'test']:
#         csv_path = os.path.join(path_to_labels_dir, f"{split}_data.csv")
#         df = pd.read_csv(csv_path)
#         labels[split] = df
#     return labels


# def build_data_entries(coco_anns, labels, path_to_images_root):
#     dataset = {}
#     for split in ['train', 'val', 'test']:
#         coco = coco_anns[split]
#         label_df = labels[split]
#         entries = []

#         for _, row in label_df.iterrows():
#             filename_from_csv = row['filename']
#             filename_only = os.path.basename(filename_from_csv)

#             matched = [
#                 img for img in coco.imgs.values()
#                 if os.path.basename(img['file_name']) == filename_only
#             ]
#             if not matched:
#                 print(f"Warning: could not find annotation for {filename_from_csv}")
#                 continue

#             img_info = matched[0]
#             img_id = img_info['id']
#             ann_ids = coco.getAnnIds(imgIds=[img_id])
#             anns = coco.loadAnns(ann_ids)

#             # NEW: centroids from full masks (polygon or RLE) via coco.annToMask
#             centroids = []
#             for ann in anns:
#                 m = coco.annToMask(ann)
#                 ys, xs = np.where(m > 0)
#                 if xs.size:
#                     centroids.append([float(xs.mean()), float(ys.mean())])

#             entry = {
#                 'filename': os.path.join(path_to_images_root, split, filename_from_csv),
#                 'cell_count': int(row['cell_count']),
#                 'width': img_info['width'],
#                 'height': img_info['height'],
#                 'annotations': anns,          # keep if you still want it
#                 'centroids': centroids,       # <-- use this in Dataset
#                 'count_json': len(anns),      # optional: sanity check vs CSV
#                 'file_name': img_info['file_name'],  # optional: original relpath
#             }
#             entries.append(entry)

#         dataset[split] = entries
#     return dataset

# def prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels):
#     unzip_jsons(path_to_original_dataset)
#     coco_anns = load_annotations(path_to_original_dataset)
#     label_dfs = load_labels_csv(path_to_labels)
#     dataset = build_data_entries(coco_anns, label_dfs, path_to_livecell_images)

#     return dataset



# def points_to_density(points, img_shape, sigma=8):
#     H, W = img_shape
#     density = np.zeros((H, W), dtype=np.float32)
#     for x, y in points:
#         # clamp to [0, W-1] and [0, H-1]
#         x = np.clip(x, 0, W - 1e-6)
#         y = np.clip(y, 0, H - 1e-6)
#         ix = int(np.floor(x))
#         iy = int(np.floor(y))
#         density[iy, ix] += 1.0
#     density = gaussian_filter(density, sigma=sigma)  # sum preserved
#     return density


# class LiveCellDataset(Dataset):
#     def __init__(self, data_entries, img_size=512):
#         self.data = data_entries
#         self.img_size = img_size
#         self.transforms = A.Compose([
#             A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         entry = self.data[idx]
#         image = np.array(Image.open(entry['filename']).convert("RGB"))

#         # scale factors
#         sx = self.img_size / float(entry['width'])
#         sy = self.img_size / float(entry['height'])

#         # use precomputed centroids
#         points = entry.get('centroids', [])
#         scaled_points = [(px * sx, py * sy) for (px, py) in points]

#         density = points_to_density(scaled_points, (self.img_size, self.img_size), sigma=8)

#         transformed = self.transforms(image=image)
#         image_tensor = transformed['image'].float()
#         density_tensor = torch.from_numpy(density).float().unsqueeze(0)
#         return image_tensor, density_tensor, torch.tensor(entry['cell_count'], dtype=torch.float32)

#     # def __getitem__(self, idx):
#     #     entry = self.data[idx]
#     #     image = np.array(Image.open(entry['filename']).convert("RGB"))
        
#     #     # Get all cell centers from annotations
#     #     orig_w = float(entry['width'])
#     #     orig_h = float(entry['height'])
#     #     sx = self.img_size / orig_w
#     #     sy = self.img_size / orig_h

#     #     # Build points (unchanged logic, but fix polygon parsing below)
#     #     points = []
#     #     for ann in entry['annotations']:
#     #         if 'segmentation' in ann:
#     #             seg = ann['segmentation']
#     #             # COCO polygon format: list of lists, each a flat [x1,y1,x2,y2,...]
#     #             if isinstance(seg, list) and len(seg) > 0:
#     #                 flat = seg[0] if isinstance(seg[0], (list, tuple)) else seg
#     #                 coords = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
#     #                 centroid = coords.mean(axis=0)  # (x, y)
#     #                 points.append(centroid.tolist())
#     #             elif isinstance(seg, dict) and 'counts' in seg:
#     #                 mask = maskUtils.decode(seg)
#     #                 y, x = np.where(mask > 0)
#     #                 if len(x) > 0:
#     #                     points.append([float(x.mean()), float(y.mean())])
#     #         elif 'bbox' in ann:
#     #             x, y, w, h = ann['bbox']
#     #             points.append([x + w/2.0, y + h/2.0])

#     #     # --- scale points to the resized canvas
#     #     scaled_points = [(px * sx, py * sy) for (px, py) in points]

#     #     # Create density map on the resized grid
#     #     density = points_to_density(scaled_points, (self.img_size, self.img_size), sigma=8)

#     #     # Apply image transforms AFTER reading the image (unchanged)
#     #     transformed = self.transforms(image=image)
#     #     image_tensor = transformed['image'].float()
#     #     density_tensor = torch.from_numpy(density).float().unsqueeze(0)
#     #     return image_tensor, density_tensor, torch.tensor(entry['cell_count'], dtype=torch.float32)


# # Debug function to check your dataset
# def debug_dataset_labels(dataset, num_samples=10):
#     """Debug function to check label distribution in your dataset"""
#     print("=== DATASET DEBUG INFO ===")
#     print(f"Dataset size: {len(dataset)}")
    
#     all_labels = []
#     for i in range(min(num_samples, len(dataset))):
#         try:
#             image, target = dataset[i]
#             labels = target['labels'].tolist()
#             all_labels.extend(labels)
#             print(f"Sample {i}: {len(labels)} objects, labels: {labels}")
#         except Exception as e:
#             print(f"Error in sample {i}: {e}")
    
#     if all_labels:
#         print(f"\nLabel statistics:")
#         print(f"  Min label: {min(all_labels)}")
#         print(f"  Max label: {max(all_labels)}")
#         print(f"  Unique labels: {sorted(set(all_labels))}")
#         print(f"  Total labels: {len(all_labels)}")
#     else:
#         print("No labels found!")
    
# ### Example usage of the class:
# ###dataset_dict = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)
# ###train_dataset = LiveCellDataset(dataset_dict['train'])



# if __name__ == "__main__":
#     # Define your paths here
#     path_to_original_dataset = "/home/meidanzehavi/livecell"
#     path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
#     path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

#     dataset = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)

#     print(f"Train images: {len(dataset['train'])}")
#     print(f"Val images: {len(dataset['val'])}")
#     print(f"Test images: {len(dataset['test'])}")

#     # Print the first training sample
#     if dataset['train']:
#         print("\nFirst training sample:")
#         for key, value in dataset['train'][0].items():
#             if key == "annotations":
#                 print(f"{key}: {len(value)} annotations")
#                 print("First 2 annotation entries:")
#                 for ann in value[:2]:  # print first 2 annotations
#                     print(json.dumps(ann, indent=2))  # pretty print
#             else:
#                 print(f"{key}: {value}")

    
#     train_dataset = LiveCellDataset(dataset['train'])
#     debug_dataset_labels(train_dataset, num_samples=2)
#     image, target = train_dataset[0]
#     print("🔍 Image tensor shape:", image.shape)  # Should be [3, 512, 512]
#     print("📦 Number of masks:", target['masks'].shape[0])
#     print("🖼️ Mask tensor shape (per instance):", target['masks'].shape)  # [N, 512, 512]
#     print("📦 Boxes shape:", target['boxes'].shape)
#     print("📦 Boxes values:", target['boxes'])
#     print("🏷️ Labels:", target['labels'])  # List of ints
#     print("📐 Areas:", target['area'])
#     print("🚧 Is crowd:", target['iscrowd'])
#     print("🆔 Image ID:", target['image_id'])


# ========= livecell_preprocess_fast.py =========
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

# -----------------------------------------
# unzip jsons (unchanged)
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
# load COCO jsons (unchanged)
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
# load CSVs (tiny validation added)
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
# build entries  (FAST: no annToMask here)
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
# density utils (unchanged)
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
# helper: compute centroids from full mask (lazy, per-item)
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
# dataset (FAST + CORRECT)
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

# -----------------------------------------
# example
# -----------------------------------------
if __name__ == "__main__":
    path_to_original_dataset = "/home/meidanzehavi/livecell"
    path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
    path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

    dataset = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)

    print(f"Train images: {len(dataset['train'])}")
    print(f"Val images:   {len(dataset['val'])}")
    print(f"Test images:  {len(dataset['test'])}")

    train_dataset = LiveCellDataset(dataset['train'], img_size=512, imagenet_norm=True)
    debug_count_dataset(train_dataset, k=2)
