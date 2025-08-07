import os
import json
import zipfile
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
import numpy as np
from pycocotools import mask as maskUtils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


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


def load_annotations(path_to_dataset_dir):
    # Assumes unzipped json files are present in the same dir
    coco_anns = {}
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(path_to_dataset_dir, f'livecell_annotations_{split}.json')
        coco = COCO(json_path)
        coco_anns[split] = coco
    return coco_anns


def load_labels_csv(path_to_labels_dir):
    labels = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(path_to_labels_dir, f"{split}_data.csv")
        df = pd.read_csv(csv_path)
        labels[split] = df
    return labels


def build_data_entries(coco_anns, labels, path_to_images_root):
    """
    Build structured entries: list of dicts for each image.
    Matches based on basename (since COCO uses relative paths like A172/A172_Phase_...)
    """
    dataset = {}
    for split in ['train', 'val', 'test']:
        coco = coco_anns[split]
        label_df = labels[split]
        entries = []

        for _, row in label_df.iterrows():
            filename_from_csv = row['filename']
            filename_only = os.path.basename(filename_from_csv)

            # Match using basename
            matched = [
                img for img in coco.imgs.values()
                if os.path.basename(img['file_name']) == filename_only
            ]

            if not matched:
                print(f"Warning: could not find annotation for {filename_from_csv}")
                continue

            img_info = matched[0]
            img_id = img_info['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            entry = {
                'filename': os.path.join(path_to_images_root, split, filename_from_csv),
                'cell_count': row['cell_count'],
                'width': row['width'],
                'height': row['height'],
                'annotations': anns  # contains polygon info
            }
            entries.append(entry)

        dataset[split] = entries
    return dataset

def prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels):
    unzip_jsons(path_to_original_dataset)
    coco_anns = load_annotations(path_to_original_dataset)
    label_dfs = load_labels_csv(path_to_labels)
    dataset = build_data_entries(coco_anns, label_dfs, path_to_livecell_images)

    return dataset


# class LiveCellDataset(Dataset):
#     def __init__(self, data_entries):
#         self.data = data_entries
        
#         self.transforms = A.Compose([
#             A.Resize(512, 512),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         entry = self.data[idx]
#         image = np.array(Image.open(entry['filename']).convert("RGB"))
#         height, width = entry['height'], entry['width']
#         annotations = entry['annotations']
        
#         if not annotations:
#             target = {
#                 "boxes": torch.zeros((0, 4), dtype=torch.float32),
#                 "labels": torch.zeros((0,), dtype=torch.int64),
#                 "masks": torch.zeros((0, 512, 512), dtype=torch.uint8),
#                 "image_id": torch.tensor([idx]),
#                 "area": torch.zeros((0,), dtype=torch.float32),
#                 "iscrowd": torch.zeros((0,), dtype=torch.int64)
#             }
#             transformed = self.transforms(image=image)
#             return transformed['image'], target
        
#         masks, labels, areas, iscrowd = [], [], [], []
        
#         for ann in annotations:
#             rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
#             mask = maskUtils.decode(rle)
#             if mask.ndim == 3:
#                 mask = np.any(mask, axis=2).astype(np.uint8)
            
#             if mask.sum() == 0:
#                 continue
            
#             masks.append(mask)
#             labels.append(1)  # All cells are labeled as class 1
#             areas.append(ann.get('area', float(mask.sum())))
#             iscrowd.append(ann.get('iscrowd', 0))
        
#         if not masks:
#             target = {
#                 "boxes": torch.zeros((0, 4), dtype=torch.float32),
#                 "labels": torch.zeros((0,), dtype=torch.int64),
#                 "masks": torch.zeros((0, 512, 512), dtype=torch.uint8),
#                 "image_id": torch.tensor([idx]),
#                 "area": torch.zeros((0,), dtype=torch.float32),
#                 "iscrowd": torch.zeros((0,), dtype=torch.int64)
#             }
#             transformed = self.transforms(image=image)
#             return transformed['image'], target
        
#         masks = np.stack(masks, axis=0)
#         transformed = self.transforms(image=image, masks=list(masks))
#         image_tensor = transformed['image']
#         masks_tensor = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in transformed['masks']])
        
#         boxes, valid_indices = [], []
#         for i, mask in enumerate(masks_tensor):
#             pos = mask.nonzero()
#             if len(pos) == 0:
#                 continue
#             xmin = torch.min(pos[:, 1]).float()
#             xmax = torch.max(pos[:, 1]).float()
#             ymin = torch.min(pos[:, 0]).float()
#             ymax = torch.max(pos[:, 0]).float()
#             if xmax > xmin and ymax > ymin:
#                 boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
#                 valid_indices.append(i)
        
#         if boxes:
#             masks_tensor = masks_tensor[valid_indices]
#             labels = [labels[i] for i in valid_indices]
#             areas = [areas[i] for i in valid_indices]
#             iscrowd = [iscrowd[i] for i in valid_indices]
#             target = {
#                 "boxes": torch.stack(boxes),
#                 "labels": torch.tensor(labels, dtype=torch.int64),
#                 "masks": masks_tensor,
#                 "image_id": torch.tensor([idx]),
#                 "area": torch.tensor(areas, dtype=torch.float32),
#                 "iscrowd": torch.tensor(iscrowd, dtype=torch.int64)
#             }
#         else:
#             target = {
#                 "boxes": torch.zeros((0, 4), dtype=torch.float32),
#                 "labels": torch.zeros((0,), dtype=torch.int64),
#                 "masks": torch.zeros((0, 512, 512), dtype=torch.uint8),
#                 "image_id": torch.tensor([idx]),
#                 "area": torch.zeros((0,), dtype=torch.float32),
#                 "iscrowd": torch.zeros((0,), dtype=torch.int64)
#             }

#         return image_tensor, target


class LiveCellDataset(Dataset):
    def __init__(self, data_entries):
        self.data = data_entries
        self.transforms = A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),  # Critical for masks
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = np.array(Image.open(entry['filename']).convert("RGB"))
        annotations = entry['annotations']

        def empty_target():
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, 512, 512), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        if not annotations:
            transformed = self.transforms(image=image)
            return transformed['image'], empty_target()

        masks, labels, areas, iscrowd = [], [], [], []
        for ann in annotations:
            rle = maskUtils.frPyObjects(ann['segmentation'], entry['height'], entry['width'])
            mask = maskUtils.decode(rle)
            if mask.ndim == 3:
                mask = np.any(mask, axis=2).astype(np.uint8)
            if mask.sum() == 0:
                continue
            masks.append(mask)
            labels.append(1)  # Class 1 = cell
            areas.append(ann.get('area', float(mask.sum())))
            iscrowd.append(ann.get('iscrowd', 0))

        if not masks:
            transformed = self.transforms(image=image)
            return transformed['image'], empty_target()

        # Apply transforms to image AND masks together
        transformed = self.transforms(image=image, masks=masks)
        image_tensor = transformed['image']
        masks_tensor = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in transformed['masks']])

        # Calculate boxes (post-resize)
        boxes = []
        valid_indices = []
        for i, mask in enumerate(masks_tensor):
            pos = (mask == 1).nonzero()
            if len(pos) == 0:
                continue
            xmin, ymin = pos.min(dim=0)[0][1], pos.min(dim=0)[0][0]
            xmax, ymax = pos.max(dim=0)[0][1], pos.max(dim=0)[0][0]
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
            valid_indices.append(i)

        if not boxes:
            return image_tensor, empty_target()

        # Filter valid masks/labels
        masks_tensor = masks_tensor[valid_indices]
        labels = [labels[i] for i in valid_indices]
        areas = [areas[i] for i in valid_indices]
        iscrowd = [iscrowd[i] for i in valid_indices]

        target = {
            "boxes": torch.stack(boxes),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        return image_tensor, target

# Debug function to check your dataset
def debug_dataset_labels(dataset, num_samples=10):
    """Debug function to check label distribution in your dataset"""
    print("=== DATASET DEBUG INFO ===")
    print(f"Dataset size: {len(dataset)}")
    
    all_labels = []
    for i in range(min(num_samples, len(dataset))):
        try:
            image, target = dataset[i]
            labels = target['labels'].tolist()
            all_labels.extend(labels)
            print(f"Sample {i}: {len(labels)} objects, labels: {labels}")
        except Exception as e:
            print(f"Error in sample {i}: {e}")
    
    if all_labels:
        print(f"\nLabel statistics:")
        print(f"  Min label: {min(all_labels)}")
        print(f"  Max label: {max(all_labels)}")
        print(f"  Unique labels: {sorted(set(all_labels))}")
        print(f"  Total labels: {len(all_labels)}")
    else:
        print("No labels found!")
    
### Example usage of the class:
###dataset_dict = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)
###train_dataset = LiveCellDataset(dataset_dict['train'])



if __name__ == "__main__":
    # Define your paths here
    path_to_original_dataset = "/home/meidanzehavi/livecell"
    path_to_livecell_images = "/home/meidanzehavi/Cell_counter/livecell_dataset/images"
    path_to_labels = "/home/meidanzehavi/Cell_counter/livecell_dataset"

    dataset = prepare_dataset(path_to_original_dataset, path_to_livecell_images, path_to_labels)

    print(f"Train images: {len(dataset['train'])}")
    print(f"Val images: {len(dataset['val'])}")
    print(f"Test images: {len(dataset['test'])}")

    # Print the first training sample
    if dataset['train']:
        print("\nFirst training sample:")
        for key, value in dataset['train'][0].items():
            if key == "annotations":
                print(f"{key}: {len(value)} annotations")
                print("First 2 annotation entries:")
                for ann in value[:2]:  # print first 2 annotations
                    print(json.dumps(ann, indent=2))  # pretty print
            else:
                print(f"{key}: {value}")

    
    train_dataset = LiveCellDataset(dataset['train'])
    debug_dataset_labels(train_dataset, num_samples=2)
    image, target = train_dataset[0]
    print("🔍 Image tensor shape:", image.shape)  # Should be [3, 512, 512]
    print("📦 Number of masks:", target['masks'].shape[0])
    print("🖼️ Mask tensor shape (per instance):", target['masks'].shape)  # [N, 512, 512]
    print("📦 Boxes shape:", target['boxes'].shape)
    print("📦 Boxes values:", target['boxes'])
    print("🏷️ Labels:", target['labels'])  # List of ints
    print("📐 Areas:", target['area'])
    print("🚧 Is crowd:", target['iscrowd'])
    print("🆔 Image ID:", target['image_id'])
