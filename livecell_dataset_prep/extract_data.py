import os
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import zipfile
import shutil

def extract_json_from_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                if file.endswith('.json'):
                    return os.path.join(root, file)
    raise FileNotFoundError("No JSON file found in ZIP")

def save_image_and_masks_from_train_json(
    json_zip_path,
    image_name,
    image_src_dir,
    output_dir="/home/meidanzehavi/Cell_counter/my_scripts/maks_example"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cell_masks"), exist_ok=True)
    
    # Extract JSON
    temp_dir = "temp_train_json"
    annotation_path = extract_json_from_zip(json_zip_path, temp_dir)

    # Load COCO
    coco = COCO(annotation_path)

    # Match image file by basename
    matched_img = None
    for img in coco.dataset['images']:
        if os.path.splitext(os.path.basename(img['file_name']))[0] == image_name:
            matched_img = img
            break

    if not matched_img:
        raise ValueError(f"Image with name {image_name} not found in annotations")

    img_id = matched_img['id']
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)
    height, width = matched_img['height'], matched_img['width']

    # Save binary mask (all cells)
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    if len(seg) >= 6:
                        points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                        temp_mask = Image.new('L', (width, height), 0)
                        ImageDraw.Draw(temp_mask).polygon(points, outline=1, fill=1)
                        binary_mask = np.maximum(binary_mask, np.array(temp_mask))
            elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                binary_mask = np.maximum(binary_mask, coco_mask.decode(ann['segmentation']))

    binary_mask = (binary_mask * 255).astype(np.uint8)
    Image.fromarray(binary_mask).save(os.path.join(output_dir, f"{image_name}_mask.png"))

    # Save individual masks
    for i, ann in enumerate(annotations):
        cell_mask = np.zeros((height, width), dtype=np.uint8)
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    if len(seg) >= 6:
                        points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                        temp_mask = Image.new('L', (width, height), 0)
                        ImageDraw.Draw(temp_mask).polygon(points, outline=1, fill=1)
                        cell_mask = np.maximum(cell_mask, np.array(temp_mask))
            elif isinstance(ann['segmentation'], dict):
                if 'counts' in ann['segmentation']:
                    cell_mask = coco_mask.decode(ann['segmentation'])

        cell_mask = (cell_mask * 255).astype(np.uint8)
        Image.fromarray(cell_mask).save(os.path.join(output_dir, "cell_masks", f"cell_{i}.png"))

    # Copy original image
    src_img_path = os.path.join(image_src_dir, f"{image_name}.png")
    dst_img_path = os.path.join(output_dir, f"{image_name}.png")
    shutil.copyfile(src_img_path, dst_img_path)

    print(f"✅ Saved {len(annotations)} cell masks and image to: {output_dir}")

# Example usage
save_image_and_masks_from_train_json(
    json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_train.json.zip",
    image_name="A172_Phase_A7_1_00d08h00m_4",
    image_src_dir="/home/meidanzehavi/Cell_counter/livecell_dataset/images/train"
)
