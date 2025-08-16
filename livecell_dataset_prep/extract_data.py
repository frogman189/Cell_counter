import json
import zipfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from collections import defaultdict
import shutil

def extract_livecell_for_counting(
    main_zip_path,
    train_json_zip_path,
    val_json_zip_path,
    test_json_zip_path,
    output_dir="livecell_dataset"
):
    """
    Extract LIVECell dataset and organize it for cell counting task.

    Args:
        main_zip_path: Path to LIVECell_dataset_2021.zip
        train_json_zip_path: Path to livecell_annotations_train.json.zip
        val_json_zip_path: Path to livecell_annotations_val.json.zip
        test_json_zip_path: Path to livecell_annotations_test.json.zip
        output_dir: Existing directory under which ALL outputs will be written

    Returns:
        dict: Summary of extracted data
    """

    # ====== CHANGE 1: Fail fast if output_dir does not already exist ======
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output directory '{output_dir}' does not exist. Please create it first."
        )

    # ====== CHANGE 2: Everything goes under output_dir ======
    images_dir = os.path.join(output_dir, "images")
    train_dir  = os.path.join(images_dir, "train")
    val_dir    = os.path.join(images_dir, "val")
    test_dir   = os.path.join(images_dir, "test")

    # Create only subfolders (root must already exist)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    print("Step 1: Extracting JSON annotation files...")

    # Temp dirs scoped under output_dir
    temp_train = os.path.join(output_dir, "temp_train")
    temp_val   = os.path.join(output_dir, "temp_val")
    temp_test  = os.path.join(output_dir, "temp_test")

    # Extract JSON files
    json_files = {}
    json_zip_paths = {
        'train': (train_json_zip_path, temp_train),
        'val':   (val_json_zip_path,   temp_val),
        'test':  (test_json_zip_path,  temp_test),
    }

    for split, (zip_path, temp_dst) in json_zip_paths.items():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dst)
            for root, dirs, files in os.walk(temp_dst):
                for file in files:
                    if file.endswith('.json'):
                        json_files[split] = os.path.join(root, file)
                        break

    print("Step 2: Loading and processing annotations...")

    image_cell_counts = {}
    all_image_info = {}

    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")

        with open(json_files[split], 'r') as f:
            coco_data = json.load(f)

        # Count annotations per image
        cell_counts = defaultdict(int)
        for annotation in coco_data['annotations']:
            cell_counts[annotation['image_id']] += 1

        for img_info in coco_data['images']:
            img_id   = img_info['id']
            filename = img_info['file_name']
            cell_count = cell_counts[img_id]

            image_cell_counts[filename] = {
                'split': split,
                'cell_count': cell_count,
                'image_id': img_id,
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0),
            }
            all_image_info[filename] = img_info

    print("Step 3: Extracting and converting images...")

    temp_images = os.path.join(output_dir, "temp_images")
    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_images)

    # Find images directory
    images_source_dir = None
    for root, dirs, files in os.walk(temp_images):
        if 'images' in dirs:
            images_source_dir = os.path.join(root, 'images')
            break
        if any(f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')) for f in files):
            images_source_dir = root
            break

    if not images_source_dir:
        raise ValueError("Could not find images directory in extracted files")

    processed_count = 0
    conversion_summary = {'train': 0, 'val': 0, 'test': 0}

    for filename, info in image_cell_counts.items():
        source_path = None

        # try same basename with alternative extensions
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(
                images_source_dir, filename.replace('.tif', ext)
            )
            if os.path.exists(potential_path):
                source_path = potential_path
                break

        if not source_path:
            # search nested
            for root, dirs, files in os.walk(images_source_dir):
                if filename in files or filename.replace('.tif', '.tiff') in files:
                    source_path = os.path.join(root, filename)
                    break

        if source_path and os.path.exists(source_path):
            try:
                img = Image.open(source_path)

                # convert to RGB if needed
                if img.mode in ['L', 'P']:
                    arr = np.array(img)
                    img_rgb = Image.fromarray(np.stack([arr, arr, arr], axis=-1))
                elif img.mode == 'RGBA':
                    base = Image.new('RGB', img.size, (255, 255, 255))
                    base.paste(img, mask=img.split()[-1])
                    img_rgb = base
                else:
                    img_rgb = img.convert('RGB')

                split = info['split']
                base_filename = os.path.basename(filename).replace('.tif', '.png')
                if split == 'train':
                    out_path = os.path.join(train_dir, base_filename)
                elif split == 'val':
                    out_path = os.path.join(val_dir, base_filename)
                else:
                    out_path = os.path.join(test_dir, base_filename)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img_rgb.save(out_path, 'PNG')

                processed_count += 1
                conversion_summary[split] += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print("Step 4: Creating metadata files...")

    # CSVs live directly under output_dir
    for split in ['train', 'val', 'test']:
        split_data = []
        for filename, info in image_cell_counts.items():
            if info['split'] == split:
                base_filename = os.path.basename(filename).replace('.tif', '.png')
                split_data.append({
                    'filename': base_filename,
                    'cell_count': info['cell_count'],
                    'width': info['width'],
                    'height': info['height'],
                    'original_filename': filename,
                })
        df = pd.DataFrame(split_data)
        df.to_csv(os.path.join(output_dir, f"{split}_data.csv"), index=False)
        print(f"Saved {len(df)} {split} samples to {split}_data.csv")

    all_data = []
    for filename, info in image_cell_counts.items():
        base_filename = os.path.basename(filename).replace('.tif', '.png')
        all_data.append({
            'filename': base_filename,
            'split': info['split'],
            'cell_count': info['cell_count'],
            'width': info['width'],
            'height': info['height'],
            'original_filename': filename,
        })
    pd.DataFrame(all_data).to_csv(os.path.join(output_dir, "all_data.csv"), index=False)

    print("Step 5: Cleaning up temporary files...")
    for temp_dir in [temp_train, temp_val, temp_test, temp_images]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    summary = {
        'total_images_processed': processed_count,
        'splits': conversion_summary,
        'output_directory': output_dir,
        'csv_files_created': [f"{s}_data.csv" for s in ['train', 'val', 'test']] + ['all_data.csv'],
    }

    print("\n" + "="*50)
    print("EXTRACTION COMPLETE!")
    print("="*50)
    print(f"Total images processed: {processed_count}")
    print(f"Train: {conversion_summary['train']} images")
    print(f"Val:   {conversion_summary['val']} images")
    print(f"Test:  {conversion_summary['test']} images")
    print(f"\nData organized in: {output_dir}/")
    print("- images/train/, images/val/, images/test/ (RGB PNG files)")
    print("- train_data.csv, val_data.csv, test_data.csv (metadata with cell counts)")
    print("- all_data.csv (combined metadata)")

    return summary


# Usage example
if __name__ == "__main__":
    summary = extract_livecell_for_counting(
        main_zip_path="/home/meidanzehavi/livecell/LIVECell_dataset_2021.zip",
        train_json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_train.json.zip",
        val_json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_val.json.zip",
        test_json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_test.json.zip",
        output_dir="livecell_dataset"  # must exist beforehand
    )