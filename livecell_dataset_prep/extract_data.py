import json
import zipfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from collections import defaultdict
import shutil

def extract_livecell_for_counting(main_zip_path, train_json_zip_path, val_json_zip_path, test_json_zip_path, output_dir="livecell_organized"):
    """
    Extract LIVECell dataset and organize it for cell counting task.
    
    Args:
        main_zip_path: Path to LIVECell_dataset_2021.zip
        train_json_zip_path: Path to livecell_annotations_train.json.zip
        val_json_zip_path: Path to livecell_annotations_val.json.zip
        test_json_zip_path: Path to livecell_annotations_test.json.zip
        output_dir: Directory to save organized data
    
    Returns:
        dict: Summary of extracted data
    """
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/images/test", exist_ok=True)
    
    print("Step 1: Extracting JSON annotation files...")
    
    # Extract JSON files
    json_files = {}
    json_zip_paths = {
        'train': train_json_zip_path,
        'val': val_json_zip_path,
        'test': test_json_zip_path
    }
    
    for split, zip_path in json_zip_paths.items():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary location
            zip_ref.extractall(f"{output_dir}/temp_{split}")
            # Find the JSON file (might be nested)
            for root, dirs, files in os.walk(f"{output_dir}/temp_{split}"):
                for file in files:
                    if file.endswith('.json'):
                        json_files[split] = os.path.join(root, file)
                        break
    
    print("Step 2: Loading and processing annotations...")
    
    # Load annotations and count cells per image
    image_cell_counts = {}
    all_image_info = {}
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        
        with open(json_files[split], 'r') as f:
            coco_data = json.load(f)
        
        # Create mapping from image_id to filename
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Count annotations per image
        cell_counts = defaultdict(int)
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            cell_counts[image_id] += 1
        
        # Store image info with cell counts
        for img_info in coco_data['images']:
            img_id = img_info['id']
            filename = img_info['file_name']
            cell_count = cell_counts[img_id]
            
            image_cell_counts[filename] = {
                'split': split,
                'cell_count': cell_count,
                'image_id': img_id,
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0)
            }
            
            all_image_info[filename] = img_info
    
    print("Step 3: Extracting and converting images...")
    
    # Extract main dataset
    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"{output_dir}/temp_images")
    
    # Find images directory
    images_source_dir = None
    for root, dirs, files in os.walk(f"{output_dir}/temp_images"):
        if 'images' in dirs:
            images_source_dir = os.path.join(root, 'images')
            break
        # Sometimes images are directly in the extracted folder
        if any(f.endswith(('.tif', '.tiff', '.png', '.jpg')) for f in files):
            images_source_dir = root
            break
    
    if not images_source_dir:
        raise ValueError("Could not find images directory in extracted files")
    
    # Process and convert images
    processed_count = 0
    conversion_summary = {'train': 0, 'val': 0, 'test': 0}
    
    for filename, info in image_cell_counts.items():
        source_path = None
        
        # Find the actual image file (might have different extensions)
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(images_source_dir, filename.replace('.tif', ext))
            if os.path.exists(potential_path):
                source_path = potential_path
                break
        
        if not source_path:
            # Try looking in subdirectories
            for root, dirs, files in os.walk(images_source_dir):
                if filename in files or filename.replace('.tif', '.tiff') in files:
                    source_path = os.path.join(root, filename)
                    break
        
        if source_path and os.path.exists(source_path):
            try:
                # Load and convert image
                img = Image.open(source_path)
                
                # Convert grayscale to RGB if needed
                if img.mode in ['L', 'P']:
                    # Convert grayscale to RGB
                    img_rgb = Image.new('RGB', img.size)
                    img_array = np.array(img)
                    # Stack grayscale to create RGB
                    rgb_array = np.stack([img_array, img_array, img_array], axis=-1)
                    img_rgb = Image.fromarray(rgb_array)
                elif img.mode == 'RGBA':
                    # Convert RGBA to RGB
                    img_rgb = Image.new('RGB', img.size, (255, 255, 255))
                    img_rgb.paste(img, mask=img.split()[-1])
                else:
                    img_rgb = img.convert('RGB')
                
                # Save to appropriate split directory
                split = info['split']
                # Extract just the filename without subdirectories
                base_filename = os.path.basename(filename).replace('.tif', '.png')
                output_path = f"{output_dir}/images/{split}/{base_filename}"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img_rgb.save(output_path, 'PNG')
                
                processed_count += 1
                conversion_summary[split] += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print("Step 4: Creating metadata files...")
    
    # Create CSV files with image filenames and cell counts for each split
    for split in ['train', 'val', 'test']:
        split_data = []
        for filename, info in image_cell_counts.items():
            if info['split'] == split:
                # Extract just the filename without subdirectories
                base_filename = os.path.basename(filename).replace('.tif', '.png')
                split_data.append({
                    'filename': base_filename,
                    'cell_count': info['cell_count'],
                    'width': info['width'],
                    'height': info['height'],
                    'original_filename': filename
                })
        
        # Save to CSV
        df = pd.DataFrame(split_data)
        df.to_csv(f"{output_dir}/{split}_data.csv", index=False)
        print(f"Saved {len(df)} {split} samples to {split}_data.csv")
    
    # Create combined metadata file
    all_data = []
    for filename, info in image_cell_counts.items():
        base_filename = os.path.basename(filename).replace('.tif', '.png')
        all_data.append({
            'filename': base_filename,
            'split': info['split'],
            'cell_count': info['cell_count'],
            'width': info['width'],
            'height': info['height'],
            'original_filename': filename
        })
    
    combined_df = pd.DataFrame(all_data)
    combined_df.to_csv(f"{output_dir}/all_data.csv", index=False)
    
    # Clean up temporary directories
    print("Step 5: Cleaning up temporary files...")
    for temp_dir in [f"{output_dir}/temp_train", f"{output_dir}/temp_val", 
                     f"{output_dir}/temp_test", f"{output_dir}/temp_images"]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Create summary
    summary = {
        'total_images_processed': processed_count,
        'splits': conversion_summary,
        'output_directory': output_dir,
        'csv_files_created': [f"{split}_data.csv" for split in ['train', 'val', 'test']] + ['all_data.csv']
    }
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE!")
    print("="*50)
    print(f"Total images processed: {processed_count}")
    print(f"Train: {conversion_summary['train']} images")
    print(f"Val: {conversion_summary['val']} images") 
    print(f"Test: {conversion_summary['test']} images")
    print(f"\nData organized in: {output_dir}/")
    print("- images/train/, images/val/, images/test/ (RGB PNG files)")
    print("- train_data.csv, val_data.csv, test_data.csv (metadata with cell counts)")
    print("- all_data.csv (combined metadata)")
    
    return summary

# Usage example
if __name__ == "__main__":
    # Update these paths to match your downloaded files
    summary = extract_livecell_for_counting(
        main_zip_path="/home/meidanzehavi/livecell/LIVECell_dataset_2021.zip",
        train_json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_train.json.zip",
        val_json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_val.json.zip", 
        test_json_zip_path="/home/meidanzehavi/livecell/livecell_annotations_test.json.zip",
        output_dir="livecell_organized"
    )
    
    print("\nYou can now use the data like this:")
    print("import pandas as pd")
    print("train_df = pd.read_csv('livecell_organized/train_data.csv')")
    print("print(train_df.head())")
    print("# filename, cell_count, width, height columns available")