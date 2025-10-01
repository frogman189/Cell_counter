import os
import time
from pathlib import Path
import torch
from utils.gpu_selector import get_best_gpu


# Get the absolute path to the project root (Cell_counter)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set global  timestamp
TIME = time.strftime("%Y%m%d-%H%M%S")

# Set GPU
best_gpu = get_best_gpu()
DEVICE = torch.device(f'cuda:{best_gpu}' if torch.cuda.is_available() else 'cpu')

# Set model to run
MODELS = ['Mask_R_CNN_ResNet50', 'Unet', 'UNetDensity', 'DeepLabDensity', 'MicroCellUNet', 'ViT_Count', 'ConvNeXt_Count', 'CNNTransformerCounter']
MODEL_NAME = 'ConvNeXt_Count'
assert MODEL_NAME in MODELS

model_args = {
    'model_type': MODEL_NAME
}

# Set optimizer
OPTIMIZERS = ['SGD','Adam', 'RAdam']
OPTIMIZER_NAME = 'Adam'
assert OPTIMIZER_NAME in OPTIMIZERS


dataset_paths = {
    'path_to_original_dataset': '/home/meidanzehavi/livecell',
    'path_to_livecell_images': '/home/meidanzehavi/Cell_counter/livecell_dataset/images',
    'path_to_labels': '/home/meidanzehavi/Cell_counter/livecell_dataset'
}

# Run flags
SAVE_MODEL = True # True if want to save the model after training
RUN_EXP = True # True if runing a trainig run that want to document it, False will just print the logger massages to the consule

# Set output dirs
RESULT_DIR = os.path.join(PROJECT_ROOT, f'results/{MODEL_NAME}')
OUTPUT_OPTUNA_DIR = os.path.join(PROJECT_ROOT, f'optuna_trials/{MODEL_NAME}')

# Set training confugurations
train_cfg = {
    'Mask_R_CNN_ResNet50': {
        'batch_size': 4,
        'num_workers': 8,
        'num_epochs': 60,
        'learning_rate': 0.0004827061977033549,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'Unet': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'w_density': 1.0,
        'w_count': 0.5,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'ViT_Count': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'w_density': 1.0,
        'w_count': 0.5,
        'huber_delta': 5.0,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'ConvNeXt_Count': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'huber_delta': 5.0,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'UNetDensity': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'w_density': 1.0,
        'w_ssim': 0.5,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'DeepLabDensity': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'w_density': 1.0,
        'w_count': 0.5,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'MicroCellUNet': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'w_density': 1.0,
        'w_count': 0.5,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
    'CNNTransformerCounter': {
        'batch_size': 8,
        'num_workers': 8,
        'num_epochs': 20,
        'learning_rate': 3e-04,
        'w_density': 1.0,
        'w_count': 0.5,
        'optimizer_name': OPTIMIZER_NAME,
        'result_dir': RESULT_DIR
    },
}





model_pathes = {'UNetDensity': '/home/meidanzehavi/Cell_counter/results/UNetDensity/UNetDensity_20250811-121939'}