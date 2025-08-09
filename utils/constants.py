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
MODELS = ['Mask_R_CNN_ResNet50', 'Unet', 'YOLOv8']
MODEL_NAME = 'Mask_R_CNN_ResNet50'
assert MODEL_NAME in MODELS

model_args = {
    'model_type': MODEL_NAME
}

# Set optimizer
OPTIMIZERS = ['SGD','Adam', 'RAdam']
OPTIMIZER_NAME = 'Adam'
assert OPTIMIZER_NAME in OPTIMIZERS

# Set output dirs
RESULT_DIR = os.path.join(PROJECT_ROOT, f'results/{MODEL_NAME}')
OUTPUT_OPTUNA_DIR = os.path.join(PROJECT_ROOT, f'optuna_trials/{MODEL_NAME}')

# Set training confugurations
train_cfg = {
    'batch_size': 2,
    'num_workers': 8,
    'num_epochs': 40,
    'learning_rate': 4e-3,
    'optimizer_name': OPTIMIZER_NAME,
    'result_dir': RESULT_DIR
}


dataset_paths = {
    'path_to_original_dataset': '/home/meidanzehavi/livecell',
    'path_to_livecell_images': '/home/meidanzehavi/Cell_counter/livecell_dataset/images',
    'path_to_labels': '/home/meidanzehavi/Cell_counter/livecell_dataset'
}

# Run flags
SAVE_MODEL = False
RUN_EXP = True # True if runing a trainig run that want to document it, False will just print the logger massages to the consule