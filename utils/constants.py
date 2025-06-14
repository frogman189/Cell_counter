import os
import time
from pathlib import Path

# Get the absolute path to the project root (Cell_counter)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIME = time.strftime("%Y%m%d-%H%M%S")

MODELS = ['ResNet50']
MODEL_NAME = 'ResNet50'
assert MODEL_NAME in MODELS

model_args = {
    'model_type': MODEL_NAME
}

OPTIMIZERS = ['SGD','Adam', 'RAdam']
OPTIMIZER_NAME = 'Adam'
assert OPTIMIZER_NAME in OPTIMIZERS

RESULT_DIR = os.path.join(PROJECT_ROOT, f'results/{MODEL_NAME}')

train_cfg = {
    'batch_size': 8,
    'num_workers': 4,
    'num_epochs': 4,
    'learning_rate': 1e-3,
    'optimizer_name': OPTIMIZER_NAME,
    'result_dir': RESULT_DIR
}

dataset_paths = {
    'path_to_original_dataset': '/home/meidanzehavi/livecell',
    'path_to_livecell_images': '/home/meidanzehavi/Cell_counter/livecell_dataset/images',
    'path_to_labels': '/home/meidanzehavi/Cell_counter/livecell_dataset'
}

SAVE_MODEL = False