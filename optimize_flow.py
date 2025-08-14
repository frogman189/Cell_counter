import os
import sys

import optuna
import matplotlib.pyplot as plt
from utils.constants import DEVICE, OUTPUT_OPTUNA_DIR, TIME, MODEL_NAME, dataset_paths, train_cfg
from train_CellCounter import train
from models import get_model
from preprocess import prepare_dataset, LiveCellDataset


NTRAILS = 100
STUDY_NAME = f"optimize_hyperparameters_{TIME}"

def save_optuna_results(study, timestamp):
    """Save Optuna study results and plots in OUTPUT_OPTUNA_DIR."""

    # Create output directory
    output_dir = os.path.join(OUTPUT_OPTUNA_DIR, f"optimization_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save the study results as a CSV
    csv_path = os.path.join(output_dir, f"{STUDY_NAME}.csv")
    study.trials_dataframe().to_csv(csv_path)
    print(f"Results saved at: {csv_path}")

    # Plot and save contour plot
    contour_plot = optuna.visualization.plot_contour(study, params=["lr", "weight_count", "batch_size"])
    contour_path = os.path.join(output_dir, f"{STUDY_NAME}_lr_aweight_count_BatchSize_contour.png")
    contour_plot.write_image(contour_path)

    # Plot and save contour plot of lr and batch_size
    contour_plot = optuna.visualization.plot_contour(study, params=["lr", "batch_size"])
    contour_path = os.path.join(output_dir, f"{STUDY_NAME}_lr_batch_size_contour.png")
    contour_plot.write_image(contour_path)

    # Plot and save parameter importance
    importance_plot = optuna.visualization.plot_param_importances(study)
    importance_path = os.path.join(output_dir, f"{STUDY_NAME}_param_importance.png")
    importance_plot.write_image(importance_path)

    plt.close('all')  # Close plots to avoid memory leaks
    print(f"Plots saved in: {output_dir}")

    try:
        best = study.best_trial
        best_txt = os.path.join(output_dir, "best_params.txt")
        with open(best_txt, "w") as f:
            f.write(f"Study: {STUDY_NAME}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Best value: {best.value}\n")
            f.write("Best params:\n")
            for k, v in best.params.items():
                f.write(f"  {k}: {v}\n")
        print(f"Best params saved at: {best_txt}")
    except Exception as e:
        print(f"Could not save best params: {e}")


def optimize_hyperparameters():
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=STUDY_NAME, direction="maximize", sampler=sampler)

    dataset_dict = prepare_dataset(dataset_paths['path_to_original_dataset'], dataset_paths['path_to_livecell_images'], dataset_paths['path_to_labels'])
    train_dataset = LiveCellDataset(dataset_dict['train'])
    val_dataset = LiveCellDataset(dataset_dict['val'])

    model = get_model()
    model.to(DEVICE)
    
    study.optimize(lambda trial:  train(model, train_dataset, val_dataset, train_cfg, device=DEVICE, optuna=True, trial=trial), n_trials=NTRAILS) #trial.report  # , timeout=16*60


    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")

    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Save results and plots
    save_optuna_results(study, TIME)
    
if __name__ == '__main__':
    optimize_hyperparameters()