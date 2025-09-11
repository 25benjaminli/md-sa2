import torch

# use sys path append to import from parent directory
import sys
import os
from pathlib import Path

# Add parent directory
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import os
from data_utils import get_dataloaders
from omegaconf import OmegaConf
from utils import set_deterministic, join
from models import initialize_mdsa2

sys.path.pop(0)

# Set matplotlib to use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
import argparse

if __name__ == "__main__":
    # run all tests
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_val", type=int, default=-1, help="Fold to evaluate. -1 means do the full CV.")
    args = parser.parse_args()

    if args.fold_val == -1:
        folds_to_test = list(range(10))
    else:
        folds_to_test = [args.fold_val]

    final_metrics = {}
    for fold_val in folds_to_test:
        print(f"---- evaluating fold {fold_val} ----")
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        model_config.config_folder = "sam2_tenfold"
        model_config.fold_val = [fold_val]

        details = OmegaConf.load(open(join(os.getenv("PROJECT_PATH", ""), 'MDSA2', 'config', "sam2_tenfold", 'details.yaml'), 'r'))
        model_config = OmegaConf.merge(model_config, details)
        model_config.dataset = "brats_africa"
        # set batch size to 1 for comparison w/ unet
        set_deterministic(42)
        train_loader, val_loader, file_paths = get_dataloaders(model_config, use_preprocessed=True, modality_to_repeat=-1, verbose=False)
        mdsa2 = initialize_mdsa2(model_config, (train_loader, val_loader))

        for batch in val_loader:
            # print("batch title", batch["image_title"])
            with torch.no_grad():
                metrics_sa2, metrics_mdsa2 = mdsa2(batch)
        print("SA2 dice:", metrics_sa2["dice"]["classwise_avg"])
        print("MD-SA2 dice:", metrics_mdsa2["dice"]["classwise_avg"])

        # print("SA2 hd95:", metrics_sa2["hd95"]["classwise_avg"])
        # print("MD-SA2 hd95:", metrics_mdsa2["hd95"]["classwise_avg"])

        # send to json
        if args.fold_val != -1:
            with open(f"test_metrics_sa2_fold_{model_config.fold_val}.json", "w") as f:
                json.dump(mdsa2.metric_accumulator_sa2.get_metrics(), f, indent=4)

            with open(f"test_metrics_mdsa2_fold_{model_config.fold_val}.json", "w") as f:
                json.dump(mdsa2.unet_model.metric_accumulator.get_metrics(), f, indent=4)

        final_metrics[f"fold_{model_config.fold_val}"] = {
            "sa2": mdsa2.metric_accumulator_sa2.get_metrics(),
            "mdsa2": mdsa2.unet_model.metric_accumulator.get_metrics()
        }

    # get the average from ALL folds
    if len(folds_to_test) > 1:
        avg_metrics = {}
        for fold in final_metrics.keys():
            for model_type in final_metrics[fold].keys():
                if model_type not in avg_metrics:
                    avg_metrics[model_type] = {}
                for metric_name in final_metrics[fold][model_type].keys():
                    # print("metric name", metric_name, final_metrics[fold][model_type][metric_name].keys())
                    if metric_name not in avg_metrics[model_type]:
                        avg_metrics[model_type][metric_name] = []
                    
                    if "classwise_avg" in final_metrics[fold][model_type][metric_name]:
                        avg_metrics[model_type][metric_name].append(final_metrics[fold][model_type][metric_name]["classwise_avg"])
                    else:
                        avg_metrics[model_type][metric_name].append(final_metrics[fold][model_type][metric_name]["avg"])
                    
        for model_type in avg_metrics.keys():
            for metric_name in avg_metrics[model_type].keys():
                if "classwise_avg" in final_metrics[fold][model_type][metric_name]:
                    avg_metrics[model_type][metric_name] = {
                        "avg": sum(np.mean(np.array(avg_metrics[model_type][metric_name]), axis=0))/3,
                        "classwise_avg": (np.mean(np.array(avg_metrics[model_type][metric_name]), axis=0)).tolist(),
                        "stdev": np.std(np.array(avg_metrics[model_type][metric_name]), ddof=1).tolist(),
                    }
                else:
                    avg_metrics[model_type][metric_name] = {
                        "avg": np.mean(np.array(avg_metrics[model_type][metric_name])),
                        "stdev": np.std(np.array(avg_metrics[model_type][metric_name]), ddof=1),
                    }
        
        final_metrics["average_over_folds"] = avg_metrics

    with open(f"cv_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        