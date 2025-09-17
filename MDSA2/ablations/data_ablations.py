import shutil
import torch

import sys
import os
from pathlib import Path

# scuffed system to avoid using packages, will fix later
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import os
from data_utils import get_dataloaders, get_aggregator_loader
from omegaconf import OmegaConf
from utils import set_deterministic, join, generate_rndm_path
from models import initialize_mdsa2

sys.path.pop(0)

import json
import argparse
from torch.utils.tensorboard import SummaryWriter # type: ignore
from tqdm import tqdm

if __name__ == "__main__":
    # run all tests
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder", type=str, default="sam2_tenfold", help="custom config folder")
    
    args = parser.parse_args()

    final_sa_metrics = {}

    max_epochs_sa = 20
    val_every_sa = 2

    modality_list = ["t2f", "t1c", "t1n", "t1w"]
    repeated_modalities = [[i for _ in range(3)] for i in modality_list]
    print("evaluating the following repeated modalities", repeated_modalities)

    for curr_modalities in repeated_modalities:
        print(f"---- training, then evaluating with modalities {curr_modalities} ----")
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", args.config_folder, "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        model_config.config_folder = args.config_folder
        model_config.fold_val = [0]
        model_config.fold_train = [a for a in range(10) if a != model_config.fold_val[0]]
        model_config.snapshot_path = generate_rndm_path("runs")
        model_config.modalities = curr_modalities

        # set batch size to 1 for comparison w/ unet
        details = OmegaConf.load(open(join(os.getenv("PROJECT_PATH", ""), 'MDSA2', 'config', args.config_folder, 'details.yaml'), 'r'))
        model_config = OmegaConf.merge(model_config, details)
        model_config.dataset = "brats_africa"

        set_deterministic(42)
        train_loader, val_loader, file_paths = get_dataloaders(model_config, verbose=False)
        print("file paths to verify: ", file_paths["train"][0])
        mdsa2 = initialize_mdsa2(model_config, dataloaders=(train_loader, val_loader), verbose=False)
        
        save_folder = model_config.snapshot_path
        print("writing to", save_folder)

        OmegaConf.save(model_config, join(save_folder, "model_config.yaml"))

        writer = SummaryWriter(log_dir=save_folder)
        best_metrics_found = {}

        for epoch in range(max_epochs_sa):
            print(f"--- Epoch {epoch+1}/{max_epochs_sa} ---")
            avg_train_loss, avg_batch_time = mdsa2.train_loop_sa(train_loader)
            print(f"avg train loss: {avg_train_loss}, avg batch time: {avg_batch_time}")
            writer.add_scalar("train/avg_loss", avg_train_loss, epoch)
            writer.add_scalar("train/avg_batch_time", avg_batch_time, epoch)

            if (epoch + 1) % val_every_sa == 0:
                print("validating...")
                json_metrics_sa2 = mdsa2.val_loop_sa(val_loader)
                os.makedirs(save_folder, exist_ok=True)

                for metric_name in json_metrics_sa2.keys():
                    if "classwise_avg" in json_metrics_sa2[metric_name]:
                        writer.add_scalar(f"val/{metric_name}_avg", json_metrics_sa2[metric_name]["avg"], epoch)
                        for class_idx, class_avg in enumerate(json_metrics_sa2[metric_name]["classwise_avg"]):
                            writer.add_scalar(f"val/{metric_name}_class_{class_idx}", class_avg, epoch)
                    else:
                        writer.add_scalar(f"val/{metric_name}_avg", json_metrics_sa2[metric_name]["avg"], epoch)

                if (best_metrics_found == {}) or (json_metrics_sa2["dice"]["avg"] > best_metrics_found["dice"]["avg"]):
                    best_metrics_found = json_metrics_sa2
                    model_path = join(save_folder, 'best_model_sam2.pth')

                    print(f"saving NEW BEST model {json_metrics_sa2['dice']['avg']} to {model_path}")
                    # save model
                    if model_config.rank > 0:
                        mdsa2.sam2_model.save_lora_parameters(model_path)
                    else:
                        torch.save(mdsa2.sam2_model.predictor.model.state_dict(), model_path)

                    with open(join(save_folder, f"best_metrics.json"), "w") as f:
                        json.dump(best_metrics_found, f, indent=4)

        final_sa_metrics[f"modalities_{model_config.modalities}"] = best_metrics_found
        writer.close()

    with open("./runs/sa2_ablations.json", "w") as f:
        json.dump(final_sa_metrics, f)