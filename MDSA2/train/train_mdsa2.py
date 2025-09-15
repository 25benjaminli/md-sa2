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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __name__ == "__main__":
    # run all tests
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_val", type=int, default=-1, help="Fold to evaluate. -1 means do the full CV.")
    parser.add_argument("--config_folder", type=str, default="sam2_tenfold", help="custom config folder")
    
    args = parser.parse_args()

    if args.fold_val == -1:
        folds_to_test = list(range(10))
    else:
        folds_to_test = [args.fold_val]

    final_sa_metrics = {}

    max_epochs_sa = 20
    val_every_sa = 2
    
    max_epochs_agg = 100
    val_every_agg = 5
    batch_size_agg = 4

    # this is a two-stage process. first, train the SA2 model, then produce outputs for the aggregator model. 
    for fold_val in folds_to_test:
        print(f"---- training, then evaluating on fold {fold_val} ----")
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", args.config_folder, "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        model_config.config_folder = args.config_folder
        model_config.fold_train = [a for a in range(10) if a != fold_val]
        model_config.fold_val = [fold_val]
        model_config.snapshot_path = generate_rndm_path("runs")

        # set batch size to 1 for comparison w/ unet
        details = OmegaConf.load(open(join(os.getenv("PROJECT_PATH", ""), 'MDSA2', 'config', args.config_folder, 'details.yaml'), 'r'))
        model_config = OmegaConf.merge(model_config, details)
        model_config.dataset = "brats_africa"

        set_deterministic(42)
        train_loader, val_loader, file_paths = get_dataloaders(model_config, modality_to_repeat=-1, verbose=False)
        mdsa2 = initialize_mdsa2(model_config, dataloaders=(train_loader, val_loader))
        
        # save_folder = "./runs/9f22oklasc/cv_fold_0"
        save_folder = join(model_config.snapshot_path, f'cv_fold_{fold_val}')

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
                    sam2_path = join(save_folder, 'best_model_sam2.pth')

                    print(f"saving NEW BEST model {json_metrics_sa2['dice']['avg']} to {sam2_path}")
                    # save model
                    if model_config.rank > 0:
                        mdsa2.sam2_model.save_lora_parameters(sam2_path)
                    else:
                        torch.save(mdsa2.sam2_model.predictor.model.state_dict(), sam2_path)

                    with open(join(save_folder, f"best_metrics.json"), "w") as f:
                        json.dump(best_metrics_found, f, indent=4)

        final_sa_metrics[f"fold_{fold_val}"] = best_metrics_found
        # train the aggregator network. first, load all predictions into a new folder. 
        
        if os.path.exists(os.getenv("UNREFINED_VOLUMES_PATH", "")):
            shutil.rmtree(os.getenv("UNREFINED_VOLUMES_PATH", ""))

        del train_loader, val_loader
        train_loader, val_loader, file_paths = get_dataloaders(config=model_config, only_val_transforms=True, modality_to_repeat=-1, verbose=False)

        os.makedirs(os.getenv("UNREFINED_VOLUMES_PATH", ""), exist_ok=True)

        model_config.ft_ckpt = join(save_folder, 'best_model_sam2.pth')
        mdsa2 = initialize_mdsa2(model_config, dataloaders=(train_loader, val_loader))

        # now, using the weights of the current best model, we produce outputs for the entire training and validation set.
        print("generating and saving")
        mdsa2.generate_and_save(train_loader, val_loader, save_folder=os.getenv("UNREFINED_VOLUMES_PATH", ""))

        # swap to the new unrefined path
        file_paths = {
            "train": [
                {
                    "image": join(os.getenv("UNREFINED_VOLUMES_PATH", ""), os.path.basename(os.path.dirname(p["label"])) + "-pred.npy"),
                    "label": join(os.getenv("UNREFINED_VOLUMES_PATH", ""), os.path.basename(os.path.dirname(p["label"])) + "-label.npy"),
                    "ref_volume": join(os.getenv("UNREFINED_VOLUMES_PATH", ""), os.path.basename(os.path.dirname(p["label"])) + "-brain_volume.npy")
                } for p in file_paths["train"]
            ],
            "val": [
                {
                    "image": join(os.getenv("UNREFINED_VOLUMES_PATH", ""), os.path.basename(os.path.dirname(p["label"])) + "-pred.npy"),
                    "label": join(os.getenv("UNREFINED_VOLUMES_PATH", ""), os.path.basename(os.path.dirname(p["label"])) + "-label.npy"),
                    "ref_volume": join(os.getenv("UNREFINED_VOLUMES_PATH", ""), os.path.basename(os.path.dirname(p["label"])) + "-brain_volume.npy")
                } for p in file_paths["val"]
            ]
        }


        # delete and reload train and val loaders to reflect the new files
        # del train_loader, val_loader
        train_loader, val_loader = get_aggregator_loader(
            batch_size=batch_size_agg, 
            roi=(128,128,128),
            num_workers=0,
            file_paths=file_paths
        )

        print("length of train and val loaders", len(train_loader), len(val_loader))

        mdsa2 = initialize_mdsa2(model_config, dataloaders=(train_loader, val_loader), verbose=False)

        best_metrics_found = {}

        print("training the aggregation network")

        for epoch in tqdm(range(max_epochs_agg)):
            print(f"--- Epoch {epoch+1}/{max_epochs_agg} ---")
            avg_train_loss = mdsa2.unet_model.train_epoch()
            print(f"avg train loss: {avg_train_loss}")
            writer.add_scalar("agg_train/avg_loss", avg_train_loss, epoch)

            if (epoch + 1) % val_every_agg == 0:
                print("validating...")
                json_metrics_agg = mdsa2.val_loop_agg(val_loader)
                os.makedirs(save_folder, exist_ok=True)

                for metric_name in json_metrics_agg.keys():
                    if "classwise_avg" in json_metrics_agg[metric_name]:
                        writer.add_scalar(f"unet_val/{metric_name}_avg", json_metrics_agg[metric_name]["avg"], epoch)
                        for class_idx, class_avg in enumerate(json_metrics_agg[metric_name]["classwise_avg"]):
                            writer.add_scalar(f"unet_val/{metric_name}_class_{class_idx}", class_avg, epoch)
                    else:
                        writer.add_scalar(f"unet_val/{metric_name}_avg", json_metrics_agg[metric_name]["avg"], epoch)

                if (best_metrics_found == {}) or (json_metrics_agg["dice"]["avg"] > best_metrics_found["dice"]["avg"]):
                    best_metrics_found = json_metrics_agg
                    unet_path = join(save_folder, 'best_model.pt')

                    print(f"saving NEW BEST UNET model {json_metrics_agg['dice']['avg']} to {unet_path}")
                    # save model
                    torch.save(mdsa2.unet_model.state_dict(), unet_path)

                    with open(join(save_folder, f"best_agg_metrics.json"), "w") as f:
                        json.dump(best_metrics_found, f, indent=4)

        writer.close()