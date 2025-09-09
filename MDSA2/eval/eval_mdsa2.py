import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch.nn as nn
import torch
import numpy as np
import time
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
import argparse
from data_utils import get_dataloaders
from utils import register_net_sam2, set_deterministic
from omegaconf import OmegaConf
from metrics import AverageMeter, MetricAccumulator
import gc
import json
from collections import OrderedDict
from utils import generate_rndm_path, join
from typing import Tuple
from models import UNetWrapper, initialize_mdsa2
import yaml

def print_metric(metric_name, metric_arr):
    print(f"{metric_name} (overall average): ", round(np.mean(np.mean(metric_arr, axis=1)),4), "+-", round(np.std(np.mean(metric_arr, axis=1), ddof=1),4))
    print(f"{metric_name} (classwise mean):", np.mean(metric_arr, axis=0))
    print(f"{metric_name} (average values): ", np.mean(metric_arr, axis=1))
    print("---------")

def eval_loop(args, model_config):
    num_all_folds = 10
    model_config.fold_train = [i for i in range(num_all_folds) if i != args.fold_eval]
    model_config.fold_val = [args.fold_eval]

    mdsa2, train_loader, val_loader = initialize_mdsa2(model_config, use_unet=True)

    ############### define metrics
    
    metric_dict = OrderedDict({
        "dice": DiceMetric(include_background=True, get_not_nans=True, ignore_empty=True, reduction='mean_batch'), # BCHW[D]
        "hd95": HausdorffDistanceMetric(include_background=True, reduction="mean_batch",get_not_nans=True, percentile=95),
        "iou": MeanIoU(include_background=True, get_not_nans=True, reduction="mean_batch")
    })

    final_json_metrics_mdsa2 = {}
    final_json_metrics_sa2 = {}
    total_batches = 0
    mdsa2_dict = {key: [] for key in metric_dict.keys()}
    for idx, batch in enumerate(val_loader):
        for key in batch.keys():
            batch[key] = batch[key].to("cuda") if isinstance(batch[key], torch.Tensor) else batch[key]
        
        json_metrics_sa2, json_metrics_mdsa2 = mdsa2(batch=batch)
        # merge json_metrics with final_json_metrics_mdsa2
        for key in json_metrics_mdsa2.keys():
            if key not in final_json_metrics_mdsa2:
                final_json_metrics_mdsa2[key] = json_metrics_mdsa2[key]
            else:
                final_json_metrics_mdsa2[key].update(json_metrics_mdsa2[key])

        for key in json_metrics_sa2.keys():
            if key not in final_json_metrics_sa2:
                final_json_metrics_sa2[key] = json_metrics_sa2[key]
            else:
                final_json_metrics_sa2[key].update(json_metrics_sa2[key])

        total_batches += batch["image"].shape[0]
        gc.collect()
        torch.cuda.empty_cache()

    # save final_json_metrics_mdsa2
    save_path = "final_json_metrics_mdsa2.json"
    with open(save_path, "w") as f:
        json.dump(final_json_metrics_mdsa2, f, indent=4)

    print_metric("dice", mdsa2_dict["dice"])
    print_metric("iou", mdsa2_dict["iou"])
    print_metric("hd95", mdsa2_dict["hd95"])

    # save final_json_metrics_sa2
    save_path = "final_json_metrics_sa2.json"
    with open(save_path, "w") as f:
        json.dump(final_json_metrics_sa2, f, indent=4)

    return final_json_metrics_mdsa2, final_json_metrics_sa2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold_eval', type=int, default=0, help='fold number')
    parser.add_argument('--do_cross_validation', action='store_true', help='do cross validation')
    parser.add_argument('--save_volumes', type=str, default="", help="list of volume names that you want to save")
    parser.add_argument('--config_folder', type=str, default='sam2_tenfold', help='folder containing configurations for the experiment')

    args = parser.parse_args()
    # merge more args into args
    
    args.roi = [128, 128, 128]
    args.sw_batch_size = 1
    args.infer_overlap = 0.5

    model_config = join(os.getenv("PROJECT_PATH"), "MDSA2", "config", args.config_folder, "config_train.yaml")
    model_config = OmegaConf.load(model_config)
    model_config.save_volumes = args.save_volumes.split(",") if len(args.save_volumes) > 0 else []

    set_deterministic(model_config.seed)

    ############### overriding train modifications
    
    model_config.num_slices = 155
    model_config.conf_threshold = 0.5

    overall_md_metrics, overall_sa_metrics = {"dice": [], "hd95": [], "iou": []}, {"dice": [], "hd95": [], "iou": []}
    torch.cuda.reset_peak_memory_stats()

    if args.do_cross_validation:
        for i in range(10):
            args.fold_eval = i
            final_json_metrics_mdsa2, final_json_metrics_sa2 = eval_loop(args, model_config)
            for key in final_json_metrics_mdsa2.keys():
                print(f"md averagemeter for {i} {key}", final_json_metrics_mdsa2[key], sum(final_json_metrics_mdsa2[key])/3)
                print(f"sa averagemeter for {i} {key}", final_json_metrics_sa2[key], sum(final_json_metrics_sa2[key])/3)
                overall_md_metrics[key].append(final_json_metrics_mdsa2[key])
                overall_sa_metrics[key].append(final_json_metrics_sa2[key])

        # convert each to numpy array for the final calculation
        for key in overall_md_metrics.keys():
            overall_md_metrics[key] = np.array(overall_md_metrics[key])
            overall_sa_metrics[key] = np.array(overall_sa_metrics[key])

            classwise_average_md = np.mean(overall_md_metrics[key], axis=0) # (3,)
            classwise_average_sa = np.mean(overall_sa_metrics[key], axis=0)

            # calculate classwise standard deviation
            classwise_std_md = np.std(overall_md_metrics[key], axis=0, ddof=1) # (3,)

            # calculate standard deviation over the average of all classes
            avg_dice = np.mean(overall_md_metrics[key], axis=1) # (10,)
            avg_sa_dice = np.mean(overall_sa_metrics[key], axis=1)
            print(f"avg {key}", avg_dice)
            print(f"avg sa {key}", avg_sa_dice)
            std_avg_dice = np.std(avg_dice, ddof=1) # scalar

            print(f"overall {key} metrics average (md)", classwise_average_md, sum(classwise_average_md)/3)
            print("classwise std", classwise_std_md, "std over avg", std_avg_dice)
            print("classwise sem", classwise_std_md/np.sqrt(10), "sem over avg", std_avg_dice/np.sqrt(10))
    else:
        md_averagemeters, sa_averagemeters = eval_loop(args, model_config)

    torch.cuda.synchronize()
    print("peak memory allocated", torch.cuda.max_memory_reserved() / 1024**3, "GB")