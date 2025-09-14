import sys
import os
from pathlib import Path

# Add parent directory
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models import UNetWrapper
from omegaconf import OmegaConf
import yaml
from utils import join
from data_utils import get_unet_loader
import json
import argparse
import numpy as np
sys.path.pop(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_folder', type=str, default="dynunet", help='folder where weights for all folds are stored')
    parser.add_argument('--model_type', type=str, default="DynUNet", help='model type (DynUNet or swinunetr)')
    parser.add_argument("--fold_val", type=int, default=-1, help="Fold to evaluate. -1 means do the full CV.")
    
    args = parser.parse_args()
    if args.fold_val == -1:
        folds_to_test = list(range(10))
    else:
        folds_to_test = [args.fold_val]

    volumes_to_collect = yaml.load(open(join(os.getenv("PROJECT_PATH", ""), "volumes_to_collect.yaml"), 'r'), Loader=yaml.FullLoader)
    roi = [128, 128, 128]
    

    if args.model_type == "DynUNet":
        kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        filters = [64, 96, 128, 192, 256, 384, 512]
        
        model_params = {
            "spatial_dims": 3,
            "in_channels": 3,
            "out_channels": 3,
            "kernel_size": kernels,
            "strides": strides,
            "upsample_kernel_size": strides[1:],
            "filters": filters,
            "norm_name": ("INSTANCE", {"affine": True}),
            "act_name": ("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            "deep_supervision": True,
            "deep_supr_num": 2,
            "res_block": False,
            "trans_bias": True,
        }
    elif args.model_type == "swinunetr":
        model_params = {
            "img_size": roi,
            "in_channels": 3,
            "out_channels": 3,
            "feature_size": 48,
            "use_checkpoint": True,
        }
    else:
        raise ValueError("model_type must be either DynUNet or swinunetr (case sensitive)")
    
    final_metrics = {}
    for fold_val in folds_to_test:
        fold_train = [a for a in range(10) if a != fold_val]

        train_loader, val_loader = get_unet_loader(batch_size=1, fold_train=fold_train,
                                           fold_val=fold_val, roi=roi, modalities=['t2f', 't1c', 't1n'])
        config = OmegaConf.create({
            "model_type": args.model_type, 
            "use_ref_volume": False,
            "batch_size": 1,
            "max_epochs": 100,
            "roi": [128, 128, 128],
            "sw_batch_size": 1,
            "infer_overlap": 0.5,
            "fold_val": fold_val,
            "fold_train": fold_train,
            "volumes_to_collect": volumes_to_collect,
            "model_name": f"{args.model_type}_fold{fold_val}",
        })
    
        unet_model = UNetWrapper(train_loader=train_loader, val_loader=val_loader, loss_func="Dice", 
        use_scaler=True, optimizer="AdamW", config=config, verbose=False, **model_params)
        weights_path = join(os.getenv("PROJECT_PATH", ""), "checkpoints", args.weights_folder, f"{args.model_type}_fold_{fold_val}.pt")
        unet_model.load_weights(weights_path=weights_path)
        
        json_metrics = unet_model.validate_epoch()
        metric_path = f"best_metrics_{config.model_type}_{fold_val}.json"
        print("sending metrics to ", metric_path)
        with open(metric_path, "w") as f:
            json.dump(json_metrics, f, indent=4)

        final_metrics[f"fold_{fold_val}"] = json_metrics

    # get the average from ALL folds
    if len(folds_to_test) > 1:
        avg_metrics = {}
        for fold in folds_to_test:
            fold_metrics = final_metrics[f"fold_{fold}"]
            for key, value in fold_metrics.items():
                """
                value: {
                    "classwise_avg": pyList,
                    "avg": sum(pyList)/3,
                    "stdev": self.meters[key].stdev.tolist(),
                }
                """
                for metric_type, metric_value in value.items():
                    if metric_type not in avg_metrics:
                        avg_metrics[metric_type] = []
                    if isinstance(metric_value, list):
                        avg_metrics[metric_type].append(np.array(metric_value))
                    else:
                        avg_metrics[metric_type].append(metric_value)

        for metric_type, metric_values in avg_metrics.items():
            if isinstance(metric_values[0], np.ndarray):
                # compute mean along dim 0
                avg_metrics[metric_type] = (np.mean(np.stack(metric_values, axis=0), axis=0)).tolist()
            else:
                avg_metrics[metric_type] = float(np.mean(metric_values))

        final_metrics["average_over_folds"] = avg_metrics

        avg_metric_path = f"average_metrics_{args.model_type}_all_folds.json"
        print("sending average metrics to ", avg_metric_path)
        with open(avg_metric_path, "w") as f:
            json.dump(final_metrics, f, indent=4)