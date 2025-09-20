import argparse
import sys
import os
from pathlib import Path

# Add parent directory
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from models import UNetWrapper
from omegaconf import OmegaConf
import yaml
from utils import join, generate_rndm_path
from data_utils import get_unet_loader
import json
import time

sys.path.pop(0)

def single_fold(args):
    fold_train = [a for a in range(10) if a != args.fold_val]
    print("current train folds", fold_train)

    volumes_to_collect = yaml.load(open(join(os.getenv("PROJECT_PATH", ""), "volumes_to_collect.yaml"), 'r'), Loader=yaml.FullLoader)
    
    roi = [128, 128, 128]
    max_epochs = 100
    # plot overall loss
    best_dice = 0
    val_every = 5

    config = OmegaConf.create({
        "model_type": args.model_type,
        "use_ref_volume": False,
        "batch_size": 1,
        "max_epochs": max_epochs,
        "roi": roi,
        "sw_batch_size": 1,
        "infer_overlap": 0.5,
        "fold_val": [args.fold_val],
        "fold_train": fold_train,
        "volumes_to_collect": volumes_to_collect,
        "model_name": f"{args.model_type}_fold{args.fold_val}",
    })
    

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

    train_loader, val_loader = get_unet_loader(batch_size=config.batch_size, fold_train=config.fold_train,
                                       fold_val=config.fold_val, roi=config.roi, modalities=['t2f', 't1c', 't1n'])
    
    unet_model = UNetWrapper(train_loader=train_loader, val_loader=val_loader, train_params={
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-4
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "T_max": max_epochs,
            },
            "loss_func": "Dice", 
            "use_scaler": True
        }, config=config, verbose=False, **model_params)

    runs_dir = generate_rndm_path(f"runs_unet")
    print("writing to", runs_dir)
    
    for epoch in range(max_epochs):
        t_start = time.time()
        avg_run_loss = unet_model.train_epoch()
        print(
            "Epoch {}/{}".format(epoch, max_epochs),
            "loss: {:.4f}".format(avg_run_loss),
            "time {:.2f}s".format(time.time() - t_start),
        )
        if (epoch+1) % val_every == 0:
            json_metrics = unet_model.validate_epoch()
            if json_metrics["dice"]["avg"] > best_dice:
                print("new best dice", json_metrics["dice"]["avg"], "old best dice", best_dice)
                best_dice = json_metrics["dice"]["avg"]
                # save model
                unet_model.save_model(epoch=epoch, save_dir=runs_dir)
                with open(join(runs_dir, f"best_metrics_{config.model_type}_{fold_val}.json"), "w") as f:
                    json_metrics["epoch"] = epoch
                    json.dump(json_metrics, f, indent=4)
                    

    print("best dice found during training", best_dice)

# single-fold train
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_val", type=int, default=-1, help="Fold to evaluate. 0-9 or -1 for total CV")
    parser.add_argument('--model_type', type=str, default="DynUNet", help='model type (DynUNet or swinunetr)')

    args = parser.parse_args()

    if args.fold_val == -1:
        args.fold_val = 0
        folds_to_test = list(range(10))
    else:
        folds_to_test = [args.fold_val]
    for fold_val in folds_to_test:
        print(f"---- training then evaluating on fold {fold_val} ----")
        single_fold(args)