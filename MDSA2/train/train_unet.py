import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNetWrapper
from omegaconf import OmegaConf
import yaml
from utils import join, generate_rndm_path
from data_utils import get_unet_loader
import json
import time

if __name__ == "__main__":
    fold_eval = 0

    volumes_to_collect = yaml.load(open(join(os.getenv("PROJECT_PATH", ""), "volumes_to_collect.yaml"), 'r'), Loader=yaml.FullLoader)
    config = OmegaConf.create({
        "model_type": "DynUNet",
        "use_ref_volume": True,
        "batch_size": 1,
        "max_epochs": 100,
        "roi": [128, 128, 128],
        "sw_batch_size": 1,
        "infer_overlap": 0.5,
        "fold_val": fold_eval,
        "volumes_to_collect": volumes_to_collect,
        "model_name": "unet"
    })
    
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

    train_loader, val_loader = get_unet_loader(batch_size=config.batch_size, fold_train=config.fold_train,
                                       fold_val=config.fold_val, roi=config.roi, modalities=['t2f', 't1c', 't1n'])
    
    
    unet_model = UNetWrapper(train_loader=train_loader, val_loader=val_loader, loss_func="Dice", 
    use_scaler=True, optimizer="AdamW", config=config, **model_params)

    max_epochs = 100
    # plot overall loss
    best_dice = 0
    val_every = 5
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
                best_dice = json_metrics["dice"]["avg"]
                # save model
                runs_dir = generate_rndm_path(f"./runs/{config.model_type}_{fold_eval}")
                unet_model.save_model(epoch=epoch, save_dir=runs_dir)
                with open(join(runs_dir, f"best_metrics_{config.model_type}_{fold_eval}"), "w") as f:
                    json_metrics["epoch"] = epoch
                    json.dump(json_metrics, f, indent=4)

    print("best dice found during training", best_dice)