import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet_models import UNetWrapper
from omegaconf import OmegaConf
import yaml
from utils import join
from data_utils import get_unet_loader
import json

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

    unet_model.load_weights(weights_path="../weights/dynunet_100/fold_0.pt")
    
    json_metrics = unet_model.validate_epoch()

    with open(f"best_metrics_{config.model_type}_{fold_eval}.json", "w") as f:
        json.dump(json_metrics, f, indent=4)