# PEP 8 import format

# standard library imports
import argparse
import json
import os
import time
import yaml
import sys

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
import monai

# data analysis imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from dotenv import load_dotenv
import pickle
# my own file imports
sys.path.append('..')

from metrics import dice_coef_not_nans, iou_not_nans, cf_matrix_metric, AverageMeter
from data_utils import get_dataloaders

from utils import ( 
    generate_rndm_path, 
    clear_cache, 
    set_deterministic, 
    register_net,
    register_net_sam2
)

sys.path.append('./eval')

load_dotenv(override=True)

# print to the nearest 4 decimal places
np.set_printoptions(precision=4)

def print_metric(metric_name, metric_arr):
    print(f"{metric_name} (overall average): ", round(np.mean(np.mean(metric_arr, axis=1)),4), "+-", round(np.std(np.mean(metric_arr, axis=1), ddof=1),4))
    print(f"{metric_name} (classwise mean):", np.mean(metric_arr, axis=0))
    print(f"{metric_name} (average values): ", np.mean(metric_arr, axis=1))
    print("---------")

def eval_model(model, dataloaders, config):
    train_loader, val_loader, file_paths = dataloaders

    print("dataloaders lengths train and val", len(train_loader), len(val_loader))

    snapshot_path = generate_rndm_path(os.path.join("runs", "brats_africa", config.run_name), make_downstream_folder=False)
    print("writing to", snapshot_path)

    # don't use meters
    metrics = {
        "dice": [],
        "iou": [],
        "hd95": [],
        "inference_time": [], # per batch
        "volume_names": []
    }

    model.eval()

    for i, batch in enumerate(val_loader):
        image, label = batch['image'].cuda(), batch['label'].cuda()
        print("image, label shapes", image.shape, label.shape)
        arr_pred = torch.zeros(size=(label.shape)).cuda()
        # arr_low_res = torch.zeros(size=(image.shape[0], image.shape[1], image.shape[2] // 4, image.shape[3] //4, image.shape[4])).cuda()

        t1 = time.time()
        # print("image shape", image.shape)
        for current_slice in range(image.shape[4]):
            image_sliced, label_sliced = image[...,current_slice], label[...,current_slice]
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                    if config.model_type == "sam_1":
                        outputs = model(image_sliced, None, out_shape=(config.img_size,config.img_size)) # can also try out_shape 512, 512?
                        arr_pred[...,current_slice] = torch.sigmoid(outputs['masks']) > config.conf_threshold

                    elif config.model_type == "sam_2":
                        outputs = model(image_sliced, repeat_image=True) # multimask_output=False,boxes=bbox_sam,repeat_image=False
                        arr_pred[...,current_slice] = torch.sigmoid(outputs['prd_masks']) > config.conf_threshold

        metrics["inference_time"].append(time.time() - t1)

        arr_pred = nn.functional.interpolate(input=arr_pred, size=(224, 224, config.num_slices), mode="nearest-exact")
        label = nn.functional.interpolate(input=label, size=(224, 224, config.num_slices), mode="nearest-exact")
        # interpolate the image to the correct size
        image = nn.functional.interpolate(input=image, size=(224, 224, config.num_slices), mode="trilinear")

        print("--- VOLUME NAMES --- ", batch["image_title"])


        # within the batch, calculate per volume
        for index in range(image.shape[0]):
            # calculate dsc
            volume_name = batch["image_title"][index]
            metrics["volume_names"].append(volume_name)

            dsc_fn = DiceMetric(include_background=True, ignore_empty=True, reduction='mean_batch') # BCHW[D]
            dsc_fn(y_pred=arr_pred[index].unsqueeze(0), y=label[index].unsqueeze(0))
            val_dice_3D = dsc_fn.aggregate()

            # calculate iou
            iou_fn = monai.metrics.MeanIoU(include_background=True, reduction='mean_batch')

            iou_fn(y_pred=arr_pred[index].unsqueeze(0), y=label[index].unsqueeze(0))
            val_iou = iou_fn.aggregate()

            # calculate hd95
            hd95_fn = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, reduction='mean_batch')
            hd95_fn(y_pred=arr_pred[index].unsqueeze(0), y=label[index].unsqueeze(0))
            val_hd95 = hd95_fn.aggregate()

            metrics["dice"].append(val_dice_3D.squeeze().cpu().numpy().tolist())
            metrics["iou"].append(val_iou.squeeze().cpu().numpy().tolist())
            metrics["hd95"].append(val_hd95.squeeze().cpu().numpy().tolist())

            avg_dice = "{:.3f}".format(np.mean(val_dice_3D.squeeze().cpu().numpy()),3)
            # print("avg dice", avg_dice)
            arr_np = arr_pred[index].detach().cpu().numpy()
            label_np = label[index].detach().cpu().numpy() # 
            volume_image = image[index].detach().cpu().numpy() # 3xIMGxIMGxSLICE

            if volume_name in config.saved_volumes:
                # save to an np array
                save_path = os.path.join(snapshot_path, "predictions")
                os.makedirs(save_path, exist_ok=True)
                fname = f"{volume_name}_fold_{config.fold_val[0]}_{avg_dice}.npy"
                
                np.save(os.path.join(save_path, fname), arr_np)
            
            elif config.saved_volumes == "all": # save all volumes for CV
                # print(f"saved {volume_name}")
                # print("shapes of the predicted, label, image", arr_np.shape, label_np.shape, volume_image.shape)
                # the shapes of all these arrays must be the same
                assert arr_np.shape == label_np.shape == volume_image.shape, f"shapes of the predicted, label, image are not the same {arr_np.shape} {label_np.shape} {volume_image.shape}"
                np.save(f'{os.getenv("UNREFINED_VOLUMES_PATH")}/pred_{volume_name}', arr_np)
                np.save(f'{os.getenv("UNREFINED_VOLUMES_PATH")}/label_{volume_name}', label_np)
                # save modalities
                np.save(f'{os.getenv("UNREFINED_VOLUMES_PATH")}/brain_volume_{volume_name}', volume_image)

    # printing metrics now
    print("---- METRICS ----")
    print("Volume names list:", metrics["volume_names"])
    print_metric("dice", metrics["dice"])
    print_metric("iou", metrics["iou"])
    print_metric("hd95", metrics["hd95"])
    print(f"Average inference time per batch size {config.batch_size_val}:", np.mean(metrics["inference_time"]))

    # save the metrics
    saved_metrics = {
        # "dice": metrics["dice"],
        # "iou": metrics["iou"],
        # "hd95": metrics["hd95"],
        "inference_time": metrics["inference_time"],
        "volume_names": metrics["volume_names"]
    }

    dice_volname_dict = {k: {"dice": metrics["dice"][idx], "iou": metrics["iou"][idx], "hd95": metrics["hd95"][idx]} for idx, k in enumerate(metrics["volume_names"])}

    # merge saved_metrics and dice_volname_dict
    final_dict = {**dice_volname_dict, **saved_metrics}
    with open(snapshot_path + "/metrics.json", "w") as f:
        json.dump(final_dict, f, indent=4)
        print("metrics saved to", snapshot_path + "/metrics.json")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default='fold4_curr', help='folder containing configurations for the experiment')
    parser.add_argument('--save_volumes', type=str, help='whether to save volumes. separate volume names by comma')
    parser.add_argument('--for_cross_val', action='store_true', help='use cross validation')
    args = parser.parse_args()


    config_final = OmegaConf.load(open(os.path.join('../train', 'runs', "brats_africa", args.run_name, 'current_config.yaml'), 'r'))
    config_final.run_name = args.run_name
    if args.save_volumes is None:
        print("not saving volumes")
        config_final.saved_volumes = []
    else:
        if "," in args.save_volumes:
            config_final.saved_volumes = args.save_volumes.split(",")
        elif args.save_volumes == "all":
            config_final.saved_volumes = "all"

    model_str = 'best_model_lora.pth' if config_final.rank > 0 and config_final.model_type == 'sam_2' else 'best_model_sam2.pth' 
    if config_final.model_type == 'sam_1':
        model_str = 'best_model_sam1.pth'
    # model_str = 'best_model_lora.pth'
    config_final.ft_ckpt = os.path.join('../train', 'runs', config_final.dataset, args.run_name, model_str)

    assert os.path.exists(config_final.ft_ckpt), f"finetuned ckpt {config_final.ft_ckpt} does not exist"
    print("finetuned ckpt", config_final.ft_ckpt)


    # ! Scuffed: because it's val force the num slices to be 155
    # These are overrides for the validation processs
    
    config_final.num_slices = 155
    config_final.conf_threshold = 0.5

    save_to_continuous = os.path.join('../eval', 'runs', config_final.dataset, args.run_name)

    volumes_to_collect = yaml.load(open(os.path.join(os.getenv("PROJECT_PATH"), "volumes_to_collect.yaml"), 'r'), Loader=yaml.FullLoader)

    config_final.volumes_to_collect = volumes_to_collect

    # dump omegaconf datapath
    # dump config final to the folder
    OmegaConf.save(config_final, os.path.join(os.getenv("DATA_PATH"), 'current_eval_config.yaml'))

    set_deterministic(config_final.seed)

    if config_final.model_type == "sam_1":
        model = register_net(model_config = config_final)

    else:
        model = register_net_sam2(model_config = config_final)

    dataloaders = get_dataloaders(config_final, use_preprocessed=True)
    
    eval_model(model=model, dataloaders=dataloaders, config=config_final)

    if args.for_cross_val:
        # swap fold_tain and fold_val
        # the reason why we're doing this is because the training images need to be used to train the aggregator
        fold_train = config_final.fold_train
        fold_val = config_final.fold_val

        config_final.fold_train = fold_val
        config_final.fold_val = fold_train
        print("starting second eval for", config_final.fold_val)

        clear_cache()
        set_deterministic(config_final.seed)

        if config_final.model_type == "sam_1":
            model = register_net(model_config = config_final)

        else:
            model = register_net_sam2(model_config = config_final)

        dataloaders = get_dataloaders(config_final, use_preprocessed=True)
        
        eval_model(model=model, dataloaders=dataloaders, config=config_final)