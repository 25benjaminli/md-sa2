# PEP 8 import format

# standard library imports
import argparse
import json
import os
import random
import shutil
import time
import monai.transforms
import yaml
import sys

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2

# import tensorflow as tf

from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
import monai.transforms as transforms
from monai.data import DataLoader
import monai
# import map_transforms
from monai.transforms import MapTransform
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

from metrics import dice_coef_not_nans, iou_not_nans, cf_matrix_metric, AverageMeter, binarize
from data_utils import get_dataloaders, AddNameField, ConvertToMultiChannel, datafold_read, RepeatModality

from utils import ( 
    generate_snapshot_path, 
    clear_cache, 
    set_deterministic, 
    register_net,
    register_net_sam2,
    register_medsam
)
#!! Your own imports

sys.path.append('./eval')

load_dotenv(override=True)


def eval_model(model, val_loader, config):
    # dataloaders are 3D

    print("dataloaders lengths val", len(val_loader))

    # TODO: manually randomize order in which the model sees the slices

    dice_meter_3D = AverageMeter(name='3D dice')
    dice_array = []

    model.eval()
    # validation loop

    t1 = time.time()
    orig_time = time.time()
    val_len = len(val_loader)
    dict_scores = {}


    print("threshold: ", config.conf_threshold)

    # ignore np errors
    np.seterr(divide='ignore', invalid='ignore')

    binary_dice_meter = AverageMeter(name='binary dice')
    inference_duration_list = []
    for i_val, batch in enumerate(val_loader):
        image, label = batch["image"].cuda(), batch["label"].cuda()

        # first two shape stay the same, after that it gets divideed by 4
        batch_size, num_classes, H, W, D = label.shape
        
        arr_low_res = torch.zeros(size=(batch_size, num_classes, H//4, W//4, D)).cuda()
        binary_label_volume = torch.zeros(size=(batch_size, 1, H, W, D)).cuda()

        print("arr low res, binary shape", arr_low_res.shape, binary_label_volume.shape, "min of image", image.min(), "max of image", image.max())
        # matplotlib display a slice

        print("image shape", image.shape, "label shape", label.shape)

        
        for current_slice in range(image.shape[4]):
            image_sliced, label_sliced = image[...,current_slice], label[...,current_slice]

            for label_channel in range(num_classes):
                # print("label sliced shape", label_sliced.shape)
                binary_label = label_sliced[:, label_channel].cpu().numpy()

                # print("binary label shape", binary_label.shape)

                bbox_sam = monai.transforms.BoundingRect()(binary_label)
                # convert each one in xmin, ymin, xmax, ymax from the ymin,ymax,xmin,xmax format
                for i in range(bbox_sam.shape[0]):
                    ymin, ymax, xmin, xmax = bbox_sam[i]
                    # edit in place
                    bbox_sam[i][0] = xmin
                    bbox_sam[i][1] = ymin
                    bbox_sam[i][2] = xmax
                    bbox_sam[i][3] = ymax

                bbox_np = bbox_sam.copy()

                bbox_sam = torch.from_numpy(bbox_sam).cuda().unsqueeze(1)

                with torch.inference_mode():
                    t1 = time.time()
                    # print("image sliced shape", image_sliced.shape, "bbox sam shape", bbox_sam.shape)

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.use_amp):
                        # t1 = time.time()
                        if config_final.model_type == "medsam" or config_final.model_type == "sam_1":
                            outputs = model(image_sliced, multimask_output=False,image_size=config.img_size, boxes=bbox_sam) # can also try out_shape 512, 512?

                        elif config_final.model_type == "sam_2":
                            # bbox_sam = bbox_sam.squeeze(0)
                            outputs = model(image_sliced, multimask_output=False,boxes=bbox_sam,repeat_image=False)
                            

                    # !! avoid calculating loss for now, coefficient more important
                    inference_duration_list.append(time.time()-t1)
                    arr_low_res[:, label_channel, :, :, current_slice] = (torch.sigmoid(outputs['low_res_logits']) > config.conf_threshold).squeeze(1)

        arr_pred = nn.functional.interpolate(input=arr_low_res, size=(224, 224, config.num_slices), mode="nearest-exact")
        label = nn.functional.interpolate(input=label, size=(224, 224, config.num_slices), mode="nearest-exact")
        assert arr_pred.shape == label.shape, f"arr pred shape {arr_pred.shape} not equal to label shape {label.shape}"

        for b in range(batch_size):
            dsc_fn = DiceMetric(include_background=True, get_not_nans=True, ignore_empty=True, reduction='mean_batch') # BCHW[D]
            dsc_fn(y_pred=arr_pred[b].unsqueeze(0), y=label[b].unsqueeze(0))
            val_dice_3D, val_dice_3D_not_nans = dsc_fn.aggregate()
            dice_array.append(val_dice_3D.squeeze().cpu().numpy().tolist())
       
        # calculate non batched dice
        # val_dice_nonbatched, val_dice_not_nans_nonbatched = [], []
        st = f"{val_dice_3D.cpu().numpy()}, {sum(val_dice_3D.cpu().numpy())/3}"
        dice_meter_3D.update(val_dice_3D.squeeze().cpu().numpy(), n=val_dice_3D_not_nans.squeeze().cpu().numpy())

        for i in range(image.shape[0]):

            arr_transp, label_transp = arr_pred[i].permute((0, 3, 1, 2)), label[i].permute((0, 3, 1, 2))
            volume_pred, volume_label = arr_transp.cpu().numpy(), label_transp.cpu().numpy()
            # must be the same shape
            assert len(volume_pred.shape) == 4, f"volume shape should be 4 (3xIMGxIMGxSLICE), got {volume_pred.shape}"
            assert volume_pred.shape == volume_label.shape, f"pred shape of {volume_pred.shape} must equal label shape of {volume_label.shape}"
            # os.makedirs(os.getenv("UNREFINED_VOLUMES_PATH"), exist_ok=True)
            if config.fold_val[0] in config.volumes_to_collect.keys():
                # print(batch_data["image_title"][0])
                if batch["image_title"][i] in config.volumes_to_collect[config.fold_val[0]]:
                    print("saving volume", batch["image_title"][i], volume_pred.shape)
                    # save the volume to config.save_dir, volume name
                    predictions_base = os.path.join(os.getenv("PROJECT_PATH"),"extra_software", "visualization", "volumes", config.model_type)

                    # create folder under predictions base called sa if it doesn't exist
                    os.makedirs(predictions_base, exist_ok=True)

                    np.save(os.path.join(predictions_base,f"{batch['image_title'][i]}_fold_{config.fold_val[0]}.npy"), volume_pred)

        # print(batch["image_title"])
        print(f"batch {batch['image_title']} {i_val}/{val_len} dice took {time.time()-t1}", st)

        t1 = time.time()
            

    final_val_dice_3D = dice_meter_3D.avg

    print(f"3D val dice, {final_val_dice_3D} {sum(final_val_dice_3D)/3}")
    print("dice array", dice_array)
    print(f"Took {time.time()-orig_time}")
    print("length of dictionary", len(dict_scores.keys()))

    avg_2d_inf = sum(inference_duration_list)/len(inference_duration_list)
    avg_3d_inf = avg_2d_inf*155/config.batch_size_val
    print("average inference duration", avg_3d_inf)

    dict_scores["summary"] = {
        "3D": final_val_dice_3D.tolist(),
        "3D average": float(sum(final_val_dice_3D)/3),
        # "iou average": float(sum(iou_meter.avg)/3),
        # "hd95 average": float(sum(hd95_meter.avg)/3),
    }

    return dict_scores

def get_val_dataloader(config, use_preprocessed=False):

    val_transforms = transforms.Compose(
        [
            AddNameField(keys=["image"]),
            transforms.LoadImaged(keys=["image", "label"]), # assuming loading multiple at the same time.
            ConvertToMultiChannel(keys="label", use_softmax=False),
            transforms.CastToTyped(keys=["image", "label"], dtype=(torch.float16, torch.uint8)),
            transforms.ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )
    # ! TODO: change learning rate
    
    json_path = os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2", "train.json")
    dataset_path = os.getenv("DATASET_PATH") if not use_preprocessed else os.getenv("PREPROCESSED_PATH")
    
    train_files, validation_files = datafold_read(dataset_path=dataset_path, fold_val=config.fold_val, fold_train=config.fold_train,
                                                              modalities=config.modalities, json_path=json_path)

    print("num of validation files", len(validation_files), "batch size val",  config.batch_size_val)
    val_dataset = monai.data.Dataset(data=validation_files, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False, num_workers=0)

    # need to resolve training slow ness??
    return val_loader

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # therefore, xmin and ymin are the top left corner
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--net", type=str, choices=['sam_2', 'medsam', 'sam_1'], help='net name')
    parser.add_argument("--for_cross_val", action='store_true', help='whether to use cross validation')
    args = parser.parse_args()

    # clear_cache()
    model_paths = {
        "sam_2": os.path.join(os.getenv("PROJECT_PATH"), "MDSA2", "checkpoints", "sam2_hiera_tiny.pt"),
        "medsam": os.path.join(os.getenv("PROJECT_PATH"), "MDSA2", "checkpoints", "medsam_vit_b.pth"),
        "sam_1": os.path.join(os.getenv("PROJECT_PATH"), "MDSA2", "checkpoints", "sam_vit_b_01ec64.pth")
    }

    config_final = OmegaConf.create({
        "batch_size_val": 4,
        "fold_val": [0],
        "fold_train": [-1],
        "modalities": ["t2f", "t2f", "t2f"],
        "use_preprocessed": True,
        "img_size": 384,
        "use_amp": True,
        "model_type": args.net,
        "conf_threshold": 0.5,
        "num_slices": 155,
        # need to fix this reference
        "ft_ckpt": None,
        "sam_ckpt": model_paths[args.net],
        "vit_name": "tiny" if args.net == "sam_2" else "vit_b",
        "compile": False,
        "rank": -1,
        "custom_name": "",
        "dataset": "brats_africa",
        "num_classes": 3,
        "volumes_to_collect": yaml.load(open(os.path.join(os.getenv("PROJECT_PATH"), "volumes_to_collect.yaml"), 'r'), Loader=yaml.FullLoader)
    })
    
    set_deterministic(1234)
    print("preprocessing")

    if config_final.model_type == "sam_2":
        model = register_net_sam2(model_config = config_final)
    elif config_final.model_type == "medsam" or config_final.model_type == "sam_1":
        model = register_medsam(model_config = config_final)

    if args.for_cross_val:
        # get all the folds
        scores = []
        ious = []
        hd95s = []
        for fold in range(10):
            config_final.fold_val = [fold]
            val_dataloader = get_val_dataloader(config_final, use_preprocessed=True)
            dict_scores = eval_model(model=model, val_loader=val_dataloader, config=config_final)
            scores.append(dict_scores["summary"]["3D average"])
            ious.append(dict_scores["summary"]["iou average"])
            hd95s.append(dict_scores["summary"]["hd95 average"])
        scores = np.array(scores)
        print("scores", scores)
        print("mean", np.mean(scores))
    else:
        val_dataloader = get_val_dataloader(config_final, use_preprocessed=True)

        dict_scores = eval_model(model=model, val_loader=val_dataloader, config=config_final)
