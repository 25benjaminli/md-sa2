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
from utils import set_deterministic, join, generate_rndm_path, register_net_sam1
from metrics import MetricAccumulator, AverageMeter
from models import initialize_mdsa2

sys.path.pop(0)

import json
import argparse
from torch.utils.tensorboard import SummaryWriter # type: ignore
from tqdm import tqdm
import torch.functional as F
import time
import monai
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def train_loop_sa1(sa_model,
                   optimizer,
                   loss_fct,
                   scaler,
                   train_loader,
                   model_config):
    total_train_loss = 0
    inference_meter = AverageMeter("inference")
    for i_batch, batch in tqdm(enumerate(train_loader)):
        image = batch["image"].cuda()
        size_resized = (model_config.img_size // 4, model_config.img_size // 4, image.shape[-1])

        label = nn.functional.interpolate(batch["label"].cuda(), size_resized, mode="nearest-exact")
        # print("label shape", label.shape)
        running_train_batch_loss = 0
        t1 = time.time()

        for current_slice in range(image.shape[4]):
            image_sliced, label_sliced = image[...,current_slice], label[...,current_slice] # shape (4, 3, X, X)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=model_config.use_amp):
                outputs = sa_model(image_sliced, None, out_shape=(model_config.img_size,model_config.img_size)) # can also try out_shape 512, 512? about 0.11 sec per batch of 4
                loss = loss_fct(outputs['low_res_logits'], label_sliced)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_train_batch_loss += loss.item()

        inference_meter.update(time.time()-t1, 1)
        total_train_loss += running_train_batch_loss/image.shape[4]

    return total_train_loss/len(train_loader), inference_meter.avg

def val_loop_sa1(sa_model,
                 val_loader,
                 model_config):
    
    accumulator = MetricAccumulator()

    for i_val, batch in enumerate(val_loader):
        image, label = batch["image"].cuda(), batch["label"].cuda()
        print("image, label shape", image.shape, label.shape)

        arr_pred = torch.zeros(size=(image.shape)).cuda()

        for current_slice in range(image.shape[4]):
            image_sliced, label_sliced = image[...,current_slice], label[...,current_slice]
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=model_config.use_amp):
                    outputs = sa_model(image_sliced, None, out_shape=(model_config.img_size,model_config.img_size)) # can also try out_shape 512, 512?

                high_res_pred = torch.sigmoid(outputs['masks']) > model_config.conf_threshold
                arr_pred[..., current_slice] = high_res_pred

        accumulator.update(y_pred=arr_pred, y_true=label)

    return accumulator.get_metrics()

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # therefore, xmin and ymin are the top left corner
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2) # type: ignore
    )

def val_loop_medsam(sa_model,
                 val_loader,
                 model_config,
                 ):
    
    # the main difference here is that we need to loop thru every class individually and use bbox prompts
    
    accumulator = MetricAccumulator()

    for i_val, batch in enumerate(val_loader):
        image, label = batch["image"].cuda(), batch["label"].cuda()
        t1 = time.time()
        arr_low_res = torch.zeros(size=(label.shape[0], label.shape[1], label.shape[2]//4, label.shape[3]//4, label.shape[4])).cuda()
        for current_slice in tqdm(range(image.shape[4])):
            image_sliced, label_sliced = image[...,current_slice], label[...,current_slice]

            for label_channel in range(3):
                binary_label = label_sliced[:, label_channel].cpu().numpy()
                bbox_sam = monai.transforms.BoundingRect()(binary_label) # type: ignore
                # convert each one in xmin, ymin, xmax, ymax from the ymin,ymax,xmin,xmax format
                for i in range(bbox_sam.shape[0]):
                    ymin, ymax, xmin, xmax = bbox_sam[i]
                    bbox_sam[i][0] = xmin
                    bbox_sam[i][1] = ymin
                    bbox_sam[i][2] = xmax
                    bbox_sam[i][3] = ymax

                bbox_np = bbox_sam.copy()
                bbox_sam = torch.from_numpy(bbox_sam).cuda().unsqueeze(1)

                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=model_config.use_amp):
                    outputs = sa_model(image_sliced, multimask_output=False,image_size=model_config.img_size, boxes=bbox_sam) # can also try out_shape 512, 512?

                    arr_low_res[:, label_channel, :, :, current_slice] = (torch.sigmoid(outputs['low_res_logits']) > model_config.conf_threshold).squeeze(1)

                # uncomment for visualizing boxes
                # if current_slice >50:
                #     fig, axes = plt.subplots(1, 2)
                #     # print("bbox sam shape", bbox_np.shape, bbox_np[0])

                #     axes[0].imshow(binary_label[0], cmap="gray")
                #     axes[1].imshow(arr_low_res[0, label_channel, :, :, current_slice].cpu().numpy(), cmap="gray")
                #     show_box(bbox_np[0], axes[0])
                #     plt.show()

        arr_pred = nn.functional.interpolate(input=arr_low_res, size=(224, 224, model_config.num_slices), mode="nearest-exact")
        label = nn.functional.interpolate(input=label, size=(224, 224, model_config.num_slices), mode="nearest-exact")

        accumulator.update(y_pred=arr_pred, y_true=label, time_spent=(time.time()-t1)/image.shape[0])

    return accumulator.get_metrics()

# this is fully fine-tuned!
def ablation_sa1_lora():
    max_epochs_sa = 20
    val_every_sa = 1
    config_folder = "sam1_tenfold"
    model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", config_folder, "config_train.yaml")
    model_config = OmegaConf.load(model_config)
    model_config.config_folder = config_folder
    model_config.fold_val = [0]
    model_config.fold_train = [a for a in range(10) if a != model_config.fold_val[0]]
    model_config.snapshot_path = generate_rndm_path("sa1lora_runs")
    model_config.sam_ckpt = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "checkpoints", "sam_vit_b_01ec64.pth")

    # set batch size to 1 for comparison w/ unet
    details = OmegaConf.load(open(join(os.getenv("PROJECT_PATH", ""), 'MDSA2', 'config', config_folder, 'details.yaml'), 'r'))
    model_config = OmegaConf.merge(model_config, details)
    model_config.dataset = "brats_africa"

    set_deterministic(42)
    train_loader, val_loader, file_paths = get_dataloaders(model_config, verbose=False)
    print("file paths to verify: ", file_paths["train"][0])
    
    save_folder = model_config.snapshot_path
    print("writing to", save_folder)

    sa_model = register_net_sam1(model_config)

    OmegaConf.save(model_config, join(save_folder, "model_config.yaml"))

    writer = SummaryWriter(log_dir=save_folder)
    best_metrics_found = {}
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sa_model.parameters()), lr=model_config.base_lr, **model_config.optimizers.ADAMW) # type: ignore
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=model_config.max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=model_config.use_amp)
    loss_fct = monai.losses.GeneralizedDiceFocalLoss(**model_config.losses.GDF) # type: ignore

    for epoch in range(max_epochs_sa):
        print(f"--- Epoch {epoch+1}/{max_epochs_sa} ---")
        avg_train_loss, avg_batch_time = train_loop_sa1(sa_model,
                                        optimizer,
                                        loss_fct,
                                        scaler,
                                        train_loader,
                                        model_config)
        print(f"avg train loss: {avg_train_loss}, avg batch time: {avg_batch_time}")
        writer.add_scalar("train/avg_loss", avg_train_loss, epoch)
        writer.add_scalar("train/avg_batch_time", avg_batch_time, epoch)

        if (epoch + 1) % val_every_sa == 0:
            print("validating...")
            json_metrics_sa2 = val_loop_sa1(sa_model,
                                val_loader,
                                model_config)
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
                model_path = join(save_folder, 'best_model_sam1lora.pth')

                print(f"saving NEW BEST LoRA model {json_metrics_sa2['dice']['avg']} to {model_path}")
                # save model
                sa_model.save_lora_parameters(model_path)

                with open(join(save_folder, f"best_metrics.json"), "w") as f:
                    json.dump(best_metrics_found, f, indent=4)

    return best_metrics_found

# zero-shot
def ablation_medsam():
    model_config = OmegaConf.create({
        "batch_size_val": 4,
        "fold_val": [0],
        "fold_train": [a for a in range(10) if a != 0],
        "modalities": ["t2f", "t2f", "t2f"],
        "use_preprocessed": True,
        "img_size": 384,
        "use_amp": True,
        "model_type": "medsam",
        "conf_threshold": 0.5,
        "num_slices": 155,
        "ft_ckpt": None,
        "sam_ckpt": "../checkpoints/medsam_vit_b.pth",
        "vit_name": "vit_b",
        "compile": False,
        "rank": -1,
        "custom_name": "",
        "dataset": "brats_africa",
        "num_classes": 3,
        "batch_size_train": 4,
        "batch_size_val": 4,
    })

    set_deterministic(42)
    _, val_loader, file_paths = get_dataloaders(model_config, verbose=False, only_val_transforms=True)

    sa_model = register_net_sam1(model_config = model_config)

    save_folder = generate_rndm_path("medsam_runs")
    OmegaConf.save(model_config, join(save_folder, "model_config.yaml"))

    metrics = val_loop_medsam(sa_model, val_loader, model_config)

    with open(join(save_folder, f"best_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

if __name__ == "__main__":
    # ablation_medsam()
    ablation_sa1_lora()
