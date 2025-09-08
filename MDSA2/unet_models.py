from monai.networks.nets.unet import UNet
from monai.networks.nets.dynunet import DynUNet
from utils import AverageMeter
import torch
import time
import json
import numpy as np
import os

import os
import time
from metrics import AverageMeter, MetricAccumulator, calculate_binary_dice
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from functools import partial
from monai.inferers import sliding_window_inference
from utils import generate_rndm_path
import torch.nn as nn

class UNetWrapper():
    """
    Comprehensive module for training & validating a UNet 
    model with support for "reference volumes" and custom metric accumulation. 
    """
    def __init__(self, train_loader, val_loader,
                  loss_func, scaler, optimizer,
                 config, **model_params):
        
        if config.model_type == "UNet":
            self.model = UNet(**model_params).to("cuda")
            
        elif config.model_type == "DynUNet":
            self.model = DynUNet(**model_params).to("cuda")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.scaler = scaler
        self.optimizer = optimizer
        self.config = config
        self.metric_accumulator = MetricAccumulator(
            additional_metrics={
                "binary_dice": calculate_binary_dice
            }
        )
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)

    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def save_model(self, epoch, save_dir):
        save_path = generate_rndm_path(save_dir)
        # add "u_net" in front of the last part of the path
        base, fname = os.path.split(save_path)
        fname = f"{self.config.model_name}_{fname}"
        save_path = os.path.join(base, fname)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved for epoch {epoch}")

    def train_epoch(self, epoch):
        self.model.train()
        run_loss = AverageMeter('run loss')
        device = torch.device("cuda:0")
        start_time = time.time()

        for idx, batch_data in enumerate(self.train_loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            if self.config.use_ref_volume:
                ref_volume = batch_data["ref_volume"].to(device)
                data = torch.cat([data, ref_volume], dim=1) # concatenate the label and modalities
                del ref_volume

            if idx == 0:
                print(f'{batch_data["image_title"]}, target shape', data.shape, target.shape, data.mean(), data.std())

            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                logits = self.model(data)
                loss = self.loss_func(logits, target) # only for unet++

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                run_loss.update(loss.item(), n=self.config.batch_size)

            del data, target, logits, loss

        print(
        "Epoch {}/{} {}/{}".format(epoch, self.config.max_epochs, idx, len(self.train_loader)),
        "loss: {:.4f}".format(run_loss.avg),
        "time {:.2f}s".format(time.time() - start_time),
        )

        return run_loss.avg
    
    def validate_epoch(self, epoch):
        self.model.eval()
        
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=[self.config.roi[0], self.config.roi[1], self.config.roi[2]],
            sw_batch_size=self.config.sw_batch_size,
            predictor=self.model,
            overlap=self.config.infer_overlap,
        )

        device = torch.device("cuda:0")
        print("starting validation for epoch", epoch)

        inference_durations = []

        with torch.no_grad():
            for idx, batch_data in enumerate(self.val_loader):
                data, target = batch_data["image"].to(device), batch_data["label"].to(device)
                if self.config.use_ref_volume:
                    ref_volume = batch_data["ref_volume"].to(device)
                    data = torch.cat([data, ref_volume], dim=1)

                inf = time.time()
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                    logits = self.model_inferer(data)

                inference_durations.append((time.time()-inf)/self.config.batch_size)
                val_labels_list = decollate_batch(target)
                val_outputs_list = decollate_batch(logits)
                post_output = [self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
                self.metric_accumulator.update(y_pred=post_output, y_true=val_labels_list)

                if self.config.fold_val[0] in self.config.volumes_to_collect.keys():
                    if batch_data["image_title"][0] in self.config.volumes_to_collect[self.config.fold_val[0]]:
                        # print("saving volume", batch_data["image_title"][0], val_output_convert[0].shape)
                        predictions_base = os.path.join(os.getenv("PROJECT_PATH", ""),"extra_software", "visualization", "volumes", "aggregator")
                        os.makedirs(predictions_base, exist_ok=True)
                        np.save(os.path.join(predictions_base, f"{batch_data['image_title'][0]}_fold_{self.config.fold_val[0]}.npy"), post_output[0])

                del data, target, logits, val_labels_list, val_outputs_list, post_output

        # save json
        final_metrics = self.metric_accumulator.get_metrics()
        with open(f"json_metrics_{self.config.model_name}.json", "w") as f:
            json.dump(final_metrics, f, indent=4)

        return final_metrics
    
    @torch.no_grad()
    def run(self, batch):
        if not self.model_inferer:
            self.model_inferer = partial(
                sliding_window_inference,
                roi_size=[self.config.roi[0], self.config.roi[1], self.config.roi[2]],
                sw_batch_size=self.config.sw_batch_size,
                predictor=self.model,
                overlap=self.config.infer_overlap,
            )
        
        device = torch.device("cuda:0")

        data, target = batch["image"].to(device), batch["label"].to(device)
        if self.config.use_ref_volume:
            ref_volume = batch["ref_volume"].to(device)
            data = torch.cat([data, ref_volume], dim=1)

        inf = time.time()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            logits = self.model_inferer(data) # assumes that you've run validate_epoch already? 

        val_labels_list = decollate_batch(target)
        val_outputs_list = decollate_batch(logits)
        post_output = [self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
        self.metric_accumulator.update(y_pred=post_output, y_true=val_labels_list)
        return post_output
    
# TODO: implement example usage
