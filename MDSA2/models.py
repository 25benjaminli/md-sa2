from monai.networks.nets.unet import UNet
from monai.networks.nets.dynunet import DynUNet
import torch
import time
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
from utils import generate_rndm_path, AverageMeter, join, register_net_sam2, get_without_name
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# import torch parameter
from torch.nn.parameter import Parameter

from typing import Tuple
from data_utils import get_dataloaders
from monai.losses import DiceLoss
import torch.optim as optim

import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
import monai
from monai.networks.nets import SwinUNETR

class UNetWrapper():
    """
    module for training & validating a UNet 
    support for "reference volumes" and custom metric accumulation. 
    """
    def __init__(self, train_loader, val_loader,
                  loss_func, use_scaler,
                  train_params, config, verbose, **model_params):
        
        if config.model_type == "UNet":
            self.model = UNet(**model_params).to("cuda")
            
        elif config.model_type == "DynUNet":
            self.model = DynUNet(**model_params).to("cuda")

        elif config.model_type == "SwinUNETR":
            self.model = SwinUNETR(**model_params).to("cuda")

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if loss_func == "Dice":
            self.loss_func = DiceLoss(to_onehot_y=False, sigmoid=True)

        if use_scaler:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        if train_params["optimizer"]["name"] == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), **get_without_name(train_params["optimizer"]))
            
        if train_params["scheduler"]["name"] == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **get_without_name(train_params["scheduler"]))
        elif train_params["scheduler"]["name"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **get_without_name(train_params["scheduler"]))
        elif train_params["scheduler"]["name"] == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **get_without_name(train_params["scheduler"]))


        self.config = config
        self.metric_accumulator = MetricAccumulator(
            # additional_metrics={
            #     "binary_dice": calculate_binary_dice
            # }
            track_time=True
        )
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        self.device = torch.device("cuda:0")
        self.verbose = verbose


    def load_weights(self, weights_path):
        thing = torch.load(weights_path, weights_only=False)
        if "state_dict" in thing.keys():
            state_dict = thing["state_dict"]
            print("loaded u-net:", thing["best_acc"])
        else:
            state_dict = thing
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def save_model(self, epoch, save_dir):
        save_path = join(save_dir, "best_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved for epoch {epoch}")

    def train_epoch(self):
        self.model.train()
        run_loss = AverageMeter('run loss')

        for idx, batch_data in enumerate(self.train_loader): # TQDM? 
            data, target = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
            if self.config.use_ref_volume:
                ref_volume = batch_data["ref_volume"].to(self.device)
                data = torch.cat([data, ref_volume], dim=1) # concatenate the label and modalities
                del ref_volume

            if idx == 0 and self.verbose:
                print("current learning rate", self.optimizer.param_groups[0]['lr'])
                print(f'{batch_data["image_title"]}, target shape', data.shape, target.shape, data.mean(), data.std())

            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                logits = self.model(data)
                # loss = self.loss_func(logits, target)
                if len(logits.shape) == 6:
                    # print("using deep supervision")
                    logits = torch.unbind(logits, dim=1)
                    loss, weights = 0.0, 0.0
                    for i, logit in enumerate(logits):
                        # print("logits shape", logits[:, i].shape)
                        loss += self.loss_func(logit, target) * 0.5**i
                        weights += 0.5**i
                    loss /= weights
                else:
                    loss = self.loss_func(logits, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                run_loss.update(loss.item(), n=self.config.batch_size)

            # del data, target, logits, loss

        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        return run_loss.avg
    
    def validate_epoch(self):
        self.model.eval()
        
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=[self.config.roi[0], self.config.roi[1], self.config.roi[2]],
            sw_batch_size=self.config.sw_batch_size,
            predictor=self.model,
            overlap=self.config.infer_overlap,
        )

        with torch.no_grad():
            for idx, batch_data in tqdm(enumerate(self.val_loader)):
                batch_data["image"] = batch_data["image"].to(self.device)
                batch_data["label"] = batch_data["label"].to(self.device)
                if self.config.use_ref_volume:
                    ref_volume = batch_data["ref_volume"].to(self.device)
                    data = torch.cat([data, ref_volume], dim=1) # concatenate the label and modalities
                    del ref_volume
                
                post_output = self.run_batch(batch_data) # run batch already cuda-izes it
                
                if self.config.fold_val[0] in self.config.volumes_to_collect.keys():
                    if batch_data["image_title"][0] in self.config.volumes_to_collect[self.config.fold_val[0]]:
                        # print("saving volume", batch_data["image_title"][0], val_output_convert[0].shape)
                        predictions_base = join(os.getenv("PROJECT_PATH", ""),"extra_software", "visualization", "volumes", "aggregator")
                        os.makedirs(predictions_base, exist_ok=True)
                        np.save(join(predictions_base, f"{batch_data['image_title'][0]}_fold_{self.config.fold_val[0]}.npy"), post_output[0])

                # del batch_data["image"], batch_data["label"], logits, val_labels_list, val_outputs_list, post_output

        return self.metric_accumulator.get_metrics()
    
    @torch.no_grad()
    def run_batch(self, batch): # assumes it's already cuda-ized
        if not hasattr(self, 'model_inferer'):
            self.model_inferer = partial(
                sliding_window_inference,
                roi_size=[self.config.roi[0], self.config.roi[1], self.config.roi[2]],
                sw_batch_size=self.config.sw_batch_size,
                predictor=self.model,
                overlap=self.config.infer_overlap,
            )

        # the assumption is that batch["image"] already contains reference volume if needed
        # if self.config.use_ref_volume:
        #     ref_volume = batch["ref_volume"]
        #     batch["image"] = torch.cat([batch["image"], ref_volume], dim=1)

        inf = time.time()
        if self.verbose:
            print("image shape", batch["image"].shape, "label shape", batch["label"].shape)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            logits = self.model_inferer(batch["image"])

        t_spent = (time.time()-inf)/self.config.batch_size
        val_labels_list = decollate_batch(batch["label"])
        val_outputs_list = decollate_batch(logits)

        
        post_output = [self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
        self.metric_accumulator.update(y_pred=post_output, y_true=val_labels_list, time_spent=t_spent)
        return post_output

class MDSA2(nn.Module):
    """
    End-to-end module combining SA2 and U-Net for medical image segmentation. 
    """
    def __init__(self, sam2_model, unet_model: UNetWrapper, config=None, loss_type="GDF", verbose=False):
        super().__init__()
        self.sam2_model = sam2_model
        self.unet_model = unet_model
        self.config = config
        self.metric_accumulator_sa2 = MetricAccumulator()

        loss_functions = {
            "DCE": monai.losses.DiceCELoss(**config.losses.DCE),
            "GDF": monai.losses.GeneralizedDiceFocalLoss(**config.losses.GDF),
            "DL": monai.losses.DiceLoss(**config.losses.DL)
        }
        self.loss_func = loss_functions[loss_type]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.sam2_model.predictor.model.parameters()),
                                    lr=self.config.base_lr, **config.optimizers.ADAMW)
        self.verbose = verbose


    def forward(self, batch, save_path=None):
        image, label, volume_name = batch["image"].cuda(), batch["label"].cuda(), batch["image_title"]
        arr_pred = torch.zeros(image.shape).cuda()
        arr_low_res = torch.zeros(size=(image.shape[0], image.shape[1], image.shape[2] // 4, image.shape[3] //4, image.shape[4])).cuda()
        
        for current_slice in range(image.shape[4]):
            image_sliced = image[...,current_slice]
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.use_amp):
                    outputs = self.sam2_model(image_sliced, repeat_image=True)
                    
                arr_pred[..., current_slice] = torch.sigmoid(outputs['prd_masks']) > self.config.conf_threshold
                arr_low_res[..., current_slice] = outputs['low_res_logits']

        # Interpolate arr pred
        size_resized = (self.config.seg_size, self.config.seg_size, image.shape[-1])

        arr_pred = nn.functional.interpolate(input=arr_pred, size=size_resized, mode="nearest-exact")
        label = nn.functional.interpolate(input=label, size=size_resized, mode="nearest-exact")
        image = nn.functional.interpolate(input=image, size=size_resized, mode="nearest-exact")

        #! ---- CALCULATE SA2 METRICS ----
        for case in range(len(arr_pred)):
            self.metric_accumulator_sa2.update(y_pred=arr_pred[case].unsqueeze(0), y_true=label[case].unsqueeze(0))

        if not self.unet_model:
            return self.metric_accumulator_sa2.get_metrics(), {}
        
        # repermute image
        image = torch.permute(image, (0,1,4,2,3))
        label = torch.permute(label, (0,1,4,2,3))
        data = torch.cat([torch.permute(arr_pred, (0,1,4,2,3)), image], dim=1) # ensure that the images are aligned properly. maybe need to permute image
        batch = {
            "image": data,
            "label": label
        }
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                val_outputs = self.unet_model.run_batch(batch) # logits shape torch.Size([4, 3, 155, 224, 224])

        # if you desire to visualize vols, you can do soemthing with val_outputs here.
        
        return self.metric_accumulator_sa2.get_metrics(), self.unet_model.metric_accumulator.get_metrics() if self.unet_model else {}
    
    def train_loop_sa(self, train_loader):
        running_train_loss = 0
        train_begin = time.time()
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), disable=False):
            image, label = batch["image"].cuda(), batch["label"].cuda()
            
            # debug
            if self.verbose and idx == 0:
                print("image, label shape", image.shape, label.shape)

            for current_slice in range(image.shape[-1]):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.use_amp):            
                    low_res_labels = F.interpolate(label[...,current_slice], self.config.img_size//4, mode="nearest-exact")
                    outputs = self.sam2_model(image[...,current_slice], upscale=True)
                    
                    # debug
                    if self.verbose and current_slice == 0:
                        print("output shape", outputs['low_res_logits'].shape)

                    assert outputs['low_res_logits'].shape == low_res_labels.shape, f"shapes of logits: {outputs['low_res_logits'].shape} shape of labels {low_res_labels.shape}"
                    loss = self.loss_func(outputs['low_res_logits'], low_res_labels) # what if I move this function OUTSIDE? and calculate on overall 3d shape

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    running_train_loss += loss.detach().item()

                del outputs, loss, low_res_labels
                # low_res_pred = torch.sigmoid(outputs['low_res_logits']) > config.conf_threshold

        return running_train_loss / (len(train_loader)*image.shape[-1]), (time.time() - train_begin) / len(train_loader)

    def val_loop_sa(self, val_loader):
        self.sam2_model.eval()

        for batch in tqdm(val_loader, total=len(val_loader)):
            image, label = batch['image'].cuda(), batch['label'].cuda()
            arr_pred = torch.zeros(size=(label.shape)).cuda()

            t1 = time.time()
            # print("image shape", image.shape)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                for current_slice in range(image.shape[-1]):
                    image_sliced = image[...,current_slice]
                    outputs = self.sam2_model(image_sliced, repeat_image=True) # multimask_output=False,boxes=bbox_sam,repeat_image=False
                    arr_pred[...,current_slice] = torch.sigmoid(outputs['prd_masks']) > self.config.conf_threshold

            size_resized = (self.config.seg_size, self.config.seg_size, image.shape[-1])
            # ! avoid the resize? 
            arr_pred = nn.functional.interpolate(input=arr_pred, size=size_resized, mode="nearest-exact")
            label = nn.functional.interpolate(input=label, size=size_resized, mode="nearest-exact")
            t_spent = (time.time()-t1)/len(arr_pred)
            for case in range(len(arr_pred)):
                self.metric_accumulator_sa2.update(y_pred=arr_pred[case].unsqueeze(0), y_true=label[case].unsqueeze(0), time_spent=t_spent)

        mets = self.metric_accumulator_sa2.get_metrics()
        self.metric_accumulator_sa2.reset()
        
        return mets
    
    def generate_and_save(self, train_loader, val_loader, save_folder):
        self.sam2_model.eval()
        os.makedirs(save_folder, exist_ok=True)

        with torch.no_grad():
            for loader in [train_loader, val_loader]:
                for batch in tqdm(loader, total=len(loader)):
                    image, label, volume_name = batch["image"].cuda(), batch["label"].cuda(), batch["image_title"]
                    arr_pred = torch.zeros(image.shape).cuda()
                    
                    for current_slice in range(image.shape[4]):
                        image_sliced = image[..., current_slice]
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.use_amp):
                            outputs = self.sam2_model(image_sliced, repeat_image=True)
                            
                        arr_pred[..., current_slice] = torch.sigmoid(outputs['prd_masks']) > self.config.conf_threshold

                    # Interpolate arr pred
                    size_resized = (self.config.seg_size, self.config.seg_size, image.shape[-1])
                    arr_pred = nn.functional.interpolate(input=arr_pred, size=size_resized, mode="nearest-exact")

                    for case in range(len(arr_pred)):
                        np.save(join(save_folder, f"BraTS-SSA-{volume_name[case].zfill(5)}-000-pred.npy"), arr_pred[case].detach().cpu().numpy())
                        np.save(join(save_folder, f"BraTS-SSA-{volume_name[case].zfill(5)}-000-label.npy"), label[case].detach().cpu().numpy())
                        np.save(join(save_folder, f"BraTS-SSA-{volume_name[case].zfill(5)}-000-brain_volume.npy"), image[case].detach().cpu().numpy())

# https://github.com/25benjaminli/sam2lora
class LoRA_SAM2(nn.Module):
    def __init__(self, predictor, r: int, lora_layer=None):
        super(LoRA_SAM2, self).__init__()

        assert r > 0
        self.predictor = predictor

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(self.predictor.model.image_encoder.trunk.blocks)))  # Only apply lora to the image encoder by default
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        st = "Total number of parameters sam2 before lora:"
        print(st, sum(p.numel() for p in predictor.model.parameters() if p.requires_grad))

        # freeze the original image encoder
        for param in self.predictor.model.image_encoder.parameters():
            param.requires_grad = False

        
        print(len(self.predictor.model.image_encoder.trunk.blocks))
        for t_layer_i, blk in enumerate(self.predictor.model.image_encoder.trunk.blocks): # TRUNK
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False, device='cuda')
            w_b_linear_q = nn.Linear(r, self.dim, bias=False, device='cuda')
            w_a_linear_v = nn.Linear(self.dim, r, bias=False, device='cuda')
            w_b_linear_v = nn.Linear(r, self.dim, bias=False, device='cuda')
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.predictor.model.image_encoder.trunk.blocks[t_layer_i].attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()

    def save_lora_parameters(self, filename: str) -> None:
        

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        # if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
        #     state_dict = self.sam.module.state_dict()
        # else:
        #     state_dict = self.sam.state_dict()

        state_dict = self.predictor.model.state_dict()
        for key, value in state_dict.items():
            # print("keys", key)
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors} # mask decoder is being updated
        # TODO: save scheduler, optimizer, epoch info as well

        torch.save(merged_dict, filename)

    
    def load_lora_parameters(self, filename: str) -> None:

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, weights_only=False)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.predictor.model.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.predictor.model.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


    def forward(self, image, using_sigmoid=True, return_img_embedding=False, upscale=True):
        self.predictor.set_image_batch(image)
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=None,boxes=None,masks=None)
        high_res_features = [feat_level for feat_level in self.predictor._features["high_res_feats"]]
        # print(self.predictor._features["image_embed"].shape)
        low_res_logits, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"],
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=True,
            high_res_features=high_res_features,
            using_sigmoid=using_sigmoid
        )

        if upscale:
            prd_masks = self.predictor._transforms.postprocess_masks(low_res_logits, self.predictor._orig_hw[-1])

        if return_img_embedding:
            return low_res_logits, prd_masks, prd_scores, self.predictor._features["image_embed"]
        
        if upscale:
            return low_res_logits, prd_masks, prd_scores
        else:
            return low_res_logits, prd_scores
      

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
    
class SAM2_Regular(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.set_image_batch = False

    def get_image_embedding(self, image):
        self.predictor.set_image_batch(image)
        self.set_image_batch = True
        return self.predictor._features["image_embed"]
    
    def forward(self, image, using_sigmoid=True, lr_mask=None, upscale=True, multimask_output=True,boxes=None,repeat_image=True):
        if not self.set_image_batch:
            self.predictor.set_image_batch(image) # normalized regularly? Need to remove this
            
        else:
            self.set_image_batch = False # reset if it was set before


        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=None,boxes=boxes,masks=lr_mask)
        high_res_features = [feat_level for feat_level in self.predictor._features["high_res_feats"]]
        # print(self.predictor._features["image_embed"].shape)
        low_res_logits, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"],
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            using_sigmoid=using_sigmoid
        )
        # print(low_res_logits.shape)

        if upscale:
            prd_masks = self.predictor._transforms.postprocess_masks(low_res_logits, self.predictor._orig_hw[-1])
            return {
                "low_res_logits": low_res_logits, 
                "prd_masks": prd_masks,
                "prd_scores": prd_scores
            }
        return {
            "low_res_logits": low_res_logits, 
            "prd_scores": prd_scores
        }


def initialize_mdsa2(model_config, dataloaders, use_unet=True, verbose=False) -> MDSA2:    
    train_loader, val_loader = dataloaders

    assert model_config.ft_ckpt is None or os.path.exists(model_config.ft_ckpt), f"fine tuned ckpt {model_config.ft_ckpt} does not exist"
    print("model checkpoint", model_config.ft_ckpt)

    sam2 = register_net_sam2(model_config)

    # yeah... this is pretty ugly
    if use_unet:
        volumes_to_collect = yaml.load(open(join(os.getenv("PROJECT_PATH", ""), "volumes_to_collect.yaml"), 'r'), Loader=yaml.FullLoader)
        config = OmegaConf.create({
            "model_type": "DynUNet",
            "use_ref_volume": True,
            "batch_size": 1,
            "max_epochs": 100,
            "roi": [128, 128, 128],
            "sw_batch_size": 1,
            "infer_overlap": 0.5,
            "fold_val": model_config.fold_val,
            "volumes_to_collect": volumes_to_collect,
            "model_name": "aggregator",
        })
        
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        filters = [16, 24, 32, 48, 64, 96, 128]
        
        model_params = {
            "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": strides,
            "filters": filters,
            "upsample_kernel_size": strides[1:],
            "spatial_dims": 3,
            "in_channels": 6,
            "out_channels": 3,
            "norm_name": ("INSTANCE", {"affine": True}),
            "act_name": ("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            "deep_supervision": False,
            # "deep_supr_num": self.args.deep_supr_num,
            "res_block": False,
            "trans_bias": True,
        }

        aggregator_unet = UNetWrapper(train_loader=train_loader, val_loader=val_loader, loss_func="Dice", 
        use_scaler=True, train_params= {
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-5
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "T_max": config.max_epochs,
            }
        }, config=config, verbose=verbose, **model_params)

        # load aggregator unet weights
        if model_config.agg_ckpt is not None and os.path.exists(model_config.agg_ckpt):
            print("loading aggregator unet weights from", model_config.agg_ckpt)
            aggregator_unet.load_weights(model_config.agg_ckpt)

    else:
        aggregator_unet = None
        
    model = MDSA2(sam2, unet_model=aggregator_unet, config=model_config, verbose=verbose)
    model.eval()

    return model