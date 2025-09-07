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
import os
from data_utils import get_dataloaders
from utils import register_net_sam2, set_deterministic
from functools import partial
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from monai.networks.nets import DynUNet
from metrics import dice_coef_not_nans, iou_not_nans, cf_matrix_metric, AverageMeter, MetricAccumulator
import matplotlib.pyplot as plt
import gc
import json
import resource
from collections import OrderedDict
from utils import generate_rndm_path

def print_metric(metric_name, metric_arr):
    print(f"{metric_name} (overall average): ", round(np.mean(np.mean(metric_arr, axis=1)),4), "+-", round(np.std(np.mean(metric_arr, axis=1), ddof=1),4))
    print(f"{metric_name} (classwise mean):", np.mean(metric_arr, axis=0))
    print(f"{metric_name} (average values): ", np.mean(metric_arr, axis=1))
    print("---------")

class MDSA2(nn.Module):
    def __init__(self, sam2_model, unet_inferer, config):
        super().__init__()
        self.sam2_model = sam2_model
        self.unet_inferer = unet_inferer
        self.config = config
        # self.metric_dict = metric_dict
        self.metric_accumulator_mdsa2 = MetricAccumulator()
        self.metric_accumulator_sa2 = MetricAccumulator()


    def forward(self, batch):
        image, label, volume_name = batch["image"], batch["label"], batch["image_title"]

        arr_pred = torch.zeros(image.shape).cuda()
        arr_low_res = torch.zeros(size=(image.shape[0], image.shape[1], image.shape[2] // 4, image.shape[3] //4, image.shape[4])).cuda()
        
        for current_slice in range(image.shape[4]):
            image_sliced = image[...,current_slice]
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.use_amp):
                    outputs = self.sam2_model(image_sliced, repeat_image=True)
                    
                arr_pred[..., current_slice] = torch.sigmoid(outputs['prd_masks']) > self.config.conf_threshold
                arr_low_res[..., current_slice] = outputs['low_res_logits']

        # interpolate arr pred
        size_resized = (self.config.seg_size, self.config.seg_size, image.shape[-1])

        arr_pred = nn.functional.interpolate(input=arr_pred, size=size_resized, mode="nearest-exact")
        label = nn.functional.interpolate(input=label, size=size_resized, mode="nearest-exact")
        image = nn.functional.interpolate(input=image, size=size_resized, mode="nearest-exact")

        #! ---- CALCULATE SA2 METRICS ----
        for case in range(len(arr_pred)):
            self.metric_accumulator_sa2.update(y_pred=arr_pred[case].unsqueeze(0), y_true=label[case].unsqueeze(0))

        if not self.unet_inferer:
            return self.metric_accumulator_sa2.get_metrics(), {}
        
        # repermute image
        image = torch.permute(image, (0,1,4,2,3))
        label = torch.permute(label, (0,1,4,2,3))
        data = torch.cat([torch.permute(arr_pred, (0,1,4,2,3)), image], dim=1) # ensure that the images are aligned properly. maybe need to permute image
        
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                logits = self.unet_inferer(data) # logits shape torch.Size([4, 3, 155, 224, 224])

        post_sigmoid = Activations(sigmoid=True)
        post_pred = AsDiscrete(argmax=False, threshold=0.5)

        val_outputs_list = decollate_batch(logits) # logits, might use for smth else
        val_outputs = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]

        for case in range(len(arr_pred)):
            # save to an np array
            self.metric_accumulator.update(y_pred=val_outputs[case].unsqueeze(0), y_true=label[case].unsqueeze(0))
            save_path = os.path.join(self.config.snapshot_path, "predictions")
            os.makedirs(save_path, exist_ok=True)
            fname = f"{volume_name[case]}_fold_{self.config.fold_val[0]}_{avg_dice}.npy"
            arr_np = val_outputs[case].unsqueeze(0).detach().cpu().numpy()
            
            np.save(os.path.join(save_path, fname), arr_np)
            # print("saved to", os.path.join(save_path, fname))

        
        return self.metric_accumulator_sa2.get_metrics(), self.metric_accumulator_mdsa2.get_metrics()
        
def eval_loop(args, model_config):
    num_all_folds = 10
    model_config.fold_train = [i for i in range(num_all_folds) if i != args.fold_eval]
    model_config.fold_val = [args.fold_eval]

    path_thing = os.path.join(f"{args.config_folder}_cv", f"cv_fold_{args.fold_eval}")

    model_config.ft_ckpt = os.path.join(os.getenv("PROJECT_PATH"), "MDSA2", "train", "runs", "brats_africa", path_thing, "best_model_sam2.pth")
    model_config.batch_size_train=1 # manually set to 1 for comparison
    model_config.batch_size_val=1 # manually set to 1 for comparison
    model_config.snapshot_path = generate_rndm_path(os.path.join("eval", "runs", "aggregator"))
    
    assert os.path.exists(model_config.ft_ckpt), f"lora ckpt {model_config.ft_ckpt} does not exist"
    print("model checkpoint", model_config.ft_ckpt)

    sam2_model = register_net_sam2(model_config = model_config)
    
    ############### load aggregator model

    checkpoint_to_load = os.path.join(os.getenv("PROJECT_PATH"), "MDSA2", "train", "runs", "aggregator", path_thing, "best_model.pth")
    loaded = torch.load(checkpoint_to_load)
    print("loading model from", checkpoint_to_load)
    print("epoch of loaded", loaded['epoch'], "best acc", loaded['best_acc'])

    ############### get dataloaders
    train_loader, val_loader, file_paths = get_dataloaders(model_config, use_preprocessed=True, modality_to_repeat=-1)
    

    ############### define unet_inferer
    
    kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    filters = [16, 24, 32, 48, 64, 96, 128]

    aggregator_model = DynUNet(
        spatial_dims=3,
        in_channels=6,
        out_channels=3,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=filters,
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
        deep_supervision=False,
        # deep_supr_num=self.args.deep_supr_num,
        res_block=False,
        trans_bias=True,
    ).to("cuda")
    
    aggregator_model.load_state_dict(loaded['state_dict'])
    
    unet_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi[0], args.roi[1], args.roi[2]],
        sw_batch_size=args.sw_batch_size,
        predictor=aggregator_model,
        overlap=args.infer_overlap,
    )

    ############### define metrics
    
    metric_dict = OrderedDict({
        "dice": DiceMetric(include_background=True, get_not_nans=True, ignore_empty=True, reduction='mean_batch'), # BCHW[D]
        "hd95": HausdorffDistanceMetric(include_background=True, reduction="mean_batch",get_not_nans=True, percentile=95),
        "iou": MeanIoU(include_background=True, get_not_nans=True, reduction="mean_batch")
    })

    ############### define and run MDSA2
    mdsa2 = MDSA2(sam2_model=sam2_model, unet_inferer=unet_inferer, config=model_config)

    md_averagemeters = {key: AverageMeter(name="hi") for key in metric_dict.keys()}
    sa_averagemeters = {key: AverageMeter(name="hi") for key in metric_dict.keys()}
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

    model_config = os.path.join(os.getenv("PROJECT_PATH"), "MDSA2", "config", args.config_folder, "config_train.yaml")
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