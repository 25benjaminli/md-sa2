import monai.metrics as metrics_monai
import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU, CumulativeIterationMetric
import os
import monai
from monai.utils.enums import MetricReduction

class MetricAccumulator():
    def __init__(self, exclude_metrics=None, additional_metrics=None, track_time=False):
        self.metric_dict = {
            "dice": DiceMetric(include_background=True, get_not_nans=True, ignore_empty=True, reduction='mean_batch'), # BCHW[D]
            "hd95": HausdorffDistanceMetric(include_background=True, reduction="mean_batch",get_not_nans=True, percentile=95),
            "iou": MeanIoU(include_background=True, get_not_nans=True, reduction="mean_batch")
        }
        if exclude_metrics is not None:
            for metric in exclude_metrics.keys():
                if metric in self.metric_dict:
                    del self.metric_dict[metric]
        
        if additional_metrics is not None:
            # add to self.metric_dict
            for metric in additional_metrics.keys():
                self.metric_dict[metric] = additional_metrics[metric]

        self.meters = {key: AverageMeter(name=key) for key in self.metric_dict.keys()}
        
        if track_time:
            self.inference_meter = AverageMeter(name="inference_time")
    
    def update(self, y_pred, y_true, save_pred_path=None, time_spent=None):
        """
        Assumes y_pred and y_true are the following format: (B, C, H, W)
        """
        if hasattr(self, 'inference_meter') and time_spent is not None:
            self.inference_meter.update(time_spent, n=1) # assumes it's already averaged over batch

        for metric_name in self.metric_dict.keys():
            # print("metric_name", type(self.metric_dict[metric_name]))
            if type(self.metric_dict[metric_name]) != "function":
                self.metric_dict[metric_name](y_pred=y_pred, y=y_true)
                val, not_nans = self.metric_dict[metric_name].aggregate()
                self.meters[metric_name].update(val.cpu().numpy(), n=not_nans.cpu().numpy())
                self.metric_dict[metric_name].reset()
                # print("calculated", val, "for metric", metric_name)
            else:
                # assume that it just yields an output
                val = self.metric_dict[metric_name](y_pred=y_pred, y=y_true)
                self.meters[metric_name].update(val.cpu().numpy(), n=1)

        if save_pred_path:
            np.save(save_pred_path, y_pred.cpu().numpy())

    def get_metrics(self):
        summary_dict = {}
        for key in self.meters.keys():
            # print("avg, stdev", self.meters[key].avg, self.meters[key].stdev)
            pyList = self.meters[key].avg.tolist()

            summary_dict[key] = {
                "classwise_avg": pyList,
                "avg": sum(pyList)/3,
                "stdev": self.meters[key].stdev.tolist(),
            }
        if hasattr(self, 'inference_meter'):
            summary_dict["inference_time"] = {
                "avg": self.inference_meter.avg,
                "stdev": self.inference_meter.stdev.tolist()
            }
        return summary_dict


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()
        self.arr = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.arr = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

        if isinstance(val, np.ndarray):
            self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
        else:
            self.avg = self.sum / self.count

        self.arr.append(val)
        # stdev should be the same shape as val
        self.stdev = np.std(np.array(self.arr), axis=0, ddof=1) if len(self.arr) > 1 else np.zeros_like(val)

# print_config()
def binarize(img):
    # print(len(img), img[0].shape)
    # convert image of a certain length to a torch tensor
    if type(img) == list:
        img = torch.stack(img, dim=0)
    assert len(img.shape) == 4 or (len(img.shape) == 5 and img.shape[0] == 1), f"expected 4D tensor, got {img.shape}"
    # print("img shape", img.shape) #  
    max_indices = torch.argmax(img, dim=1)
    return (max_indices != 0).detach().cpu().numpy().squeeze().astype(np.uint8) # 1 for tumor, 0 for background


def calculate_binary_dice(y_pred, y):
    binarized_pred = torch.from_numpy(binarize(y_pred)).unsqueeze(0)
    binarized_label = torch.from_numpy(binarize(y)).unsqueeze(0)

    # print("binarized prediction shape", binarized_pred.shape, "binarized label shape", binarized_label.shape)

    # assert that the binarized image has only 1 channel and 1s and 0s only via uniques
    # assert len(torch.unique(binarized_pred)) <= 2 and binarized_pred.shape[0] == 1, f"got {torch.unique(binarized_pred)} and shape {binarized_pred.shape}"
    # assert len(torch.unique(binarized_label)) == 2 and binarized_label.shape[0] == 1, f"got {torch.unique(binarized_label)} and shape {binarized_label.shape}"
    # print("binarized prediction shape", binarized_pred.shape, "binarized label shape", binarized_label.shape)

    # calculate binary dice

    # print("binarized pred, binarized label", binarized_pred.shape, binarized_label.shape)
    dsc_fn = DiceMetric(include_background=True, get_not_nans=True, ignore_empty=True)
    dsc_fn(y_pred=binarized_pred, y=binarized_label)
    val_dice_3D_bin, val_dice_3D_not_nans_bin = dsc_fn.aggregate()

    return val_dice_3D_bin