# PEP 8 import format

# standard library imports
import json
import os
import random
import shutil
import time
import posixpath
import math
import pathlib

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Sampler, BatchSampler

# data analysis imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

import monai
from monai.metrics.meandice import DiceMetric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset

import monai.transforms as transforms
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, CastToTyped, ToTensord
from monai.transforms.spatial.dictionary import (
    RandFlipd, RandZoomd, Resized
)
from monai.transforms.intensity.dictionary import (
    NormalizeIntensityd, RandShiftIntensityd, RandScaleIntensityd, ScaleIntensityRangePercentilesd
)
from monai.transforms.croppad.dictionary import (
    CenterSpatialCropd, CropForegroundd, RandSpatialCropd
)
from utils import get_volume_number, join

load_dotenv(override=True)

class PrintCallback(transforms.transform.MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            print("key", key, "shape", d[key].shape, "dtype", d[key].dtype)

        return d

def replace_ending(path, endings):
    for ending in endings:
        if ending in path:
            return path.replace(ending, "npy")
    return path

def generate_pseudo_files(di, fold_val):
    # given di and fold val generate a new folder in data and set the environmen
    # loop thru all files
    # base_data_dir = os.path.basename(os.path.normpath(os.getenv('DATASET_PATH')))
    
    base_data_dir = os.getenv("DATA_PATH")
    # new_path = join(base_data_dir, 'pseudo_val_volumes')

    # print("new path", new_path)
    print("GENERATING PSEUDO FILES")
    import nibabel as nib
    endings = ["nii.gz", "nii"] # nii.gz comes first so it's replaced first
    for idx_patient, patient in tqdm(enumerate(di['training'])):
        
        
        seg_path = replace_ending(patient['label'], endings)

        if patient['fold'] in fold_val and not os.path.exists(seg_path):
            segmentation = nib.load(patient['label']).get_fdata()

            tf_nonzero = np.any(segmentation, axis=(0,1))
            # convert true false nonzeros for each index to a list of indices
            indices_non_zero = np.where(tf_nonzero)[0]

            seg_path = replace_ending(patient['label'], endings)
            seg = segmentation[...,indices_non_zero]
            np.save(join(seg_path), seg) # save as nii?
            di['training'][idx_patient]['label'] = seg_path
            assert os.path.exists(seg_path)

            for img_idx, image_path in enumerate(patient['image']):
                img = nib.load(image_path).get_fdata()
                # crop the image
                img_path = replace_ending(image_path, endings)
                img_new = img[...,indices_non_zero]

                np.save(img_path, img_new)
                di['training'][idx_patient]['image'][img_idx] = img_path

                assert os.path.exists(img_path) # just have it be in the same location as the old dataset

                # assert the shapes are the same
                assert img_new.shape == seg.shape, f"img shape {img_new.shape} seg shape {seg.shape}"

        elif os.path.exists(seg_path):
          # send training
          di['training'][idx_patient]['label'] = seg_path
          assert os.path.exists(seg_path)

          for img_idx, image_path in enumerate(patient['image']):
            img_path = replace_ending(image_path, endings)
            
            di['training'][idx_patient]['image'][img_idx] = img_path

            assert os.path.exists(img_path) # just have it be in the same location as the old dataset

def generate_json(dataset_path, fold_num, config, seg_key='label', modalities=['flair', 't1ce', 't1', 't2'], ending="nii.gz"):
    # TODO: remove because we already have one?

    if "name_mapping.csv" in os.listdir(dataset_path):
        # this would only be used if training on a regular BraTS dataset with LGG/HGG data available
        di, sorted_fnames = generate_json_stratify(dataset_path, fold_num, seg_key, modalities)
    else:
        di, sorted_fnames = generate_json_no_stratify(dataset_path, fold_num, seg_key, modalities, ending=ending)

    return di, sorted_fnames


def generate_json_stratify(dataset_path, fold_num, seg_key, modalities):
    df = pd.read_csv(posixpath.join(dataset_path, 'name_mapping.csv'))

    # 369 total files, 41 each fold for 9 total folds
    # for each fold, proportional amounts of HGG and LGG should be in each fold
    # 293/9 = 32.55555555555556
    # 76/9 = 8.444444444444445

    shuff_hgg = df[df['Grade'] == 'HGG'].sample(frac=1, random_state=0).reset_index(drop=True)
    shuff_lgg = df[df['Grade'] == 'LGG'].sample(frac=1, random_state=0).reset_index(drop=True) # deterministic random generation

    num_hgg_each_fold = len(shuff_hgg) // fold_num
    num_lgg_each_fold = len(shuff_lgg) // fold_num

    print('number of hgg per fold: ', num_hgg_each_fold)
    print('number of lgg per fold: ', num_lgg_each_fold)

    folds = []
    for i in range(fold_num):
        fold = []
        for j in range(num_hgg_each_fold):
            fold.append(shuff_hgg.loc[i*num_hgg_each_fold + j].at['BraTS_2020_subject_ID'])
        for j in range(num_lgg_each_fold):
            fold.append(shuff_lgg.loc[i*num_lgg_each_fold + j].at['BraTS_2020_subject_ID'])
        folds.append(fold) # BraTS20_Training_087
    # do for the traininng first
    di = {}
    di['training'] = []


    running_di_idx = 0 # keeps track of the index within the dictionary
    all_fold_data = {}
    for fold_idx in range(len(folds)): # 9
        # fold is a list of 40 strings e.g. [Brats20_Training_001, Brats20_Training_002, ...]
        all_fold_data[fold_idx] = []
        for idx_data, fold_data in enumerate(folds[fold_idx]): # 40
            # print("fold", fold_data)
            di['training'].append({})
            di['training'][running_di_idx]['fold'] = fold_idx
            di['training'][running_di_idx]['image'] = []
            di['training'][running_di_idx]['label'] = ''

            # get all images
            for j in range(len(modalities)):
                di['training'][running_di_idx]['image'].append(posixpath.join(dataset_path, fold_data, fold_data + '_' + modalities[j] + '.nii'))

            # get label
            di['training'][running_di_idx]['label'] = posixpath.join(dataset_path, fold_data, fold_data + f'_{seg_key}.nii')

            # confirm that these files exist
            for j in range(len(modalities)):
                # print(di['training'][running_di_idx]['image'][j])
                assert posixpath.exists(di['training'][running_di_idx]['image'][j])

            assert posixpath.exists(di['training'][running_di_idx]['label'])

            running_di_idx += 1

            # all_fold_data.append(fold_data)
            all_fold_data[fold_idx].append(fold_data)

    # sort each key in fold_data_orig
    for key in all_fold_data:
        all_fold_data[key] = sorted(all_fold_data[key])

    return di, sorted(all_fold_data)

    
# mostly for brats africa
def generate_json_no_stratify(dataset_path, fold_num, seg_key='seg', modalities=['t2f', 't1c', 't1n', 't2w'], ending='nii.gz'):

    di = {}

    di['training'] = []
    # generate indices corresponding to folds
    indices = list(range(0, len(os.listdir(dataset_path))))

    fold_len = len(indices) // fold_num
    current_fold = 0

    # randomize the items in the dataset path
    random_shuffled_dataset = sorted(os.listdir(dataset_path))
    random.shuffle(random_shuffled_dataset)
    print("first few items", random_shuffled_dataset[:5])

    fold_data_orig = {0: []}
    for idx_patient, patient in enumerate(random_shuffled_dataset):
        # send
        if idx_patient % fold_len == 0 and idx_patient != 0:
            current_fold += 1
            fold_data_orig[current_fold] = []
            # assert current_fold < fold_num
        di['training'].append({})
        
        di['training'][idx_patient]['image'] = []
        di['training'][idx_patient]['fold'] = current_fold
        fold_data_orig[current_fold].append(patient)

        for modality in modalities:
            # print(posixpath.join(dataset_path, patient, f"{patient}-{modality}.{ending}"))
            assert posixpath.exists(posixpath.join(dataset_path, patient, f"{patient}-{modality}.{ending}"))
            di['training'][idx_patient]['image'].append(posixpath.join(dataset_path, patient, f"{patient}-{modality}.{ending}"))

        di['training'][idx_patient]['label'] = posixpath.join(dataset_path, patient, f"{patient}-{seg_key}.{ending}")
        assert posixpath.exists(posixpath.join(dataset_path, patient, f"{patient}-{seg_key}.{ending}"))
        
    for key in fold_data_orig:
        fold_data_orig[key] = sorted(fold_data_orig[key])
    
    return di, fold_data_orig
    

def datafold_read(dataset_path, fold_train,fold_val, key="training", cap=60, modalities = ['t2f', 't1c', 't1n', 't2w'], json_path='train.json'):
    with open(json_path) as f:
        json_data = json.load(f)

    json_data = json_data[key]
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                # only select the file paths that contain one of the modalities in the name
                d[k] = [f for f in d[k] if any([m in f for m in modalities])]

    tr = []
    val = []

    for d in json_data:
        # print(d, fold_val)
        if "fold" in d and d["fold"] in fold_val:
            # remove fold key
            d.pop("fold")
            val.append(d)

        elif fold_train == "-1":
            d.pop("fold")
            tr.append(d) # everything else goes to training
        elif fold_train != "-1" and "fold" in d and d["fold"] in fold_train:
            d.pop("fold")
            tr.append(d)

    # print("sanity", val[:2])

    if cap is not None:
        return tr, val
    else:
        return tr, val


class ConvertToMultiChannel(transforms.transform.MapTransform):
    def __init__(self, keys, allow_missing_keys=False, use_softmax=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.use_softmax = use_softmax

    def __call__(self, data):
        # call it on label
        
        # label is 240x240x155, must be 3x240x240x155

        d = dict(data)
        d["label"] = d["label"].squeeze()
        if not self.use_softmax:
            result = [(d["label"] == 1), (d["label"] == 2), (d["label"] == 3)]

        else:
            result = [(d["label"] == 0), (d["label"] == 1), (d["label"] == 2), (d["label"] == 3)]
            
        d["label"] = torch.stack(result, dim=0)

        return d
    
class RepeatModality(transforms.transform.MapTransform):
    def __init__(self, keys, modality_to_repeat=0, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.modality_to_repeat = modality_to_repeat # 0 is t2f

    def __call__(self, data):
        d = dict(data)

        orig_img = d["image"][self.modality_to_repeat] # 384x384x155
        new_img = torch.stack([orig_img] * 3, dim=0)
        d["image"] = new_img

        del new_img
        return d

class AddNameField(transforms.transform.MapTransform):
    def __init__(self, keys, send_real_path=False, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.send_real_path = send_real_path

    def __call__(self, data):
        d = dict(data)

        if not "image_title" in d:
            # print("------- d image -----------", d["image"])
            # arbitrary select the first index
            # name = os.path.splitext(d["image"][0])[0].split("_")[-2] if "_" in d["image"][0] else os.path.splitext(d["image"][0])[0].replace('-seg', '')
            if not self.send_real_path:
                name = os.path.basename(os.path.dirname(d["image"][0]))
                # if it's brats normal, then just take the last part
                if "BraTS20" in name:
                    name = name.split("_")[-1] # volume num
                else:
                    name = str(int(name.split("-")[-2])).zfill(3) # refill with z because it has 5 digits for some reason originally
                d["image_title"] = name
            else:
                d["image_title"] = d["image"]
                d["label_title"] = d["label"]

        return d

class AddNameFieldAggregator(transforms.transform.MapTransform):
    def __init__(self, keys, send_real_path=False, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.send_real_path = send_real_path

    def __call__(self, data):
        d = dict(data)

        if not "image_title" in d:
            # print("------- d image -----------", d["image"])
            # arbitrary select the first index
            # name = os.path.splitext(d["image"][0])[0].split("_")[-2] if "_" in d["image"][0] else os.path.splitext(d["image"][0])[0].replace('-seg', '')
            if not self.send_real_path:
                p = d["image"]
                basename = os.path.basename(p)
                name, ext = os.path.splitext(basename)
                # print("NAME", name)
                # if it's brats normal, then just take the last part
                d["image_title"] = name.split("_")[1]
            else:
                d["image_title"] = d["image"]
                d["label_title"] = d["label"]

        return d
class RandomizeSlices(transforms.transform.MapTransform):
    def __init__(self, keys, oversample=0, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.oversample = oversample

    def __call__(self, data):
        # call it on label
        
        # label is 240x240x155, must be 3x240x240x155

        d = dict(data)

        current_indices = list(range(data["image"].shape[-1]))

        # pick random indices, generate array of integers of from 0, data["image"].shape[-1]-1 to represent indices without replacement

        indices = np.random.choice(current_indices, len(current_indices), replace=False)
        # get indices of slices that are not zero
        indices_non_zero = torch.nonzero(torch.sum(data["label"], dim=(0, 1, 2))).squeeze(1).tolist() # squeeze tolist?


        randomized_image = torch.zeros_like(data["image"])
        randomized_mask = torch.zeros_like(data["label"])
        for slice_idx in range(data["image"].shape[-1]):
            # if the slice is zero, then we can oversample it
            if self.oversample > 0 and len(torch.nonzero(data["image"][..., slice_idx])) == 0 and np.random.rand() < self.oversample:
                rand_val = np.random.randint(0, len(indices_non_zero))
                randomized_image[..., slice_idx] = data["image"][..., indices_non_zero[rand_val]]
                randomized_mask[..., slice_idx] = data["label"][..., indices_non_zero[rand_val]]
            else:
                randomized_image[..., slice_idx] = data["image"][..., indices[slice_idx]] # copy slice by slice - is it too slow
                randomized_mask[..., slice_idx] = data["label"][..., indices[slice_idx]]
        
        d["image"] = randomized_image
        d["label"] = randomized_mask

        return d

# credit for batching method: https://medium.com/@haleema.ramzan/how-to-build-a-custom-batch-sampler-in-pytorch-ce04161583ee
class CurriculumSampler(BatchSampler):
    """
    reverse: If you want descending order

    Note: randomized slices cannot be used with this
    """
    def __init__(self, dataset, reverse=True, batch_size=2, epochs=50, grow_epochs=5, start_rate=0.1):
        self.dataset = dataset
        # self.num_slices = num_slices
        # self.difficulty = difficulty
        self.reverse = reverse
        self.batch_size = batch_size

        self.curriculum = {
            "ema_metric": 0,
            "max_patience": 3,
            "div": 5, # divide into five sections
            "step_init": 0,
            # "alpha": 0.2, # influence of the current step, kind of arbitrary right now. similar to nnunet
            "alpha": 0.5, # influence of the current step, kind of arbitrary right now. similar to nnunet

            "current_patience": 0,
            "thresh_improve": 0.001,
            "weights": {
                "metrics": 0, # greater metric is easier
                "num_pixels": 0, # the more pixels, the easier 
                "distance_vars": 0, # less distance variance is easier
                "losses": 1, # if true, then the more distance variance, the easier
            
            }, # set one weight to zero to just not use it at all.
            "minus": {
                "metrics": True, # if true, then the greater the metric, the easier
                "num_pixels": True, # if true, then the more pixels, the easier
                "distance_vars": False, # if true, then the more distance variance, the easier
                "losses": False, # if true, then the more distance variance, the easier

            },
            "average_metrics": [],
            "ema_metrics": [],
        }

        # self.weights = [0.33, 0.33, 0.33]
        # self.div = 10
        # self.ema_metric = 0 # ! REFINE to go off a moving average? (today*alpha) + previous*(1-alpha)
        # self.patience = 2 # 2 epochs to wait
        # self.ema_details
        self.current_epoch = 0
        self.grow_epochs = grow_epochs
        self.start_rate = start_rate

    def run_heuristic(self):
        from utils import calculate_heuristic

        heuristic_summed, heuristic_arr = calculate_heuristic(self.dataset.info, self.curriculum['weights'], self.curriculum['minus'], self.dataset.vol_list)
        sorted_indices = sorted(heuristic_summed, key=lambda x: x[1], reverse=False)

        growth = self.grow_subset()
        data_rate = min(1.0, growth)    
        end_indices = int(len(self.dataset)*data_rate)
        
        self.sorted_indices = sorted_indices[:end_indices]
        self.current_epoch += 1

        print("current epoch", self.current_epoch)

    def grow_subset(self):
        return round(self.start_rate + ((1.0-self.start_rate)/self.grow_epochs * self.current_epoch), 5)


    def __iter__(self):
        if self.dataset.use_heuristic:
            self.run_heuristic()
        else:
            # just do a typical training epoch for the first one, the sorted indices aren't really sorted
            self.sorted_indices = [(idx, 0) for idx in range(len(self.dataset))]

        batch = []
        for idx_batch, (idx_volume, _) in enumerate(self.sorted_indices):
            batch.append(idx_volume)
            if (idx_batch == len(self.sorted_indices) - 1 and len(batch) < self.batch_size) or len(batch) == self.batch_size:
                yield batch
                batch = []              

        
        # plot metrics vs distance_vars and num_pixels at the end of each epoch
        plt.scatter(self.dataset.info['distance_vars'], self.dataset.info['metrics'])
        plt.xlabel("Distance Variance")
        plt.ylabel("Metrics")
        plt.title("Metrics vs. Distance Variance")
        plt.savefig("metrics_vs_distance.png")
        plt.clf()

        plt.scatter(self.dataset.info['num_pixels'], self.dataset.info['metrics'])
        plt.xlabel("Number of Pixels")
        plt.ylabel("Metrics")
        plt.title("Metrics vs. Number of Pixels")
        plt.savefig("metrics_vs_pixels.png")
        plt.clf()

        # plot for average metric vs epoch
        plt.plot(range(len(self.curriculum['average_metrics'])), self.curriculum['average_metrics'], color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Average Metric/ema metric")
        # also plot the ema metrics but put a line through it
        plt.plot(range(len(self.curriculum['ema_metrics'])), self.curriculum['ema_metrics'], color='red')
        plt.title("Average Metric vs. Epoch")
        plt.savefig("average_metric_vs_epoch.png")

        
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size) if not self.dataset.use_heuristic else math.ceil(len(self.sorted_indices) / self.batch_size) # ! TODO: fix this not yielding correct size
        # number of batches is the number of samples divided by the batch size but rounded up

class OverSampler(Sampler):
    """
    reverse: If you want descending order
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        # Sort indices by loss in descending order
        # sort difficulty by number of pixels

        # sorted_indices = sorted(difficulty, key=lambda x: x[1], reverse=self.reverse)
        # return iter(sorted_indices)

        # oversample
        pct_masks = 0.5
        tumor_indices = [idx for idx in range(len(self.dataset)) if torch.sum(self.dataset[idx]['label'] > 0).item() > 0]
        non_tumor_indices = [idx for idx in range(len(self.dataset)) if torch.sum(self.dataset[idx]['label'] > 0).item() == 0]

        # sample with replaceement
        selected_tumor_indices = np.random.choice(tumor_indices, int(pct_masks * len(tumor_indices)), replace=True)
        selected_non_tumor_indices = np.random.choice(non_tumor_indices, int((1-pct_masks) * len(non_tumor_indices)), replace=True)
        idx_finals = np.hstack([selected_tumor_indices, selected_non_tumor_indices])
        
        idx_finals = idx_finals[:len(self.dataset)] # cap it at the length of the dataset to avoid any rounding errors
        return iter(idx_finals)

    def __len__(self):
        return len(self.dataset)
    
class LossSampler(Sampler):
    """
    reverse: If you want descending order
    """
    def __init__(self, dataset, reverse=True, batch_size=4):
        self.dataset = dataset
        self.reverse = reverse
        self.batch_size = batch_size

    def __iter__(self):
        sorted_indices = sorted(range(len(self)), key=lambda idx: self.dataset.losses[idx], reverse=self.reverse)
        # print("sorted indices", sorted_indices)
        # print("orig indices", self.dataset.losses)

        batch = []
        for idx in sorted_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size) # number of batches is the number of samples divided by the batch size but rounded up
    

class BrainDataset(Dataset):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

        self.info = {
            "metrics": np.zeros(shape=(300,)), # arbitrary size
            "distance_vars": np.zeros(shape=(300,)),
            "num_pixels": np.zeros(shape=(300,)),
            "losses": np.zeros(shape=(300,)),

        }
        self.use_heuristic = False
        self.vol_list = []

    # def __len__(self):
    #     return math.ceil(len(self.data) / self.batch_size)

    def calculate_variance(self, label):
        # label is 3x224x224xslices
        # calculate the variance of the label
        aggregated_variance = 0
        assert len(label.shape) == 4, "label must be 3ximgximgxslices"
        for i in range(label.shape[0]):
            # label[i] is 224x224xslices, so torch nonzero yields 2D dims
            indices = torch.nonzero(label[i] > 0).to(torch.float16)

            for axis in range(indices.shape[1]):
                var = torch.var(indices[:, axis], unbiased=False)
                aggregated_variance += var if not torch.isnan(var) else 0 # if there are no positive pixels, then the variance is zero
        
        return aggregated_variance.cpu().numpy() # maybe also calculate it on the binarized version
    
    def calculate_numpixels(self, label_binary):
        # label must already be thresholded, 3ximgximgxnumslices

        label_binary = torch.sum(label_binary, dim=0) > 0 # 224x224, overlappnig pixels considered as one
        # calculate the number of positive pixels
        return torch.sum(label_binary).item()


    def update_info(self, pred, gt, metric, losses, vol_name):
        # each label corresponds to a ground truth array. assumes label is sigmoided and thresholded already

        for batch_idx in range(gt.shape[0]):
            # 3ximgximgxslices
            label_var = self.calculate_variance(label=gt[batch_idx])
            num_pixels = self.calculate_numpixels(label_binary=gt[batch_idx])
            
            self.info['metrics'][vol_name[batch_idx]] = sum(metric[batch_idx])/3 # np array
            self.info['distance_vars'][vol_name[batch_idx]] = label_var
            self.info['num_pixels'][vol_name[batch_idx]] = num_pixels
            self.info['losses'][vol_name[batch_idx]] = losses[batch_idx]

    def save_vol_list(self, vol_list):
        self.vol_list = vol_list

class RandomSampler(BatchSampler):
    def __init__(self, dataset, batch_size=2, limit=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.limit = limit

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.limit < 1:
            random.shuffle(indices)
            indices = indices[:int(self.limit * len(indices))] # limit the number of samples
        print("length indices", len(indices))
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch
    
    def __len__(self):
        return int(len(self.dataset) * self.limit) - 1 # subtract one because it's zero indexed 



def preprocess(config, use_normalize=True):
    preprocess_transforms = transforms.compose.Compose(
        [
            AddNameField(keys=["image"],send_real_path=True),
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CastToTyped(keys=["image", "label"], dtype=(torch.float16, torch.uint8)),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[224, 224, -1]), #  -1 means keep the same size
            Resized(keys=["image", "label"], spatial_size=tuple(config.preprocessing.resize_dims) + (-1,), mode=("bilinear", "nearest-exact")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True) if use_normalize else ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True),
        ]
    )

    json_path = join(os.getenv("PROJECT_PATH"), "MDSA2", "train.json")
    
    train_files, validation_files = datafold_read(dataset_path=os.getenv("DATASET_PATH"), fold_val=config.fold_val, fold_train=config.fold_train,
                                                              modalities=config.modalities, json_path=json_path)


    print("length of train, val files", len(train_files), len(validation_files))
    # merge train files and validation files such that both of them get preprocessed
    all_files = train_files + validation_files

    # use a dataloader on these

    preprocess_dataset = monai.data.Dataset(data=all_files, transform=preprocess_transforms)
    preprocess_loader = DataLoader(preprocess_dataset, num_workers=config.num_workers, batch_size=config.batch_size_train, shuffle=False)

    # preprocess everything
    preprocess_folder = os.getenv("PREPROCESSED_PATH")
    # if exists, remove it. create it
    if os.path.exists(preprocess_folder):
        shutil.rmtree(preprocess_folder)

    os.makedirs(preprocess_folder, exist_ok=False)
    curr_idx = 0
    for data in tqdm(preprocess_loader):
        # send to appropriate place based on the name
        loc = data["image_title"] # (3, 4) where 3 is modalities, 4 is batch size. I need to transpose it
        loc = np.array(loc).transpose(1, 0) # (4, 3)
        if curr_idx == len(preprocess_loader)-1:
            fig, axs = plt.subplots(4, 3)
            print("image shape", data["image"].shape)

        for batch_idx in range(len(loc)):
            # create a folder for each batch idx
            current_loc = loc[batch_idx] # (3,)
            # for each image, save it to the folder
            for modality_idx, modality_path in enumerate(current_loc):

                new_path = modality_path.replace(os.getenv("DATASET_PATH"), preprocess_folder)
                # create the folder if it doesn't exist
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                new_path = new_path.replace(".nii.gz", ".npy")

                # save the image
                np.save(new_path, data["image"][batch_idx][modality_idx])

            # save the label under label title
            new_label_path = data["label_title"][batch_idx].replace(os.getenv("DATASET_PATH"), preprocess_folder)
            os.makedirs(os.path.dirname(new_label_path), exist_ok=True)

            new_label_path = new_label_path.replace(".nii.gz", ".npy")
            np.save(new_label_path, data["label"][batch_idx])

        curr_idx += 1

def get_dataloaders(config, use_preprocessed=False, modality_to_repeat=-1):
    train_transforms = transforms.compose.Compose(
        [
            AddNameField(keys=["image"]),
            LoadImaged(keys=["image", "label"]), # assuming loading multiple at the same time.
            # RepeatModality(keys=["image", "label"], modality_to_repeat=modality_to_repeat) if modality_to_repeat != -1 else None, # repeat t2f 3 times
            ConvertToMultiChannel(keys="label", use_softmax=False),
            CastToTyped(keys=["image", "label"], dtype=(torch.float16, torch.uint8)),
            # # now, randomized stuff
            RandFlipd(keys=["image", "label"], prob=config.augmentation.flipH.p, spatial_axis=0) if config.augmentation.flipH is not None else None,
            RandFlipd(keys=["image", "label"], prob=config.augmentation.flipW.p, spatial_axis=1) if config.augmentation.flipW is not None else None,
            RandFlipd(keys=["image", "label"], prob=config.augmentation.flipW.p, spatial_axis=2) if config.augmentation.flipD is not None else None, # !!
            RandZoomd(keys=["image", "label"], prob=config.augmentation.zoom.p, min_zoom=config.augmentation.zoom.min, max_zoom=config.augmentation.zoom.max, mode="nearest-exact") if config.augmentation.zoom is not None else None,
            RandShiftIntensityd(keys=["image"], offsets=0.5, prob=0.3, channel_wise=True) if config.augmentation.FG is not None else None,
            RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.3, channel_wise=True) if config.augmentation.FG is not None else None,
            ToTensord(keys=["image", "label"], track_meta=False)
        ]
    )

    val_transforms = transforms.compose.Compose(
        [
            AddNameField(keys=["image"]),
            LoadImaged(keys=["image", "label"]), # assuming loading multiple at the same time.
            # RepeatModality(keys=["image", "label"], modality_to_repeat=modality_to_repeat) if modality_to_repeat != -1 else None, # repeat t2f 3 times
            ConvertToMultiChannel(keys="label", use_softmax=False),
            CastToTyped(keys=["image", "label"], dtype=(torch.float16, torch.uint8)),
            ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )
    # ! TODO: change learning rate
    
    json_path = join(os.getenv("PROJECT_PATH"), "MDSA2", "train.json")
    dataset_path = os.getenv("DATASET_PATH") if not use_preprocessed else os.getenv("PREPROCESSED_PATH")
    print("modalities", config.modalities)
    train_files, validation_files = datafold_read(dataset_path=dataset_path, fold_val=config.fold_val, fold_train=config.fold_train,
                                                              modalities=config.modalities, json_path=json_path)


    print("length of train, validation files", len(train_files), len(validation_files))
    print("first train", train_files[0])
    # send to a yaml file
    file_paths = {
        "train": train_files,
        "val": validation_files,
    }

    print("num workers", config.num_workers)

    train_dataset = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=config.batch_size_train, shuffle=True, pin_memory=False)

    val_dataset = monai.data.Dataset(data=validation_files, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False, num_workers=0)

    return train_loader, val_loader, file_paths


def generate_folds_aggregator(direc, fold_train, fold_val, modalities, use_ref_volume=False):
    
    json_path = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "train.json")
    train_files, validation_files = datafold_read(
        dataset_path=os.getenv("DATASET_PATH"),
        fold_val=fold_val,
        fold_train=fold_train,
        modalities=modalities,
        json_path=json_path,
    )

    # Extract volume numbers for train and validation
    vols_train = [get_volume_number(file['image'][0]) for file in train_files]
    vols_val = [get_volume_number(file['image'][0]) for file in validation_files]

    print("train & validation counts for aggregator", len(vols_train), len(vols_val))
    
    train_paths, valid_paths = [], []

    def create_file_paths(num, use_ref_volume):
        base = {
            "image": join(direc, f"pred_{num}.npy"),
            "label": join(direc, f"label_{num}.npy"),
        }
        if use_ref_volume:
            base["ref_volume"] = join(direc, f"brain_volume_{num}.npy")
        return base


    for f in os.listdir(direc):
      if not 'brain_volume' in f:
        continue # just skip since we don't want duplicates

      num = os.path.splitext(f)[0].split("_")[-1] # this is fine...
      if num in vols_train:
        train_paths.append(create_file_paths(num, use_ref_volume))
      elif num in vols_val:
        valid_paths.append(create_file_paths(num, use_ref_volume))
      else:
        # throw exception
        raise Exception(f"Didn't find the current file {f} in train or val")

    print("final lengths of train and valid paths", len(train_paths), len(valid_paths))

    return train_paths, valid_paths

class Transpose_Transform(transforms.transform.MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.keys = keys

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            # originally (3, 141, 240, 240). consider using low res images though
            print(d[key].shape)
            d[key] = torch.permute(d[key], (0, 2,3,1)) # keep everything the same but add metadata

        return d


def get_aggregator_loader(batch_size, roi=(128,128,128), direc='./output_volumes', num_workers=0,
                          modalities=['t2f', 't1c', 't1n'],
                            fold_train=[0], fold_val=[1], use_ref_volume=False):
    from utils import set_deterministic

    set_deterministic(seed=1234)
    train_files, validation_files = generate_folds_aggregator(
        direc, fold_train, fold_val, modalities, use_ref_volume)
    
    # print("first few train files", train_files[:3])
    # print("first few validation files", validation_files[:3])
    image_label_arr = ["image", "label"] if not use_ref_volume else ["image", "label", "ref_volume"]
    

    train_transform = Compose(
        [
            AddNameFieldAggregator(keys=image_label_arr, send_real_path=False),
            LoadImaged(keys=image_label_arr),
            # Transpose_Transform(keys=image_label_arr),
            # EnsureChannelFirstd(keys=image_label_arr),
            CropForegroundd(
                keys=image_label_arr,
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            RandSpatialCropd(
                keys=image_label_arr,
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            RandFlipd(keys=image_label_arr, prob=0.3, spatial_axis=0),
            RandFlipd(keys=image_label_arr, prob=0.3, spatial_axis=1),
            RandFlipd(keys=image_label_arr, prob=0.3, spatial_axis=2),
            # rand zoom
            RandZoomd(keys=image_label_arr, prob=0.3, min_zoom=0.8, max_zoom=1.2),
            
        ]
    )
    val_transform = transforms.compose.Compose(
        [
            AddNameFieldAggregator(keys=image_label_arr, send_real_path=False),
            LoadImaged(keys=image_label_arr),
        ]
    )
    train_ds = monai.data.Dataset(data=train_files, transform=train_transform)
    

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_ds = monai.data.Dataset(data=validation_files, transform=val_transform)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader