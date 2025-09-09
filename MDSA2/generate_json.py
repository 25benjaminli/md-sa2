# UNDER DEVELOPMENT

import argparse
import json
import sys

sys.path.append('..')

from utils import set_deterministic, join
from data_utils import generate_json

sys.path.append('./train')

import os

from omegaconf import OmegaConf
from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_folder', type=str, default='sam2_tenfold', help='folder containing configurations for the experiment')
    parser.add_argument('--use_preprocessed', action='store_true', help='use preprocessed data')
    args = parser.parse_args()

    # load config
    print("GENERATING JSON WITH CONFIG: ", args.config_folder)
    config = OmegaConf.load(open(join('config', args.config_folder, 'config_train.yaml'), 'r'))

    set_deterministic(config.seed)
    if args.use_preprocessed:
        dataset_path = os.getenv("PREPROCESSED_PATH")
        ending = "npy"
    else:
        dataset_path = os.getenv("DATASET_PATH")
        ending = "nii.gz"

    print("dataset path", dataset_path)

    di, fold_data_orig = generate_json(dataset_path=dataset_path, modalities=config.modalities, fold_num=config.fold_num, \
                        seg_key='seg', config=config, ending=ending) # only use three modalities, each fold is 369/16=23
    
    json_path = "train.json"
    with open(json_path, 'w') as f:
        json.dump(di, f, indent=4)

    fold_data_orig_path = "fold_data.json"
    # send sorted fnames to json
    with open(fold_data_orig_path, 'w') as f:
        json.dump(fold_data_orig, f, indent=4)