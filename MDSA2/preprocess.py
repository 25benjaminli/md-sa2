import argparse

from utils import set_deterministic
from data_utils import preprocess, load_from_expected_json

import os
from omegaconf import OmegaConf
from dotenv import load_dotenv
from utils import join

load_dotenv(override=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_folder', type=str, default='sam2_tenfold', help='folder containing configurations for the experiment')
    parser.add_argument('--no_normalize', action='store_true', help='dont use preprocessed data')
    parser.add_argument('--overrides', type=str, help='additional overrides for config folder')
    parser.add_argument('--expected_folds', type=str, help='load from expected folds')
    args = parser.parse_args()

    # first, generate the json
    if args.expected_folds is not None:
        load_from_expected_json(args.expected_folds, modalities=['t2f', 't1c', 't1n'], preprocessed=False, ending='nii.gz')
    else:
        os.chdir(join(os.getenv("PROJECT_PATH", ""), "MDSA2"))
        os.system("python generate_json.py --config_folder " + args.config_folder)


    # load config
    print("-------------- USING CONFIG: ", args.config_folder)
    config_final = OmegaConf.load(open(join('config', args.config_folder, 'config_train.yaml'), 'r'))
    details = OmegaConf.load(open(join('config', args.config_folder, 'details.yaml'), 'r'))
    # print("config img size", config_final.img_size)

    # apply overrides
    if args.overrides is not None:
        # split the string by space, e.g. key1=value1 key2=value2
        overrides_list = args.overrides.split(' ')
        print("Applying overrides: ", overrides_list)
        config_final = OmegaConf.merge(config_final, OmegaConf.from_dotlist(overrides_list)) 
        print("preprocess resize dims", config_final.preprocessing.resize_dims)

    # merge details into config_final
    config_final = OmegaConf.merge(config_final, details)
    config_final.dataset = "brats_africa"

    config_final.sam_ckpt = join(os.getenv('PROJECT_PATH', ""),"MDSA2", config_final.sam_ckpt)
    print("*******fine-tuned ckpt", config_final.ft_ckpt)
    
    if config_final.ft_ckpt != None:
        config_final.ft_ckpt = join(os.getenv('PROJECT_PATH', ""),"MDSA2",config_final.ft_ckpt)
    
    set_deterministic(42)
    
    print("****All Arguments****")

    # remove the preprocessed folder
    import shutil
    print("removing preprocessed folder")
    if os.path.exists(os.getenv("PREPROCESSED_PATH", "")):
        shutil.rmtree(os.getenv("PREPROCESSED_PATH", ""))

    preprocess(config_final, use_normalize=not args.no_normalize)
    os.chdir(join(os.getenv("PROJECT_PATH", ""), "MDSA2"))

    # run generate json script
    if args.expected_folds is None:
        os.system("python generate_json.py --config_folder " + args.config_folder + " --use_preprocessed")
    else:
        load_from_expected_json(args.expected_folds, modalities=['t2f', 't1c', 't1n'], preprocessed=True, ending='npy')
    
    config_final = OmegaConf.merge(OmegaConf.create({"config_folder": args.config_folder}), config_final)
    OmegaConf.save(config_final, join(os.getenv("DATA_PATH", ""), 'current_data_config.yaml'))