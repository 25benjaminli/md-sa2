import argparse

from utils import set_deterministic
from data_utils import preprocess

import os
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv(override=True)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_folder', type=str, default='sam2_tenfold', help='folder containing configurations for the experiment')
    parser.add_argument('--no_normalize', action='store_true', help='dont use preprocessed data')

    args = parser.parse_args()

    # first, generate the json
    os.chdir(os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2"))

    os.system("python generate_json.py --config_folder " + args.config_folder)
    # os.chdir(os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2", "train"))


    # load config
    print("-------------- USING CONFIG: ", args.config_folder)
    config_final = OmegaConf.load(open(os.path.join('config', args.config_folder, 'config_train.yaml'), 'r'))
    details = OmegaConf.load(open(os.path.join('config', args.config_folder, 'details.yaml'), 'r'))

    # merge details into config_final
    config_final = OmegaConf.merge(config_final, details)
    config_final.dataset = "brats_africa"

    config_final.sam_ckpt = os.path.join(os.getenv('PROJECT_PATH', ""),"MDSA2", config_final.sam_ckpt)
    
    if config_final.ft_ckpt != None:
        print("*******fine-tuned ckpt", config_final.ft_ckpt)
        config_final.ft_ckpt = os.path.join(os.getenv('PROJECT_PATH', ""),"MDSA2",config_final.ft_ckpt)
    else:
        print("*******fine-tuned ckpt", config_final.ft_ckpt)
        config_final.ft_ckpt = None
    
    set_deterministic(config_final.seed)
    
    print("****All Arguments****")

    # remove the preprocessed folder
    import shutil
    print("removing preprocessed folder")
    if os.path.exists(os.getenv("PREPROCESSED_PATH", "")):
        shutil.rmtree(os.getenv("PREPROCESSED_PATH", ""))

    preprocess(config_final, use_normalize=not args.no_normalize)
    os.chdir(os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2"))

    # run generate json script
    os.system("python generate_json.py --config_folder " + args.config_folder + " --use_preprocessed")

    config_final = OmegaConf.merge(OmegaConf.create({"config_folder": args.config_folder}), config_final)
    OmegaConf.save(config_final, os.path.join(os.getenv("DATA_PATH", ""), 'current_data_config.yaml'))