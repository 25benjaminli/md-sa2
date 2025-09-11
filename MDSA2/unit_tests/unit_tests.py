"""
- Check if data generation (via preprocess.py, generate_json.py, data_utils) work properly. 
    - This entails getting the right files and they must all exist.
- Check if metric collection is working properly (via MetricAccumulator from metrics.py)
- Save a few volumes and manually verify if they look correct

For MD-SA2 specifically, we need to test in two stages due to it being a two-stage model:
1) Test the SAM segmentation stage
2) Test the UNet refinement stage

- Metric collection via cross-validation (or single-fold validation) must work properly without error

Because deep learning algos are not always going to generate the same results even with random seed, 
manually verify that the models are doing things correctly. 
"""

import torch

# use sys path append to import from parent directory
import sys
import os
from pathlib import Path

# Add parent directory
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import os
from data_utils import get_dataloaders
from omegaconf import OmegaConf
from utils import set_deterministic, join
from models import initialize_mdsa2

sys.path.pop(0)

# Set matplotlib to use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json

class TestData:
    @staticmethod
    def test_preprocess():
        os.chdir("..")
        os.system("python preprocess.py --config_folder sam2_tenfold --expected_folds expected_folds.json")
        os.chdir("unit_tests")
        # confirm that the preprocessed folder exists and has number of files equivalent to brats_africa dataset
        assert os.path.exists(join(os.getenv("PROJECT_PATH", ""), "data", "preprocessed")), "Preprocessed folder does not exist."
        preprocessed_folder_names = os.listdir(join(os.getenv("PROJECT_PATH", ""), "data", "preprocessed"))
        brats_africa_folder_names = os.listdir(join(os.getenv("PROJECT_PATH", ""), "data", "brats_africa"))
        # the fnames should ALL match each other
        assert sorted(preprocessed_folder_names) == sorted(brats_africa_folder_names), "File names in preprocessed folder do not match brats_africa dataset."

        # check if the files within within each folder in preprocessed are correct (i.e. it should be 3 modalities: t1c, t1n, t2f)
        # BraTS-SSA-00002-000-seg.npy
        for folder_name in preprocessed_folder_names:
            folder_path = join(os.getenv("PREPROCESSED_PATH", ""), folder_name)
            files = os.listdir(folder_path)
            assert len(files) == 4, f"Folder {folder_name} does not contain 4 files (3 modality, one seg)."
            modalities = [f.split('-')[-1].split('.')[0] for f in files]
            assert sorted(modalities) == sorted(['seg', 't1c', 't1n', 't2f']), f"Folder {folder_name} does not contain correct modality files."
        
        # this is a pretty janky way to check that they match, but it works
        # read train.json and confirm that all fold 0s correspond to expected_folds.json
        with open(join(os.getenv("PROJECT_PATH", ""), "MDSA2", "train.json"), 'r') as f:
            train_data = json.load(f)
        
        with open(join(os.getenv("PROJECT_PATH", ""), "MDSA2", "expected_folds.json"), 'r') as f:
            expected_folds = json.load(f)
            expected_folds = {int(k): v for k, v in expected_folds.items()}

        end_dict = [[] for _ in range(10)]
        for item in train_data["training"]:
            # get directory name of the item["image"][0]
            end_dict[item["fold"]].append(os.path.basename(os.path.dirname(item["image"][0])))

        end_dict = [sorted(fold) for fold in end_dict]

        for i, fold in enumerate(end_dict):
            assert fold == expected_folds[i], f"Fold {i} does not match expected folds, found {fold}, expected {expected_folds[i]}"

        print("Data preprocessing test passed.")
    
    @staticmethod
    def test_dataloading():
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        set_deterministic(42)
        
        train_loader, val_loader, file_paths = get_dataloaders(model_config, use_preprocessed=True, modality_to_repeat=-1)
        # check if the dataloaders are not empty
        assert len(train_loader) > 0, "Train dataloader is empty"
        assert len(val_loader) > 0, "Validation dataloader is empty"
        # check if the first batch has the correct shape
        for batch in train_loader:
            images, labels, image_title = batch["image"], batch["label"], batch["image_title"]

            assert images.shape == (4, 3, 384, 384, 155), f"Expected 5D tensor for images, got {images.shape}"
            assert labels.shape == (4, 3, 384, 384, 155), f"Expected 5D tensor for labels, got {labels.shape}"
            # labels should only be binarized
            unique_labels = torch.unique(labels)
            print("Unique labels in the label:", unique_labels)
            assert all(l in [0, 1] for l in unique_labels), f"Labels should be binarized to 0 and 1, got {unique_labels}"
            
            # save the first image and label to disk, then you can manually verify that it's the same vol/label as in the original dataset
            np.save(f"test_image_{image_title[0]}.npy", images[0].cpu().numpy())
            np.save(f"test_label_{image_title[0]}.npy", labels[0].cpu().numpy())
            break


        print("Data utilities test passed")

    @staticmethod
    def visualize_test_and_image():
        # load the saved test image and label
        curr_files = os.listdir(".")
        available_images = [f for f in curr_files if f.startswith("test_image_") and f.endswith(".npy")]
        if not available_images:
            print("no test images found run test_dataloading() first.")
            return
        random_image_file = np.random.choice(available_images)
        test_image = np.load(random_image_file)
        test_label = np.load(random_image_file.replace("test_image_", "test_label_"))
        
        C, H, W, D = test_image.shape  # Assuming shape is (C, H, W, D)
        print(f"image shape: {test_image.shape}")
        print(f"label shape: {test_label.shape}")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
        plt.subplots_adjust(bottom=0.25)
        
        initial_slice = 0
        
        def update_plots(slice_idx):
            slice_idx = int(slice_idx)
            
            for i in range(2):
                for j in range(3):
                    axes[i, j].clear()
            
            for channel in range(min(3, C)):
                axes[0, channel].imshow(test_image[channel, :, :, slice_idx], cmap='gray')
                axes[0, channel].set_title(f'Image Channel {channel} - Slice {slice_idx}')
                axes[0, channel].axis('off')
            
            for channel in range(min(3, C)):
                axes[1, channel].imshow(test_label[channel, :, :, slice_idx], cmap='gray')
                axes[1, channel].set_title(f'Label Channel {channel} - Slice {slice_idx}')
                axes[1, channel].axis('off')
            
            for channel in range(C, 3):
                axes[0, channel].axis('off')
                axes[1, channel].axis('off')
            
            fig.canvas.draw_idle()
        
        update_plots(initial_slice)
        
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
        depth_slider = Slider(ax_slider, 'Depth', 0, D-1, valinit=initial_slice, valstep=1)
        
        depth_slider.on_changed(update_plots)
        
        plt.tight_layout()
        plt.show()
        print("Visualization complete, manually verify the saved .npy files")

class TestMDSA2:
    @staticmethod
    def test_onepass(use_unet=True, fold_eval=0):
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        model_config.config_folder = "sam2_tenfold"
        model_config.fold_eval = fold_eval
        details = OmegaConf.load(open(join(os.getenv("PROJECT_PATH", ""), 'MDSA2', 'config', "sam2_tenfold", 'details.yaml'), 'r'))
        model_config = OmegaConf.merge(model_config, details)
        model_config.dataset = "brats_africa"
        # set batch size to 1 for comparison w/ unet
        set_deterministic(42)
        train_loader, val_loader, file_paths = get_dataloaders(model_config, use_preprocessed=True, modality_to_repeat=-1, verbose=False)

        mdsa2 = initialize_mdsa2(model_config, dataloaders=(train_loader, val_loader), use_unet=use_unet)
        
        for batch in val_loader:
            with torch.no_grad():
                metrics_sa2, metrics_mdsa2 = mdsa2(batch)
                # print("SA2 metrics:", metrics_sa2)
                # print("MD-SA2 metrics:", metrics_mdsa2)
        
        # send to json
        with open(f"test_metrics_sa2_fold_{model_config.fold_eval}.json", "w") as f:
            json.dump(mdsa2.metric_accumulator_sa2.get_metrics(), f, indent=4)

        if use_unet:
            with open(f"test_metrics_mdsa2_fold_{model_config.fold_eval}.json", "w") as f:
                json.dump(mdsa2.unet_model.metric_accumulator.get_metrics(), f, indent=4)
        print("MDSA2 one-pass test passed for fold", model_config.fold_eval)

if __name__ == "__main__":
    # run all tests
    TestData.test_preprocess()
    TestData.test_dataloading()
    TestData.visualize_test_and_image()
    TestMDSA2.test_onepass(use_unet=False, fold_eval=0)  # test only SAM stage
    TestMDSA2.test_onepass(use_unet=True, fold_eval=0)   # test full MD-SA2 model