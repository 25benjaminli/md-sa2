"""
For EACH model that is to be tested, we do the following:
- Check if data generation (via preprocess.py, generate_json.py, data_utils) work properly. 
    - This entails getting the right files and they must all exist.
- Check if metric collection is working properly (via MetricAccumulator from metrics.py)
- Save a few volumes and manually verify if they look correct

For MD-SA2 specifically, we need to test in two stages due to it being a two-stage model:
1) Test the SAM segmentation stage
2) Test the UNet refinement stage

MD-SA2 is run via pipeline.py. 
- Metric collection via cross-validation (or single-fold validation) must work properly without error

Because deep learning algos are not always going to generate the same results even with random seed, 
manually verify that the models are doing things correctly. 
"""

import torch

# use sys path append to import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import os
from data_utils import get_dataloaders
from omegaconf import OmegaConf
from MDSA2.eval.eval_mdsa2 import MDSA2
from monai.inferers import sliding_window_inference
from functools import partial
from monai.networks.nets import DynUNet

from metrics import MetricAccumulator, calculate_binary_dice
from utils import register_net_sam2, generate_rndm_path, AverageMeter, visualize_3D_volumes, set_deterministic, join
from MDSA2.eval.eval_mdsa2 import MDSA2, initialize_mdsa2
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

class TestData:
    @staticmethod
    def test_preprocess():
        os.chdir("..")
        os.system("python preprocess.py --config_folder sam2_tenfold")
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
        print("Data preprocessing test passed.")
    
    @staticmethod
    def test_dataloading():
        set_deterministic(42)
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        
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
        test_image = np.load("test_image_221.npy")
        test_label = np.load("test_label_221.npy")
        
        C, H, W, D = test_image.shape  # Assuming shape is (C, H, W, D)
        print(f"Image shape: {test_image.shape}")
        print(f"Label shape: {test_label.shape}")
        
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
    def test_onepass(use_unet=False):
        model_config = join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        mdsa2, train_loader, val_loader = initialize_mdsa2(model_config, use_unet=use_unet)
        
        mdsa2_averagemeter = AverageMeter(name="mdsa2")
        sa_averagemeter = AverageMeter(name="sa2")
        
        for batch in val_loader:
            with torch.no_grad():
                metrics_sa2, metrics_mdsa2 = mdsa2(batch)
                for key in metrics_sa2.keys():
                    sa_averagemeter.update(metrics_sa2[key]['avg'], n=1)
                if use_unet:
                    for key in metrics_mdsa2.keys():
                        mdsa2_averagemeter.update(metrics_mdsa2[key]['avg'], n=1)
        print("SA2 Average Metrics:", sa_averagemeter.avg, sa_averagemeter.cache)

        if use_unet:
            print("MD-SA2 Average Metrics:", mdsa2_averagemeter.avg, mdsa2_averagemeter.cache)
        
        print("MDSA2 one-pass test passed")

if __name__ == "__main__":
    # run all tests
    # TestData.test_preprocess()
    # TestData.test_dataloading()
    # TestData.visualize_test_and_image()
    TestMDSA2.test_onepass(use_unet=False)  # test only SAM stage
    TestMDSA2.test_onepass(use_unet=True)   # test full MD-SA2 model