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

Because deep learning algos are not always going to generate the same results even with random seed, manually verify that the models are doing things correctly. 
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
from pipeline import MDSA2
from monai.inferers import sliding_window_inference
from functools import partial
from monai.networks.nets import DynUNet

from metrics import MetricAccumulator, calculate_binary_dice
from utils import register_net_sam2, generate_rndm_path, AverageMeter, visualize_3D_volumes, set_deterministic
from pipeline import MDSA2
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

class TestData:
    @staticmethod
    def test_preprocess():
        os.chdir("..")
        os.system("python preprocess.py --config_folder sam2_tenfold")
        # confirm that the preprocessed folder exists and has number of files equivalent to brats_africa dataset
        assert os.path.exists(os.path.join(os.getenv("PROJECT_PATH", ""), "data", "preprocessed")), "Preprocessed folder does not exist."
        preprocessed_folder_names = os.listdir(os.path.join(os.getenv("PROJECT_PATH", ""), "data", "preprocessed"))
        brats_africa_folder_names = os.listdir(os.path.join(os.getenv("PROJECT_PATH", ""), "data", "brats_africa"))
        # the fnames should ALL match each other
        assert sorted(preprocessed_folder_names) == sorted(brats_africa_folder_names), "File names in preprocessed folder do not match brats_africa dataset."

        # check if the files within within each folder in preprocessed are correct (i.e. it should be 3 modalities: t1c, t1n, t2f)
        # BraTS-SSA-00002-000-seg.npy
        for folder_name in preprocessed_folder_names:
            folder_path = os.path.join(os.getenv("PREPROCESSED_PATH", ""), folder_name)
            files = os.listdir(folder_path)
            assert len(files) == 4, f"Folder {folder_name} does not contain 4 files (3 modality, one seg)."
            modalities = [f.split('-')[-1].split('.')[0] for f in files]
            assert sorted(modalities) == sorted(['seg', 't1c', 't1n', 't2f']), f"Folder {folder_name} does not contain correct modality files."
        print("Data preprocessing test passed.")
    
    @staticmethod
    def test_dataloading():
        # this tests transforms and data loading
        set_deterministic(42)
        model_config = os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        
        train_loader, val_loader, file_paths = get_dataloaders(model_config, use_preprocessed=True, modality_to_repeat=-1)
        # check if the dataloaders are not empty
        assert len(train_loader) > 0, "Train dataloader is empty."
        assert len(val_loader) > 0, "Validation dataloader is empty."
        # check if the first batch has the correct shape
        for batch in train_loader:
            images, labels, image_title = batch["image"], batch["label"], batch["image_title"]

            assert images.shape == (4, 3, 384, 384, 155), f"Expected 5D tensor for images, got {images.shape}."
            assert labels.shape == (4, 3, 384, 384, 155), f"Expected 5D tensor for labels, got {labels.shape}."
            # labels should only be binarized
            unique_labels = torch.unique(labels)
            print("Unique labels in the label:", unique_labels)
            assert all(l in [0, 1] for l in unique_labels), f"Labels should be binarized to 0 and 1, got {unique_labels}."
            
            # save the first image and label to disk, then you can manually verify that it's the same vol/label as in the original dataset
            np.save(f"test_image_{image_title[0]}.npy", images[0].cpu().numpy())
            np.save(f"test_label_{image_title[0]}.npy", labels[0].cpu().numpy())
            break

        print("Data utilities test passed.")

    @staticmethod
    def visualize_test_and_image():
        # load the saved test image and label
        test_image = np.load("test_image_221.npy")
        test_label = np.load("test_label_221.npy")
        
        # Get dimensions
        C, H, W, D = test_image.shape  # Assuming shape is (C, H, W, D)
        print(f"Image shape: {test_image.shape}")
        print(f"Label shape: {test_label.shape}")
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows (image, label), 3 cols (channels)
        plt.subplots_adjust(bottom=0.25)  # Leave space for slider
        
        # Initial slice index (start from middle)
        initial_slice = D // 2
        
        def update_plots(slice_idx):
            slice_idx = int(slice_idx)
            
            # Clear all axes
            for i in range(2):
                for j in range(3):
                    axes[i, j].clear()
            
            # Plot image channels in first row
            for channel in range(min(3, C)):  # Plot up to 3 channels
                axes[0, channel].imshow(test_image[channel, :, :, slice_idx], cmap='gray')
                axes[0, channel].set_title(f'Image Channel {channel} - Slice {slice_idx}')
                axes[0, channel].axis('off')
            
            # Plot label channels in second row
            for channel in range(min(3, C)):  # Plot up to 3 channels
                axes[1, channel].imshow(test_label[channel, :, :, slice_idx], cmap='gray')
                axes[1, channel].set_title(f'Label Channel {channel} - Slice {slice_idx}')
                axes[1, channel].axis('off')
            
            # Hide unused subplots if C < 3
            for channel in range(C, 3):
                axes[0, channel].axis('off')
                axes[1, channel].axis('off')
            
            fig.canvas.draw_idle()
        
        # Initial plot
        update_plots(initial_slice)
        
        # Create slider
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
        depth_slider = Slider(ax_slider, 'Depth', 0, D-1, valinit=initial_slice, valstep=1)
        
        # Connect slider to update function
        depth_slider.on_changed(update_plots)
        
        plt.tight_layout()
        plt.show()
        print("Visualization complete, manually verify the saved .npy files.")

class TestMetricAccumulator:
    @staticmethod
    def test_accumulator():
        # Create a MetricAccumulator instance
        accumulator = MetricAccumulator()

        # Simulate some predictions and ground truths
        # y_true should be the same dimensions as a brain mri scan: (e.g. B, C, H, W)
        # y_true can only be 0, 1, 2, 3 representing different classes in seg
        repeat = 5
        expected_dice = 0
        expected_iou = 0
        for _ in range(repeat):
            y_true = np.random.randint(0, 4, (2, 1, 64, 64))  # Example ground truth with 4 classes
            # y_pred adds noise to y_true
            y_pred = y_true.copy()
            noise = np.random.binomial(1, 0.1, y_true.shape)
            y_pred = np.clip(y_pred + noise, 0, 3)  # Ensure values stay within valid range
            # round them to get 0, 1, 2, 3 values
            y_true = torch.from_numpy(y_true).round()
            y_pred = torch.from_numpy(y_pred).round()

            assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match."
            assert len(y_true.shape) == 4, "y_true and y_pred must be 4D tensors."
            # assert all values are integer values and between 0 and 3
            unique_true = torch.unique(y_true)
            unique_pred = torch.unique(y_pred)
            assert unique_true.min() >= 0 and unique_true.max() <= 3 and len(unique_true) == 4, "y_true values must be between 0 and 3."
            assert unique_pred.min() >= 0 and unique_pred.max() <= 3 and len(unique_pred) == 4, "y_pred values must be between 0 and 3."

            accumulator.update(y_true, y_pred)

            expected_dice += calculate_binary_dice(y_true.numpy(), y_pred.numpy())
            expected_iou += expected_dice / (2 - expected_dice)

        dice_score = accumulator.metric_dict["dice"].avg
        iou_score = accumulator.metric_dict["iou"].avg

        expected_dice /= repeat
        expected_iou /= repeat

        assert np.isclose(dice_score, expected_dice), f"Dice score mismatch: {dice_score} vs {expected_dice}"
        assert np.isclose(iou_score, expected_iou), f"IoU score mismatch: {iou_score} vs {expected_iou}"
        print("MetricAccumulator tests passed.")

class TestMDSA2:
    @staticmethod
    def test_onepass(use_unet=False):
        model_config = os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2", "config", "sam2_tenfold", "config_train.yaml")
        model_config = OmegaConf.load(model_config)
        
        train_loader, val_loader, file_paths = get_dataloaders(model_config, use_preprocessed=True, modality_to_repeat=-1)
        
        path_thing = os.path.join(f"{model_config.config_folder}_cv", f"cv_fold_{model_config.fold_eval}")

        model_config.ft_ckpt = os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2", "train", "runs", "brats_africa", path_thing, "best_model_sam2.pth")
        model_config.batch_size_train=1 # manually set to 1 for comparison
        model_config.batch_size_val=1 # manually set to 1 for comparison
        model_config.snapshot_path = generate_rndm_path(os.path.join("eval", "runs", "aggregator"))
        
        assert os.path.exists(model_config.ft_ckpt), f"fine tuned ckpt {model_config.ft_ckpt} does not exist"
        print("model checkpoint", model_config.ft_ckpt)

        sam2 = register_net_sam2(model_config)
        if use_unet:
            checkpoint_to_load = os.path.join(os.getenv("PROJECT_PATH", ""), "MDSA2", "train", "runs", "aggregator", path_thing, "best_model.pth")
            unet_loaded = torch.load(checkpoint_to_load)
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
            
            aggregator_model.load_state_dict(unet_loaded['state_dict'])

            unet_inferer = partial(
                sliding_window_inference,
                roi_size=[model_config.roi[0], model_config.roi[1], model_config.roi[2]],
                sw_batch_size=model_config.config.sw_batch_size,
                predictor=aggregator_model,
                overlap=model_config.config.infer_overlap,
            )
        else:
            unet_inferer = None
        model = MDSA2(sam2, unet_inferer=unet_inferer, config=model_config)
        model.eval()
        mdsa2_averagemeter = AverageMeter(name="mdsa2")
        sa_averagemeter = AverageMeter(name="sa2")
        for batch in val_loader:
            with torch.no_grad():
                metrics_sa2, metrics_mdsa2 = model(batch)
                for key in metrics_sa2.keys():
                    sa_averagemeter.update(metrics_sa2[key]['avg'], n=1)
                if use_unet:
                    for key in metrics_mdsa2.keys():
                        mdsa2_averagemeter.update(metrics_mdsa2[key]['avg'], n=1)

        print("SA2 Average Metrics:", sa_averagemeter.avg, sa_averagemeter.cache)

        if use_unet:
            print("MD-SA2 Average Metrics:", mdsa2_averagemeter.avg, mdsa2_averagemeter.cache)
        
        print("MDSA2 one-pass test passed.")

if __name__ == "__main__":
    # run all tests
    TestData.test_preprocess()
    TestData.test_dataloading()
    TestData.visualize_test_and_image()
    TestMetricAccumulator.test_accumulator()
    TestMDSA2.test_onepass(use_unet=False)  # test only SAM stage
    TestMDSA2.test_onepass(use_unet=True)   # test full MD-SA2 model