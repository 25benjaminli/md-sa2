## Code for [MD-SA2: optimizing Segment Anything 2 for multimodal, depth-aware brain tumor segmentation in sub-Saharan populations](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-12/issue-02/024007/MD-SA2--optimizing-Segment-Anything-2-for-multimodal-depth/10.1117/1.JMI.12.2.024007.full). 
 
This repository is a revamp (to improve clarity, reduce bloat, and ease reproducibility) of my original code for this paper. MD-SA2 is a hybrid architecture (Segment Anything 2 + dynamic UNet w/ deep supervision) for brain tumor segmentation with low-quality MRI scans from sub-Saharan Africa, as well as a comprehensive suite of evaluation criteria & comparisons. See the checklist at the bottom of this README for which modules are currently usable. 

## Tutorial

Note: replace [base_folder] with your base folder path. Also, make sure you've downloaded the weights. Make sure you have at plenty of extra space available on your system (>50 GB to be safe) - I have written scripts that preprocess the images and store to the disk, which consumes storage. Also, NVIDIA GPU with CUDA support is required for accelerated computing (mine is an RTX 3060 and has CUDA version 12.2). 

**1. download things (dataset, weights)**

Clone this repository `git clone https://github.com/25benjaminli/md-sa2`. [base_folder] would be `md-sa2` by default, but feel free to change the name. 

Visit https://www.synapse.org/Synapse:syn51156910 and make a data request for the sub-Saharan Africa dataset. Once it's downloaded, create a folder called "data" in your [base_folder] and move your dataset into there. 

The weights for all models are available [here](https://drive.google.com/drive/folders/1aNFBVwMLzDVrq7z4rtE1ofvIq_U2oXtn). Expected metrics are also included in the same parent directory but separate folder for your reference. 


**2. install dependencies for the base project.**
Create a conda environment (python 3.10). Then run:
```
cd [base_folder]
pip install -r requirements.txt
```

**3. install dependencies for SA2**
```
cd [base_folder]/MDSA2/segment_anything_2
pip install --no-build-isolation -e .
```

**4. update your environmental variables, e.g. create a file called .env (example given below).**
```
PROJECT_PATH="/home/.../md-sa2"
DATASET_PATH="/home/.../md-sa2/data/brats_africa"
UNREFINED_VOLUMES_PATH="/home/.../md-sa2/data/unrefined_volumes_3D"
REFINED_VOLUMES_PATH="/home/.../md-sa2/data/refined_volumes_3D"
PREPROCESSED_PATH="/home/.../md-sa2/data/preprocessed"
DATA_PATH="/home/.../md-sa2/data"
```
**4. run unit tests**

Go to the unit_tests folder and run unit_tests.py. This is for preprocessing the data, testing your dataloading setup and also evaluates MD-SA2 on the selected datafold. Feel free to add your own visualization scripts or modify it to run on specific samples. The general MD-SA2 class is also included in `models.py` if you want more control.

**5. comparisons**

Running MD-SA2 on all folds: go to `eval` folder and run `eval_mdsa2.py`. 

If you want to run baseline U-Net models, run `eval_unet.py`. The U-net models were trained with different image dimensions (too much memory consumption and time spent with higher dimensions), so ensure that you have run the following prior to running `eval_unet.py`: `python preprocess.py --overrides preprocessing.resize_dims=[224,224] --expected_folds expected_folds.json`. If you intend to run MD-SA2 after running U-Net, be sure to rerun the `python preprocess.py` for the dimensions to match up. 

Also, visit `extra_software/visualization/vis_volume.py` for some cool visualizations of predictions, labels, and brain volumes. Note that this does require you to have run MD-SA2 or the other models, collected their predictions, and moved them into a folder under this directory called "volumes" (this should be done automatically during MD-SA2's validation process, if not available then do it manually). 

Running other vanilla segment-anything based models detailed in the paper will be available soon. Also, the option to use a low-rank adapted version of SA2 has been included - not originally part of the paper and sparsely tested. Based on [another repository](https://github.com/25benjaminli/sam2lora) I developed.

Note: this code was primarily tested on a machine running linux. Windows may or may not be supported. 

Below are some repositories/libraries that were very helpful during this study (code samples borrowed from and modified): 
- https://github.com/facebookresearch/sam2
- https://github.com/hitachinsk/SAMed
- https://github.com/Project-MONAI/MONAI


Working modules
- [x] unit_tests.py
- [x] train_mdsa2.py
- [x] eval_mdsa2.py
- [x] eval_unet.py
- [x] train_unet.py
- [x] data_ablations.py
- [x] model_ablations.py

As a refresher:
1. Data ablations: evaluate sa2 (tiny model) with different combos of modalities - t2f only, t1c only, t1 only, t1w only, t2f+t1c+t1
2. Model ablations: compare sa1-LoRA/sa2-b+/sa2-t/mdsa2/medsam. Only sa1-LoRA + medsam (zero-shot) are included in the file since metrics for other models can be acquired via other training steps (e.g. customize config for b+ model). 

For further comparisons, I have not included them in this repository to avoid too much crowding.
- View my fork of [this](https://github.com/25benjaminli/foreground-bt) repository to compile the results for the [few-shot algorithm](https://pmc.ncbi.nlm.nih.gov/articles/PMC10093064/)
- See my fork [this](https://github.com/25benjaminli/Medical-SAM-Adapter) repository to compile 
results for [Medical SAM Adapter](https://www.sciencedirect.com/science/article/pii/S1361841525000945)