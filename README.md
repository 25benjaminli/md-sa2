## Code for [MD-SA2: optimizing Segment Anything 2 for multimodal, depth-aware brain tumor segmentation in sub-Saharan populations](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-12/issue-02/024007/MD-SA2--optimizing-Segment-Anything-2-for-multimodal-depth/10.1117/1.JMI.12.2.024007.full). 
 
This repository implements a hybrid Segment Anything 2 and U-Net architecture for tackling brain tumor segmentation with low-quality MRI scans from sub-Saharan Africa, as well as including a comprehensive suite of evaluation criteria & comparisons. The training script (and more analysis/visualization utilities) will be released at a later date (currently limited to evaluation). 

The goal of this repository is to provide a more streamlined implementation accessible to a wider audience, as compared to the original code which was not well compartmentalized. Please raise issues in the github repo if you are experiencing errors. 

## Tutorial

Note: replace base_folder with your base folder path. Also, make sure you've downloaded the weights. Make sure you have at plenty of extra space available on your system (>50 GB to be safe) - I have written scripts that preprocess the images and store to the disk, which consumes storage. Also, NVIDIA GPU with CUDA support is required for accelerated computing. 

**1. download things (dataset, weights)**

Clone this repository `git clone https://github.com/25benjaminli/md-sa2`. [base_folder] would be `md-sa2` by default, but feel free to change the name. 

Visit https://www.synapse.org/Synapse:syn51156910 and make a data request for the sub-Saharan Africa dataset. Once it's downloaded, create a folder called "data" in your [base_folder] and move your dataset into there. At this stage, it can be renamed (I renamed it to brats_africa for the purposes of clarify & ease). 

The weights for all models are available [here](https://drive.google.com/drive/folders/1aNFBVwMLzDVrq7z4rtE1ofvIq_U2oXtn). 


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

Go to the unit_tests folder and run unit_tests.py. This is for testing your dataloading setup and also evaluates MD-SA2 on the selected datafold. Feel free to add your own visualization scripts or modify it to run on specific samples. The general MD-SA2 class is also included in `models.py` if you want more control.

**5. comparisons**

If you want to run baseline U-Net models, visit `eval/eval_unet.py`. The U-net models were trained with different image dimensions (too much memory consumption and time spent with higher dimensions), so ensure that you have run the following prior to running `eval_unet.py`: `python preprocess.py --overrides preprocessing.resize_dims=[224,224]`. 

Running other vanilla segment-anything based models detailed in the paper will be available soon. Also, the option to use a low-rank adapted version of SA2 has been included - not originally part of the paper and sparsely tested. 

Helpful repositories during this study: 
- https://github.com/facebookresearch/sam2
- https://github.com/hitachinsk/SAMed