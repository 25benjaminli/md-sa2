## Process

Note: replace base_folder with your base folder path. Also, make sure you've downloaded the weights. They must be transferred to [TODO: find a good way to process this]. 

**1. download dataset**
Visit https://www.synapse.org/Synapse:syn51156910 and make a data request for the sub-Saharan Africa dataset. Once it's downloaded, create a folder called "data" in your [base_folder] and move your dataset into there. At this stage, it can be renamed (I renamed it to brats_africa for the purposes of clarify & ease).

**2. install dependencies for the base project.**
```
cd [base_folder]
pip install -r requirements.txt
```

**3. install dependencies for the segment anything 2**
```
cd [base_folder]/MDSA2/segment_anything_2
pip install --no-build-isolation -e .
```
resolving the max int problem: https://github.com/rcremese/MONAI/commit/41076d16e988d46b75331141aa965cbd1948d3bf

**4. update your environmental variables, e.g. create a file called .env (example given below).**
```
PROJECT_PATH="/home/.../MDSA-2"
DATASET_PATH="/home/.../MDSA-2/data/brats_africa"
UNREFINED_VOLUMES_PATH="/home/.../MDSA-2/data/unrefined_volumes_3D"
REFINED_VOLUMES_PATH="/home/.../MDSA-2/data/refined_volumes_3D"
PREPROCESSED_PATH="/home/.../MDSA-2/data/preprocessed"
DATA_PATH="/home/.../MDSA-2/data"
```
**4. run unit tests**

Go to the unit_tests folder and run unit_tests.py. This will run some miscellaneous tests to see if your setup is functioning. 

**5. run MDSA2**

Run pipeline.py, which has arguments:
- fold_eval
- do_cross_validation
- save_volumes
- config_folder

Eventually, I will get around to building a component that allows you to watch the segmentation be generated live. 

**6. comparisons**

If you want to run other U-Net models, run unet_models with the corresponding input arguments. 

If you want to run other vanilla segment-anything based models, run `eval_pretrain.py` within the `eval` folder. 