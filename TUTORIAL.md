## Process
Currently the process for training & evaluating the model is clunky and is *not* meant for production use. Please raise issues in the github repository if you are experiencing errors during implementation. The goal is to port the entire process over soon to a docker image to reduce complexity. 

Note: replace base_folder with your base folder path

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
pip install -e .
```
resolving the max int problem: https://github.com/rcremese/MONAI/commit/41076d16e988d46b75331141aa965cbd1948d3bf

**4. update your environmental variables (example given below).**
```
PROJECT_PATH="/home/.../MDSA-2"
DATASET_PATH="/home/.../MDSA-2/data/brats_africa"
UNREFINED_VOLUMES_PATH="/home/.../MDSA-2/data/unrefined_volumes_3D"
REFINED_VOLUMES_PATH="/home/.../MDSA-2/data/refined_volumes_3D"
PREPROCESSED_PATH="/home/.../MDSA-2/data/preprocessed"
DATA_PATH="/home/.../MDSA-2/data"
```

**5. run preprocessing code.**
```
cd [base_folder]/MDSA2
python preprocess.py --config_folder sam2_tenfold
cd [base_folder]/MDSA2
python generate_json.py --config_folder sam2_tenfold --use_preprocessed
```
**7. to run cross validation, run the following**
```
cd [base_folder]/MDSA2
python run_cross_validation.py --config_folder sam2_tenfold
```
If you want to run the aggregator, you must move all the train logs to a singular folder. An example is this:

sam_tenfold_cv
---> sam2_tenfold_cross_validation_fold_0
---> sam2_tenfold_cross_validation_fold_1
...

**8. to run the aggregator, run the following**
```
cd [base_folder]/MDSA2/train
python run_aggregator_cv.py --folder_with_cv sam_tenfold_cv
```

**you can also run the aggregator separately**

`python run_aggregator.py --run_name sam_tenfold_cv/sam2_tenfold_cross_validation_fold_0`


**run the end-to-end version by doing the following**

`python pipeline.py --fold_eval`