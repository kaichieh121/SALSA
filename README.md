# CS448 Final Project

This project code is based on [![](https://img.shields.io/badge/Github-SALSA-blue)](https://github.com/thomeou/SALSA)

The dataset and challenge can be found at [![](https://img.shields.io/badge/Challenge-DCASE2021-red)](http://dcase.community/challenge2021/task-sound-event-localization-and-detection-results)

Description of this project can be found at [![](https://img.shields.io/badge/Project-CS448-gre)](CS448_Project.pdf)

## Sound event localization and detection
The task of sound event localization and detection (SELD) includes two subtasks, estimation and sound event detection (SED) and direction 
of arrival (DOA). This project restricts the discussion of the SELD problem to perhaps the most popular formalization of this task: 
Task 3 of IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events (DCASE2021). 
While different every year, datasets used for DCASE challenges essentially contain multiple sound event recordings of 
various categories, from multiple distances and directions of the array of microphones. These recordings are then emulated 
into sound scenes by filtering through room impulse responses, which varies in size, shape, acoustic reflective and absorptive 
properties. Proposed systems then has to solve the subtask of SED, which is to determine the occurrence and to predict the 
class of a sound event, and the subtask of DOA, which is to determine the Cartesian position of the sound event. 
Results are evaluated on their error rate (ER), F1-score (F), class-dependent localization error (LE), and localization 
recall (LR).

## Contribution
Here I list the main files that I added/modified for the purpose of this project as described in my written report. Some small modification 
might have been done to config files or experiment files but they are not important regarding to the scope of the grading of this project
```
./models/model_config/config.json
./models/seldnet_model.py
./models/stereonet_models.py
./models/stereonet_utils.py
./experiments/configs/seld.yml
```


## Prepare dataset and environment

This code is tested on Ubuntu with Python 3.8, CUDA 11.0 and Pytorch 2.0.0

1. Install the following dependencies by `pip install -r requirements.txt`. 

2. Download TAU-NIGENS Spatial Sound Events 2021 dataset [here](https://zenodo.org/record/4844825). 

3. Extract everything into the same folder. 

4. Data file structure should look like this:

```
./
├── feature_extraction.py
├── ...
└── data/
    ├──foa_dev
    │   ├── fold1_room1_mix001.wav
    │   ├── fold1_room1_mix002.wav  
    │   └── ...
    ├──foa_eval
    ├──metadata_dev
    ├──metadata_eval (might not be available yet)
    ├──mic_dev
    └──mic_eval
```

For TAU-NIGENS Spatial Sound Events 2021 dataset, please move wav files from subfolders `dev_train`, `dev_val`, 
`dev_test` to outer folder `foa_dev` or `mic_dev`. 

## Feature extraction

To extract **SALSA** feature, edit directories for data and feature accordingly in `tnsse_2021_salsa_feature_config.yml` in 
`dataset\configs\` folder. Then run `make salsa`

To extract **SALSA-Lite** feature, edit directories for data and feature accordingly in `tnsse_2021_salsa_lite_feature_config.yml` in 
`dataset\configs\` folder. Then run `make salsa-lite`

To extract *linspeciv*, *melspeciv*, *linspecgcc*, *melspecgcc* feature, 
edit directories for data and feature accordingly in `tnsse_2021_feature_config.yml` in 
`dataset\configs\` folder. Then run `make feature`

## Training and inference

To train SELD model with SALSA feature, edit the *feature_root_dir* and *gt_meta_root_dir* in the experiment config 
`experiments\configs\seld.yml`. Then run `make train`. 

To train SELD model with SALSA-Lite feature, edit the *feature_root_dir* and *gt_meta_root_dir* in the experiment config 
`experiments\configs\seld_salsa_lite.yml`. Then run `make train`. 

To do inference, run `make inference`. To evaluate output, edit the `Makefile` accordingly and run `make evaluate`.

