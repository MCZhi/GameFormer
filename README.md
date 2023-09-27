# GameFormer

This repository contains the code for the ICCV'23 paper:

**GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](https://arxiv.org/abs/2303.05760)**&nbsp;**[[Project Website]](https://mczhi.github.io/GameFormer/)**

## Overview
In this repository, you can expect to find the following features:

Included 🤟:
* Code for interaction prediction using a joint model on Waymo Open Motion Dataset (WOMD)
* Code for open-loop planning on selected dynamic scenarios within WOMD

Not included 😵:
* Code for the marginal model with EM ensemble for interaction prediction on WOMD
* Code for closed-loop planning on WOMD. Please refer to our previous work [DIPP](https://github.com/MCZhi/DIPP) for that.
* Code for packaging and submitting prediction results to the WOMD Interaction Prediction Challenge

For those interested in the nuPlan dataset experimentation, we invite you to visit the [GameFormer Planner](https://github.com/MCZhi/GameFormer-Planner) repository, which provides a more comprehensive planning framework.

## Dataset and Environment
### 1. Download
- Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1. Utilize data from ```scenario/training_20s``` or ```scenario/training``` for training, and data from ```scenario/validation``` and ```scenario/validation_interactive``` for testing.
- Clone this repository and navigate to the directory:
```
git clone https://github.com/MCZhi/GameFormer.git && cd GameFormer
```

### 2. Environment Setup
- Create a conda environment:
```
conda create -n gameformer python=3.8
```
- Activate the conda environment:
```
conda activate gameformer
```
- Install the required packages:
```
pip install -r requirements.txt
```

## Interaction Prediction
Navigate to the interaction_prediction directory:
```
cd interaction_prediction
```
### 1. Data Process
Preprocess data for model training using the following command:
```
python data_process.py \
--load_path path/to/your/dataset/scenario/set_path \
--save_path path/to/your/processed_data/set_path \
--use_multiprocessing \
--processes=8
```

Specify ```--load_path``` to the location of the downloaded set path, ```--save_path``` to the desired processed data path, and enable ```--use_multiprocessing``` for parallel data processing. You can perform this separately for the ```training``` and ```validation_interactive``` sets.

### 2. Training & Evaluation
Train the model using the command:
```
bash train.sh 4 #number of GPUs
```
**NOTE**: Before training, specify the processed paths for ```--train_set``` and ```--valid_set``` inside the script file. 
Set ```--name``` to save logs and checkpoints. As referred in ```train.py```, you can also adjust other arguments like ```--seed```, ```--train_epochs```, ```--batch_size``` for customed training.

## Open-loop Planning 
Navigate to the open_loop_planning directory:
```
cd open_loop_planning
```
### 1. Data Process
Preprocess data for model training using the following command:
```
python data_process.py \
--load_path path/to/your/dataset/training_20s \
--save_path path/to/your/processed_data \
--use_multiprocessing \
```

Set ```--load_path``` to the location of the downloaded dataset, ```--save_path``` to the desired processed data path, and enable ```--use_multiprocessing``` for parallel data processing. You can perform this separately for the training and validation sets.

### 2. Training
Train the model using the command:
```
python train.py \
--train_set path/to/your/processed_data/train \
--valid_set path/to/your/processed_data/valid
```

Specify the paths for ```--train_set``` and ```--valid_set```. You can set the ```--levels``` to determine the number of interaction levels. Adjust other parameters like ```--seed```, ```--train_epochs```, ```--batch_size```, and ```--learning_rate``` as needed for training.

The training log and models will be saved in ```training_log/{name}```.

### 3. Testing
For testing, run:
```
python open_loop_test.py \
--test_set path/to/your/dataset/validation \
--model_path path/to/your/saved/model
```

Specify ```--test_set``` as the path to the ```scenario/validation``` data, and ```--model_path``` as the path to your trained model. Use ```--render``` to visualize planning and prediction results.

The testing result will be saved in ```testing_log/{name}```.

## Citation
If you find this repository useful for your research, please consider giving us a star &#127775; and citing our paper.

```angular2html
@InProceedings{Huang_2023_ICCV,
    author    = {Huang, Zhiyu and Liu, Haochen and Lv, Chen},
    title     = {GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3903-3913}
}
```
