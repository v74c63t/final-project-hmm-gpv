# Semantic Segmentation for U-Net vs. U<sup>2</sup>-Net #

## Team *HMM gpV* Members ##
- Michael Mui [@Wasabi-jpg](https://github.com/Wasabi-jpg)
- Michelle Sheu [@michellesheu](https://github.com/michellesheu)
- Vanessa Tang [@v74c63t](https://github.com/v74c63t)
- Yihong(Hazel) Yu [@HazelYuAhiru](https://github.com/HazelYuAhiru)

## Project Overview ##
Our project goes back to the default focus on this course, namely identification of locations in Africa that need electricity. To be able to identify those locations, we utilized sematic segmentation to accomplish this goal of ours, with our model theme based around the U-Net. We essentially compared the multi-class accuracy of the U-Net and the U<sup>2</sup>-Net to understand each of their performances on the satellite dataset. 

## ML Pipeline ##
<img width="1000" alt="Screenshot 2024-03-22 at 1 14 58 AM" src="https://github.com/cs175cv-w2024/final-project-hmm-gpv/assets/78942001/d487ba77-c642-432e-a3da-2b1b78c30dc6">

## Structure ##
- [Model Prediction Images](/model/prediction_images)
   - Prediction results from models
- [Data Folder](/data)
   - For users to place the dataset into (more information about this is in the [Setting up the Data](#setting-up-the-data) section below)
- [Model Folder](/model)
   - Description of our model results
   - It is also where the models are populated when they are created by [train.py](/src/models/scripts/train.py)
- [Dataset Src](/src/dataset)
   - Data managing/preprocessing code
- [Models Src](/src/models)
   - Click "[supervised](/src/models/supervised)" for model code
   - "[Scripts](/src/models/scripts)" is for training and evaluating models
- [Visualization](/src/visualization)
   - Code for making visualizations based on model functions

## Getting Started ##
### Setting up the Virtual Environment ###
1. Create the virtual environment:
   ```
   python3 -m venv esdenv
   ```
2. Activate the virtual environment:
   * On macOS and Linux:
     ```
     source esdenv/bin/activate
     ```
   * On Windows:
     ```
     .\esdenv\Scripts\activate
     ```
3. Install the required packages:

    ```
    pip install -r requirements.txt
    ```

4. To deactivate the virtual environment: 
   ```
   deactivate
   ```

### Setting up the Data ###

1. Download and compile all the data into a single folder called `Train`
2. Place the `Train` folder under the [data/raw](/data/raw) directory. The structure of the data should be `data/raw/Train/` with `Train` containing multiple Tile folders that each contain .tif files.

**Notes**
* The dataset we are using is the IEEE GRSS 2021 ESD dataset
* All the files assume that the data is placed under the [data/raw](/data/raw) directory so make sure to do that before running any files

### Using Our Trained Models ###
1. Navigate to the [model](/model) directory to find the links to each of our trained models in the README
2. Click on the Google Drive link to the model and download the .ckpt file
3. In the [model](/model) directory, create a folder for the model downloaded and place the .ckpt file in there
   - If the U-Net model is downloaded, create a folder called `UNet`
   - If one of the U<sup>2</sup>-Net models is downloaded, create a folder called `U2Net`

### Training Your Own Models ###

**U-Net**: 
```
python -m src.models.scripts.train --model_type=UNet --learning_rate=1e-5 --max_epochs=5 --depth=2 --embedding_size=32 --kernel_size=7 --scale_factor=50
```

**U<sup>2</sup>-Net (Playful Sweep 4 Variant)** : 

```
python -m src.models.scripts.train --model_type=U2Net --learning_rate=1e-4 --max_epochs=5 --depth=2 --embedding_size=128 --kernel_size=5 --scale_factor=50
```

**U<sup>2</sup>-Net (Restful Sweep 9 Variant)** : 
```
python -m src.models.scripts.train --model_type=U2Net --learning_rate=1e-4 --max_epochs=5 --depth=3 --embedding_size=32 --kernel_size=3 --scale_factor=25
```

**Notes**
* Before running training on any of the models, make sure the batch_size in [train.py](/src/models/scripts/train.py) is set to "2"
* A Weights and Biases account is required to record all the metrics from training the model
* When training is called, a folder named after the model will be created under the [model](/model) directory and the model itself will be placed in that folder

### Evaluating Models ###

**U-Net**: 
```
python -m src.models.scripts.evaluate --model_path=INSERT_PATH_TO_MODEL
```

**U<sup>2</sup>-Net (Playful Sweep 4 Variant)**: 
```
python -m src.models.scripts.evaluate --model_path=INSERT_PATH_TO_MODEL
```

**U<sup>2</sup>-Net (Restful Sweep 9 Variant)**: 
```
python -m src.models.scripts.evaluate --model_path=INSERT_PATH_TO_MODEL
```

**Notes**
* `model_path` refers to the absolute path to the model (the .ckpt file) created when [train.py](/src/models/scripts/train.py) is called
* When [evaluate.py](/src/models/scripts/evaluate.py) is called, the [plots](/plots) directory will populate with a plot of Tile 1's RGB satellite image, ground truth, and the model prediction
* Depending on the model being evaluated, the [prediction/U-Net](/data/predictions/UNet) or the [prediction/U2Net](predictions/U2Net) directory will also populate with similar plots for all the tiles in the Validation dataset

## Sources of Interests and Citations ##

- U<sup>2</sup>-Net model code (thanks to xuebinqin): https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py
- U<sup>2</sup>-Net model paper: https://arxiv.org/pdf/2005.09007v3.pdf


<details>
  <summary> Behind the Scenes Planning on what to do</summary>
   
## 3/6/24 Team Focus Check In ##
- Learnings from hw03
  - Adapting Dataset class and Datamodule to run train_test_split over parent images to ensure validation set subtiles can be restitched into a whole image
  - Given train_test_split train over training set using 3 models
    - Simplest model: Seg CNN
    - Med-level model: FCN Resnet (demos how to use pretrained weights and how to change architecture to suit image dimensions from our data)
    - Med++ level model: U Net, advantage being skip connections (from scratch)
    - Configure PyTorch Lightning Wrapper for model
  - Using Weights and Biases to configure hyperparameter search and see results using their website
  - Slurm (OPTIONAL), for people who don't have enough computing resources (GPUs) to train (be able to run code on openlab)
  - Run Evaluation to get images & performance metrics (validation loss, accuracy, jaccard index, etc.) and then run RESTITCH
 
### To-do for Hw04 (final project) ###
- Task: Run multi-class segmentation (same as hw03) using different model
- Things to adapt from hw03: (initial list, can change)
  - Create new model file (similar to SegCNN, FCN Resnet, etc.) to write model code
  - What model to invest in: (whatever has more tutorials / interested in) (present any results, good or bad) (Likely U^2 model, but need research on it to make sure it's different from hw03's UNET: https://paperswithcode.com/paper/u-2-net-going-deeper-with-nested-u-structure)
  - Change _init_ in pytorch lightning wrapper to be able to instantiate new model class and run training on it
  - Adapt script files to then train said new model
  - Validate said model
  - Restitch == results
  - get segmentation images
  - get performance information
  - Push comes to shove if we cannot find another model, just make frontend for hw03
 
### What we need to do ###
- ^ same ground truth, but might want to decide which satellites to use (ex: just sentinel - 2 and viirs, but have to customize in DATASET class)
  - Decide with team which satellite, which bands (need Viirs), and model

###To Do List: ###

- [ ] Check out U-Net squared (U^2)
- [ ] Build adapted version of U-Net squared
- [ ] How to include Slurm (now just need to go through the document Hazel provided)
- [ ] Find which data preprocessing functions from hw can be used for model(s) (it looks like we're just adapting hw 03 with a new model)
- [ ] Determine which satellites or whether we use the entire satellite dataset
- [ ] Determine what findings we want (what kinds of loss functions to use?, which tiles to showcase as a visual)
- [ ] How to present findings (Poster, show graphs)

## NOTHING ABOVE IS CONCRETE ##

## What we've done ##
- Hw 01, 02, 03

## What we want to do ##
- Overall, we want a plan to execute on for the final project so we have some direction (how much is hw03 worth for what we want?)
- Then, figure out the pull request
- Then, get help on hw03 (might move up in priority)
</details>
