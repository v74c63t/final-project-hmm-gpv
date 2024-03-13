# Final Project for HMM gpV #

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


