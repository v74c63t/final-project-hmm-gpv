import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
import torch
@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions' / "FCNResnetTransfer"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "FCNResnetTransfer" / "last.ckpt"



def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # raise NotImplementedError # Complete this function using the code snippets below. Do not forget to remove this line.
    # Load datamodule
#     datamodule = ESDDataModule(
#         data_dir=options.processed_dir,
#         batch_size=options.batch_size,
#         num_workers=options.num_workers,
#         tile_size=options.tile_size_gt
#     )
#     datamodule.setup()
#     # load model from checkpoint at options.model_path
#     model = ESDSegmentation.load_from_checkpoint(checkpoint_path=str(options.model_path))
#     # set the model to evaluation mode (model.eval())
#     model.eval()
#     # this is important because if you don't do this, some layers
#     # will not evaluate properly

#     # instantiate pytorch lightning trainer
#     trainer = pl.Trainer(
#         callbacks=[
#             LearningRateMonitor(logging_interval='step'),
#             ModelCheckpoint(dirpath=options.results_dir, save_top_k=1, monitor="val_loss"),
#             RichProgressBar(),
#             RichModelSummary(max_depth=2)
#         ],
#         gpus=1 if torch.cuda.is_available() else 0  # Assuming availability of GPU
#     )
#     # run the validation loop with trainer.validate
#     trainer.validate(model, datamodule=datamodule)
#     # run restitch_and_plot
#     processed_val_dir = Path(options.processed_dir) / "Val" / "subtiles"
#     tiles = [tile for tile in processed_val_dir.iterdir() if tile.is_dir()]
#     # for every subtile in options.processed_dir/Val/subtiles
#     # run restitch_eval on that tile followed by picking the best scoring class
#     # save the file as a tiff using tifffile
#     # save the file as a png using matplotlib
#    # tiles = ...
#     for parent_tile_id in tiles:

#         # freebie: plots the predicted image as a jpeg with the correct colors
#         cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
#         fig, ax = plt.subplots(nrows=1, ncols=1)
#         ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap)
#         plt.savefig(options.results_dir / f"{parent_tile_id}.png")
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
        batch_size=options.batch_size,
        # num_workers=options.num_workers,
        tile_size_gt=options.tile_size_gt,
    )
    datamodule.prepare_data()
    datamodule.setup("val")
    # load model from checkpoint at options.model_path
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path=str(options.model_path))
    # set the model to evaluation mode (model.eval())
    model.eval()
    # this is important because if you don't do this, some layers
    # will not evaluate properly

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(dirpath=options.results_dir, save_top_k=1, monitor="val_loss"),
            # RichProgressBar(), #Need to test
            RichModelSummary(max_depth=2)
        ]
        # ,gpus=1 if torch.cuda.is_available() else 0  # Assuming availability of GPU
    )
    # run the validation loop with trainer.validate
    trainer.validate(model, datamodule=datamodule)  # UNCOMMENT
    # run restitch_and_plot #something about hardcoding to a single tile and moving on in HW3 evaluate.py Comments
    # ??
    restitch_and_plot(options, datamodule, model, "Tile1", rgb_bands=[4,3,2], image_dir=root/"plots")
    processed_val_dir = Path(options.processed_dir) / "Val" / "subtiles"
    tiles = [tile for tile in processed_val_dir.iterdir()] # if tile.is_dir()]
    # for every subtile in options.processed_dir/Val/subtiles
    processed_subtiles = set([os.path.basename(subtile).split("_")[0] for subtile in tiles]) # I made a set for the sake of restitch_eval and/or restitch_and_plot's parameter of parent_tile_id -Michael
    print(options.processed_dir)
    print(processed_val_dir)
    for subtile_parent_id in processed_subtiles: #tiles:
        # print(subtile)
        restitch_and_plot(options, datamodule=datamodule, model=model, parent_tile_id=subtile_parent_id,image_dir=options.results_dir)
        
    # run restitch_eval on that tile followed by picking the best scoring class
            # subtile_name = os.path.basename(subtile).split("_")[0] #
            # restitch_eval(processed_val_dir, "sentinel2", subtile_parent_id, (0,4),(0,4),datamodule=datamodule, model=model)
            # restitch_and_plot(options, datamodule=datamodule, model=model, parent_tile_id=subtile_parent_id,image_dir=root/"plots" )
        
            # print(subtile_name) #C:\Users\micha\Documents\UCI_W24\CS_175\hw03-segmentation-hmm-gpv\data\processed\4x4\Val\subtiles\Tile18_0_0.npz --> example
    # save the file as a tiff using tifffile tifffile.imwrite()?
    # save the file as a png using matplotlib plt.savefig() or smth
   # tiles = ...

 # ---------------------------------------------------- UnComment Later --------------------------           
    # for parent_tile_id in tiles:

    #     # freebie: plots the predicted image as a jpeg with the correct colors
    #     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
    #     fig, ax = plt.subplots(nrows=1, ncols=1)
    #     ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap) #where do we even get y_pred
    #     plt.savefig(options.results_dir / f"{parent_tile_id}.png")
    

if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))


''' Notes:
- I heard hw03 just needs one restitch image? but not too sure. something along the lines of running all 60 on the actual new model we try for ourselves
'''