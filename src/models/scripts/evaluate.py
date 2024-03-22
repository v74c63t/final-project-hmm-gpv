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

from src.dataset.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.dataset.preprocessing.subtile_esd_hw02 import Subtile
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
    results_dir: str | os.PathLike = root / 'data/predictions' / "U2Net"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "model" / "U2Net" / "last.ckpt"



def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Load datamodule
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
            RichModelSummary(max_depth=2)
        ]
        # ,gpus=1 if torch.cuda.is_available() else 0  # Assuming availability of GPU
    )
    # run the validation loop with trainer.validate
    trainer.validate(model, datamodule=datamodule) 
    # run restitch_and_plot
    restitch_and_plot(options, datamodule, model, "Tile1", rgb_bands=[4,3,2], image_dir=root/"plots")
    processed_val_dir = Path(options.processed_dir) / "Val" / "subtiles"
    tiles = [tile for tile in processed_val_dir.iterdir()] # if tile.is_dir()
    # for every subtile in options.processed_dir/Val/subtiles
    processed_subtiles = set([os.path.basename(subtile).split("_")[0] for subtile in tiles]) 

    for subtile_parent_id in processed_subtiles: #tiles:
        restitch_and_plot(options, datamodule=datamodule, model=model, parent_tile_id=subtile_parent_id,image_dir=options.results_dir)
    

if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))
