""" place your datamodule code here. """
""" This module contains the PyTorch Lightning ESDDataModule to use with the
PyTorch ESD dataset."""

import pytorch_lightning as pl
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torch
from .dataset import DSE
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ..preprocessing.subtile_esd_hw02 import grid_slice
from dataset.preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_viirs,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
)
from dataset.preprocessing.file_utils import (
    load_satellite
)
from dataset.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor
)
from torchvision import transforms
from copy import deepcopy
from typing import List, Tuple, Dict
from dataset.preprocessing.file_utils import Metadata

from sklearn.model_selection import train_test_split
import pyprojroot

def collate_fn(batch):
    Xs = []
    ys = []
    metadatas = []
    for X, y, metadata in batch:
        Xs.append(X)
        ys.append(y)
        metadatas.append(metadata)
    
    return torch.stack(Xs), torch.stack(ys), metadatas


class ESDDataModule(pl.LightningDataModule):
    """
        PyTorch Lightning ESDDataModule to use with the PyTorch ESD dataset.

        Attributes:
            processed_dir: str | os.PathLike
                Location of the processed data
            raw_dir: str | os.PathLike
                Location of the raw data
            selected_bands: Dict[str, List[str]] | None
                Dictionary mapping satellite type to list of bands to select
            tile_size_gt: int
                Size of the ground truth tiles
            batch_size: int
                Batch size
            seed: int
                Seed for the random number generator
    """
    def __init__(
            self,
            processed_dir: str | os.PathLike,
            raw_dir: str | os.PathLike,
            selected_bands: Dict[str, List[str]] | None = None,
            tile_size_gt=4,
            batch_size=32,
            seed=12378921):

       self.processed_dir = processed_dir
       self.raw_dir = raw_dir
       self.selected_bands = selected_bands
       self.tile_size_gt = tile_size_gt
       self.batch_size = batch_size
       self.seed = seed
       self.transform = None

        # utilize the RandomApply transform to apply each of the transforms 
        # with a probability of 0.5
       random_apply_AddNoise = transforms.RandomApply([AddNoise()], p=0.5)
       random_apply_Blur = transforms.RandomApply([Blur()], p=0.5)
       random_apply_RandomHFlip = transforms.RandomApply([RandomHFlip()], p=0.5)
       random_apply_RandomVFlip = transforms.RandomApply([RandomVFlip()], p=0.5)

       # making composition
       self.transform = transforms.Compose([random_apply_AddNoise,random_apply_Blur, random_apply_RandomHFlip, random_apply_RandomVFlip, ToTensor()])
       self.prepare_data_per_node = False
       self.save_hyperparameters()
       self.allow_zero_length_dataloader_with_multiple_devices = False
    
    def __load_and_preprocess(
            self,
            tile_dir: str | os.PathLike,
            satellite_types: List[str] = ["viirs", "sentinel1", "sentinel2", "landsat", "gt"]
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Metadata]]]:
        """
            Performs the preprocessing step: for a given tile located in tile_dir,
            loads the tif files and preprocesses them just like in homework 1.

            Input:
                tile_dir: str | os.PathLike
                    Location of raw tile data
                satellite_types: List[str]
                    List of satellite types to process

            Output:
                satellite_stack: Dict[str, np.ndarray]
                    Dictionary mapping satellite_type -> (time, band, width, height) array
                satellite_metadata: Dict[str, List[Metadata]]
                    Metadata accompanying the statellite_stack
        """
        preprocess_functions = {
            "viirs": preprocess_viirs,
            "sentinel1": preprocess_sentinel1,
            "sentinel2": preprocess_sentinel2,
            "landsat": preprocess_landsat,
            "gt": lambda x: x
        }
        
        satellite_stack = {}
        satellite_metadata = {}
        for satellite_type in satellite_types:
            stack, metadata = load_satellite(tile_dir, satellite_type)

            stack = preprocess_functions[satellite_type](stack)

            satellite_stack[satellite_type] = stack
            satellite_metadata[satellite_type] = metadata

        satellite_stack["viirs_maxproj"] = np.expand_dims(maxprojection_viirs(satellite_stack["viirs"], clip_quantile=0.0), axis=0)
        satellite_metadata["viirs_maxproj"] = deepcopy(satellite_metadata["viirs"])
        for metadata in satellite_metadata["viirs_maxproj"]:
            metadata.satellite_type = "viirs_maxproj"

        return satellite_stack, satellite_metadata

    def prepare_data(self):
        """
            If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory)

            For each tile,
                - load and preprocess the data in the tile
                - grid slice the data
                - for each resulting subtile
                    - save the subtile data to self.processed_dir
        """
        # if the processed_dir does not exist, process the data and create
        # subtiles of the parent image to save
        if not os.path.exists(self.processed_dir): 
            parents = []
            # fetch all the parent images in the raw_dir
            # go to directory where parent images live
            # then save all file paths into list
            # divide said list to a val set (0.2) and train set (0.8) split
            for parent_dir, _, _ in os.walk(self.raw_dir): 
                parents.append(parent_dir)
           
            parentsTrain, parentsVal = train_test_split(parents[1:], test_size=0.2, random_state=self.seed)

            for parent_file in parentsTrain:
                stack, metadata = self.__load_and_preprocess(parent_file)

                subtiles = grid_slice(stack, metadata, self.tile_size_gt)

                for subtile in subtiles:
                    subtile.save(self.processed_dir / "Train")

            for parent_file in parentsVal:
                stack, metadata = self.__load_and_preprocess(parent_file)

                subtiles = grid_slice(stack, metadata, self.tile_size_gt)

                for subtile in subtiles:
                    subtile.save(self.processed_dir / "Val")

    def setup(self, stage: str):
        """
            Create self.train_dataset and self.val_dataset.0000ff

            Hint: Use torch.utils.data.random_split to split the Train
            directory loaded into the PyTorch dataset DSE into an 80% training
            and 20% validation set. Set the seed to 1024.
        """

        generator = Generator().manual_seed(1024)
        
        self.train_dataset = DSE(f'{self.processed_dir}/Train/subtiles', self.selected_bands, self.transform)
        self.val_dataset = DSE(f'{self.processed_dir}/Val/subtiles', self.selected_bands, self.transform)
            
    def train_dataloader(self):
        """
            Create and return a torch.utils.data.DataLoader with
            self.train_dataset
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    
    def val_dataloader(self):
        """
            Create and return a torch.utils.data.DataLoader with
            self.val_dataset
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)