import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from hw02_assist_files.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile, restitch
import matplotlib
import torch
import tifffile
def restitch_and_plot(options, datamodule, model, parent_tile_id, satellite_type="sentinel2", rgb_bands=[3,2,1], image_dir: None | str | os.PathLike = None):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """
    # raise NotImplementedError # Complete this function using the code snippets below. Do not forget to remove this line.
    X, y, xHat = restitch_eval(options.processed_dir, satellite_type,parent_tile_id, (0,4),(0,4),datamodule=datamodule, model=model )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)

    fig, axs = plt.subplots(nrows=1, ncols=3)

    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    # axs[i].imshow on the 1d images in order to have the correct 
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as 
    # `im`, i.e, im = axs[i].imshow
    title = ['RGB Satellite Image', 'Ground Truth', 'Model Prediction']
    images = [X, y, xHat]
    # for i in range(3):
    #     im = axs[i].imshow(images[i][0], cmap=cmap, vmin=-0.5, vmax=3.5)
    #     axs[i].set_title(title[i])
    
    print(X.shape) #time, bands, width, height
    X = X[0]
    print(X.shape)
    print(y.shape)
    # xHat = xHat.transpose(2,1,0)
    print(xHat.shape)

    # Plot for actual satellite image:
    X = np.dstack([X[3,:,:], X[2,:,:], X[1,:,:]])
    axs[0].imshow(X, vmin = -0.5, vmax = 3.5)
    # for band in reversed(rgb_bands):
    #     axs[0].imshow(X[band], vmin = -0.5, vmax = 3.5)
    axs[0].set_title(title[0])

    # Plot for ground truth:
    axs[1].imshow(y[0], cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[1].set_title(title[1])

    # Plot for prediction:
    im = axs[2].imshow(xHat[0], cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[2].set_title(title[2])
    
    # The following lines sets up the colorbar to the right of the images    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['Sttlmnts Wo Elec', 'No Sttlmnts Wo Elec', 'Sttlmnts W Elec', 'No Sttlmnts W Elec'])
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / f"{parent_tile_id}_restitched_visible_gt_prediction.png")
        tifffile.imwrite(Path(image_dir) / f"{parent_tile_id}_restitched_visible_gt_prediction.tif", xHat[0])
        plt.close()


def restitch_eval(dir: str | os.PathLike, satellite_type: str, tile_id: str, range_x: Tuple[int, int], range_y: Tuple[int, int], datamodule, model) -> np.ndarray:
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type, 
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image. It will also get the tiles from the datamodule, evaluate
    it with model, and stitch the ground truth and predictions together.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension
        datamodule: pytorch_lightning.LightningDataModule
            Datamodule with the dataset
        model: pytorch_lightning.LightningModule
            LightningModule that will be evaluated
    
    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        stitched_ground_truth: np.ndarray
            Stitched ground truth of shape (width, height)
        stitched_prediction_subtile: np.ndarray
            Stitched predictions of shape (width, height)
    """
    
    dir = Path(dir)
    satellite_subtile = []
    ground_truth_subtile = []
    predictions_subtile = []
    satellite_metadata_from_subtile = []
    for i in range(*range_x):
        satellite_subtile_row = []
        ground_truth_subtile_row = []
        predictions_subtile_row = []
        satellite_metadata_from_subtile_row = []
        for j in range(*range_y):
            # subtile = Subtile().load(dir / 'subtiles' / f"{tile_id}_{i}_{j}.npz") #might not work based on "dir" as it ends with '4x4' (do if statement of if dir/Train or dir/Val is None )
            # can also just make sure that we do datamodule.prepare_data() beforehands so that wont be a problem (i think it shouldn't be an issue since we have a processed directory)
            # raise NotImplementedError # Complete the function in the lines below using the codes
            # snippets provided. Do NOT forget to remove the raise NotImplementedError line.
            # find the tile in the datamodule

            train = datamodule.train_dataset
            val = datamodule.val_dataset# idk if we have .tiles, but I know dataset has __getitem__ so we can return to X,y, metadata = datamodule.train_dataset[index] or val
            # dataset init has self.tiles its a list of stuff so we can use index() i think well from the few ed stuff i skimmed
            # path = dir / 'subtiles' / f"{tile_id}_{i}_{j}.npz"
            train_path = dir / 'Train' / 'subtiles' / f"{tile_id}_{i}_{j}.npz" 
            val_path = dir / 'Val' / 'subtiles' / f"{tile_id}_{i}_{j}.npz"
            try:
                index = train.tiles.index(train_path)
                subtile = Subtile().load(train_path)
                X, y, metadata = train[index]
            except ValueError:
                index = val.tiles.index(val_path)
                subtile = Subtile().load(val_path)
                X, y, metadata = val[index]
            # if index == -1:
            #     index = val.tiles.index(val_path)
            #     subtile = Subtile().load(dir / 'Val' /'subtiles' / f"{tile_id}_{i}_{j}.npz")
            #     X, y, metadata = val[index]
            # else:
            #     subtile = Subtile().load(dir / 'Train'/'subtiles' / f"{tile_id}_{i}_{j}.npz")
            #     X, y, metadata = train[index]
            # evaluate the tile with the model
            # You need to add a dimension of size 1 at dim 0 so that 
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            # make sure that the tile is in GPU memory, i.e., X = X.cuda() 
            # X = X.unsqueeze(0).cuda() 
            X = X.unsqueeze(0)
            # print(X.type())
            predictions = model(X.float()) # (prediction_x_hat) determine if predicting on subtile or whole tile

            # convert y to numpy array
            y = y.detach().numpy() 
            # detach predictions from the gradient, move to cpu and convert to numpy
            predictions = torch.transpose(predictions, 1,0).squeeze().detach().cpu().numpy()
            # predictions = predictions.detach().cpu().numpy()

            ground_truth_subtile_row.append(y)
            predictions_subtile_row.append(predictions)
            satellite_subtile_row.append(subtile.satellite_stack[satellite_type])
            satellite_metadata_from_subtile_row.append(subtile.tile_metadata)
        ground_truth_subtile.append(np.concatenate(ground_truth_subtile_row, axis=-1))
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
        satellite_subtile.append(np.concatenate(satellite_subtile_row, axis=-1))
        satellite_metadata_from_subtile.append(satellite_metadata_from_subtile_row)
    return np.concatenate(satellite_subtile, axis=-2), np.concatenate(ground_truth_subtile, axis=-2), np.concatenate(predictions_subtile, axis=-2)