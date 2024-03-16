""" This module contains functions for plotting satellite images. """
import os
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
# from ..preprocessing.file_utils import (load_satellite)
from dataset.preprocessing.file_utils import Metadata
from dataset.preprocessing.preprocess_sat import minmax_scale
from dataset.preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)


def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None: #DONE
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here

    #y log scale -lithium advice
    # for tile in range(viirs_stack.shape[0]):
    #     viirs_stack[tile] = preprocess_viirs(viirs_stack[tile])
    data_1D = viirs_stack.ravel()
    plt.hist(data_1D, bins = n_bins, log=True)
    plt.title("VIIRS Histogram")
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None: #TODO Later
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """

    # Log base if needed
    # fill in the code here
    bandLabels = metadata[0][0].bands
    # fig, axes = plt.subplots(sentinel2_stack.shape[2]//2,2, figsize = (12,4))
    fig, axes = plt.subplots(2,sentinel1_stack.shape[2], figsize = (48, 24))
    # print(f'S2 Histogram shape: {sentinel2_stack.shape}')
    for bands in range(sentinel1_stack.shape[2]):
        subarrayBand = sentinel1_stack[:,:,bands,:,:]
        # print(subarrayBand.shape)
        data_1D = subarrayBand.ravel()
        axes[0][bands].hist(data_1D, bins = n_bins, log=True)
        axes[0][bands].set_title(f'Band: {bandLabels[bands]}, Log Scale')
        axes[1][bands].hist(data_1D, bins = n_bins, log=False)
        axes[1][bands].set_title(f'Band: {bandLabels[bands]}, Linear Scale')
    # data_1D = sentinel2_stack.ravel()
    # plt.hist(data_1D, bins = n_bins, log=True)
    # plt.title("Sentinel 2 Histograms")
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None: #DONE
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here
    bandLabels = metadata[0][0].bands
    # fig, axes = plt.subplots(sentinel2_stack.shape[2]//2,2, figsize = (12,4))
    fig, axes = plt.subplots(1,sentinel2_stack.shape[2], figsize = (sentinel2_stack.shape[2]*5, 4))
    # print(f'S2 Histogram shape: {sentinel2_stack.shape}')
    for bands in range(sentinel2_stack.shape[2]):
        subarrayBand = sentinel2_stack[:,:,bands,:,:]
        # print(subarrayBand.shape)
        data_1D = subarrayBand.ravel()
        axes[bands].hist(data_1D, bins = n_bins, log=True)
        axes[bands].set_title(f'Band: {bandLabels[bands]}')
    # data_1D = sentinel2_stack.ravel()
    # plt.hist(data_1D, bins = n_bins, log=True)
    # plt.title("Sentinel 2 Histograms")
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None: #TODO
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here
    bandLabels = metadata[0][0].bands
    # fig, axes = plt.subplots(sentinel2_stack.shape[2]//2,2, figsize = (12,4))
    fig, axes = plt.subplots(1,landsat_stack.shape[2], figsize = (landsat_stack.shape[2]*5, 4))
    # print(f'S2 Histogram shape: {sentinel2_stack.shape}')
    for bands in range(landsat_stack.shape[2]):
        subarrayBand = landsat_stack[:,:,bands,:,:]
        # print(subarrayBand.shape)
        data_1D = subarrayBand.ravel()
        axes[bands].hist(data_1D, bins = n_bins, log=True)
        axes[bands].set_title(f'Band: {bandLabels[bands]}')
    # data_1D = sentinel2_stack.ravel()
    # plt.hist(data_1D, bins = n_bins, log=True)
    # plt.title("Sentinel 2 Histograms")
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None: #DONE
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    data_1D = ground_truth.ravel()
    plt.hist(data_1D, log=True)
    plt.title("Ground Truth Histogram")
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None: #DONE
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    if image_dir is None:
        plt.imshow(viirs)
        plt.title(plot_title)
        plt.show()
    else:
        plt.imshow(viirs)
        plt.title(plot_title)
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()
    return
    raise NotImplementedError


def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None: #DONE
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    fig, axes = plt.subplots(1, viirs_stack.shape[0], figsize=(48,24))
    for timestamp in range(viirs_stack.shape[0]):
        axes[timestamp].imshow(viirs_stack[timestamp][0])
        axes[timestamp].set_title(f'{metadata[timestamp].time}')

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()

    return
    raise NotImplementedError


def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray: #DONE
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """

    if satellite_type == 'viirs':
        return preprocess_viirs(satellite_stack)
    elif satellite_type == 'sentinel1':
        return preprocess_sentinel1(satellite_stack)
    elif satellite_type == 'sentinel2':
        return preprocess_sentinel2(satellite_stack)
    else:
        return preprocess_landsat(satellite_stack)
    raise NotImplementedError


def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None: #TODO, should look like splatoon colors
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    '''
    Notes:
    - we're only given 2 channels from s1, so we need to make VV - VH
    - It looks like it's one set of metadata, since we're working with 1 tile (1 location over Africa)
    '''

    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")

    # fill in the code here

    # print(processed_stack.shape) #(4,2,800,800)
    # print(len(metadata)) #(4)
    fig, axes = plt.subplots(processed_stack.shape[0], 1, figsize = (48,24))

    for timestamps in range(processed_stack.shape[0]):
        for lstOfbands in bands_to_plot:
            lstOfImgsToStack = []
            for bandIndex in range(processed_stack.shape[1]):

                lstOfImgsToStack.append(processed_stack[timestamps][bandIndex])
            VVminusVH = lstOfImgsToStack[1]-lstOfImgsToStack[0]
            minimum = np.min(VVminusVH, axis=(-2,-1), keepdims=True)
            maximum = np.max(VVminusVH, axis=(-2,-1), keepdims=True)

            minmaxed = (VVminusVH - minimum)/(maximum - minimum)
            # print(minmaxed.shape)
            lstOfImgsToStack[0], lstOfImgsToStack[1] = lstOfImgsToStack[1], lstOfImgsToStack[0]
            lstOfImgsToStack.append(minmaxed)
            
            axes[timestamps].imshow(np.stack(lstOfImgsToStack, axis=2))
            axes[timestamps].set_title(f'Time: {metadata[timestamps].time}')
            #     axes[timestamps].imshow(processed_stack[timestamps][bandIndex], alpha=0.5)
            #     # print(bands_to_plot[bandIndex])
            #     if lstOfbands[bandIndex] == 'VH':
            #         VHImg = processed_stack[timestamps][bandIndex]
            #     else:
            #         VVImg = processed_stack[timestamps][bandIndex]
            
            # minimum = np.min(VVImg-VHImg, axis=(-2,-1), keepdims=True)
            # maximum = np.max(VVImg-VHImg, axis=(-2,-1), keepdims=True)

            # minmaxed = (VVImg-VHImg - minimum)/(maximum - minimum)
            # axes[timestamps].imshow(minmaxed, alpha=0.5)
            # axes[timestamps].set_title(f'Time: {metadata[timestamps].time}')
        #plotted VH and VV, now need VV - VH
        
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()
    return 
    raise NotImplementedError



def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None: #DONE
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    # print("validate band id")
    # print(bands_to_plot)
    for lstOfBands in bands_to_plot:
        for b in lstOfBands:
            assert b in band_mapping.keys()

    return
    
    raise NotImplementedError


def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ): #DONE
    """
    This function plots the satellite images.

    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    # print(processed_stack.shape)
    fig, axes = plt.subplots(processed_stack.shape[0], len(bands_to_plot), figsize=(48,24))
    for times in range(processed_stack.shape[0]):
        for lstOfBandsIndex in range(len(bands_to_plot)):
            needToStackByBand = []
            for index in range(len(bands_to_plot[lstOfBandsIndex])):
                # axes[times][lstOfBandsIndex].imshow(processed_stack[times][band_mapping[bands_to_plot[lstOfBandsIndex][index]]], alpha = 0.5)
                needToStackByBand.append(processed_stack[times][band_mapping[bands_to_plot[lstOfBandsIndex][index]]])
            axes[times][lstOfBandsIndex].imshow(np.stack(needToStackByBand, axis=2))
            axes[times][lstOfBandsIndex].set_title(f"Time: {metadata[times].time} & Bands: {bands_to_plot[lstOfBandsIndex]}")

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(
            Path(image_dir) / f"plot_{metadata[0].satellite_type}.png"
            )
        plt.close()
    return
    raise NotImplementedError


def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None: #TODO
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    
    processed_stack = preprocess_data(satellite_stack, satellite_type)

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        # print(all_band_ids)
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        # print(band_mapping)
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(
            processed_stack,
            bands_to_plot,
            band_mapping,
            metadata,
            image_dir
            )


def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]: #DONE
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[List[str]]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """
    listOfBands = []
    for m in metadata:
        listOfBands.append(m.bands)
    return listOfBands
    raise NotImplementedError


def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None: #DONE
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    # fill in the code here
    # print(ground_truth.shape)
    plt.imshow(ground_truth[0][0])
    plt.title(plot_title)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
    return
    raise NotImplementedError

# if __name__ == '__main__':
#     pathToViirs = "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\unit_test_data\\Tile1"
#     stackOfViirs = load_satellite(pathToViirs, 'viirs')
#     print(stackOfViirs)