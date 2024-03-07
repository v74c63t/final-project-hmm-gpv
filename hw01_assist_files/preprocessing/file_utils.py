"""
This module contains functions for loading satellite data from a directory of
tiles.
"""
from pathlib import Path
from typing import Tuple, List, Set
import os
from itertools import groupby
import re
from dataclasses import dataclass
import tifffile
import numpy as np


@dataclass
class Metadata:
    """
    A class to store metadata about a stack of satellite files from the same date.
    The attributes are the following:
    satellite_type: one of "viirs", "sentinel1", "sentinel2", "landsat", or "gt"
    file_name: a list of the original filenames of the satellite's bands
    tile_id: name of the tile directory, i.e., "Tile1", "Tile2", etc
    bands: a list of the names of the bands with correspondence to the
    indexes of the stack object, i.e. ["VH", "VV"] for sentinel-1
    time: time of the observations
    """
    satellite_type: str
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str

def process_Tile_number(filename: str | os.PathLike) -> str:
    pattern = re.compile(r'(Tile\d{1,2})')
    match_obj = pattern.search(str(filename))

    return (match_obj.group(1))

def process_viirs_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    name_of_file = Path(filename).name
    pattern = re.compile(r'DNB_VNP46A1_A(\d{7})\.tif$')
    match_obj = pattern.search(str(name_of_file))

    return (match_obj.group(1), "0")

    


    raise NotImplementedError


def process_s1_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    # lst_of_filename_components = filename.split('_')
    # date_str = lst_of_filename_components[3]
    # lst_of_filename_components = lst_of_filename_components[4].split('.')
    # retTuple = (date_str, lst_of_filename_components[0])


    # return (retTuple)

    name_of_file = Path(filename).name
    pattern = re.compile(r'S1A_IW_GRDH_(\d{8})_([A-Za-z]+)\.tif$')
    match_obj = pattern.search(str(name_of_file))

    return (match_obj.group(1), match_obj.group(2))
    raise NotImplementedError


def process_s2_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)
    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "01")
    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.
    Returns
    -------
    Tuple[str, str]
    """
    name_of_file = Path(filename).name

    pattern = re.compile(r'L2A_(\d{8})_B(\d{2}|8A)\.tif$')
    match_obj = pattern.search(str(name_of_file))

    return (match_obj.group(1), match_obj.group(2))
    raise NotImplementedError


def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)
    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "B9")
    Example output: ("2020-08-30", "9")
    Parameters
    ----------
    filename : str
        The filename of the Landsat file.
    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    name_of_file = Path(filename).name

    pattern = re.compile(r'LC08_L1TP_(\d{4}-\d{2}-\d{2})_B(\d{1,2})\.tif$')
    match_obj = pattern.search(str(name_of_file))

    return (match_obj.group(1), match_obj.group(2))

    raise NotImplementedError


def process_ground_truth_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return ("0","0")
    
    raise NotImplementedError


def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    """
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """
    if type(tile_dir) == str:
        tile_dir = Path(tile_dir)
    
    return list(tile_dir.glob(get_filename_pattern(satellite_type)))
    raise NotImplementedError


def get_filename_pattern(satellite_type: str) -> str:
    """
    Return the filename pattern for the given satellite type.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    str
        The filename pattern for the given satellite type.
    """
    if satellite_type == 'viirs':
        return 'DNB_VNP46A1_*'
    elif satellite_type == 'sentinel1':
        return "S1A_IW_GRDH_*"
    elif satellite_type == 'sentinel2':
        return 'L2A_*'
    elif satellite_type == "landsat":
        return 'LC08_L1TP_*'
    elif satellite_type == "gt":
        return 'groundTruth.tif'
    
    # raise NotImplementedError


def read_satellite_files(sat_files: List[Path]) -> List[np.ndarray]:
    """
    Read satellite files into a list of numpy arrays.

    Parameters
    ----------
    sat_files : List[Path]
        A list of Path objects for each satellite file.

    Returns
    -------
    List[np.ndarray]

    """
    satArray = []
    for f in sat_files:
        satArray.append(np.array(tifffile.imread(f), dtype=np.float32))
    return satArray
    
    raise NotImplementedError


def stack_satellite_data(
        sat_data: List[np.ndarray],
        file_names: List[str],
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Stack satellite data into a single array and collect filenames.
    Parameters
    ----------
    sat_data : List[np.ndarray]
        A list containing the image data for all bands with respect to
        a single satellite (sentinel-1, sentinel-2, landsat-8, or viirs)
        at a specific timestamp.
    file_names : List[str]
        A list of filenames corresponding to the satellite and timestamp.
    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with dimensions
        (time, band, height, width) and a list of the filenames.
    """
    # Get the function to group the data based on the satellite type GET_GROUPING_FUNCTION?

    # Apply the grouping function to each file name to get the date and band (likely process_<satellite>_filename)

    # Sort the satellite data and file names based on the date and band

    # Initialize lists to store the stacked data and metadata

    # Group the data by date
        # Sort the group by band

        # Extract the date and band, satellite data, and file names from the
        # sorted group

        # Stack the satellite data along a new axis and append it to the list

        # Create a Metadata object and append it to the list

    # Stack the list of satellite data arrays along a new axis to create a
    # 4D array with dimensions (time, band, height, width)

    # Return the stacked satelliet data and the list Metadata objects.

    '''
    sat_data: a list of 2d images read from .tif files (use read_satellite files)
    file_names: read satellite data from file name, where the image will be in sat_data
    satellite_type: the type of satellite used

    notes:
    first return value: the 4D thing in discord, then list
    each time has a band and image

    file_name will contatin time and band
    for 1 timestamp, can have multiple band (so might need a helping grouping function)
    a stack consists of the same image w/all different bands at the same timestamp

    file_name order should match sat_data order

    first return value should look like ex: (2,2,800,800) = (time, band, height, width)
    example: Tile 1 sentinel-1: (4 (timestamps), 2(bands), image size (height), image size (width))
    '''
    data = []
    fileNameProcessor = get_grouping_function(satellite_type) #To get dates and bands based on filenames
    for f in range(len(file_names)):
        dateNband = fileNameProcessor(str(file_names[f])) #kept getting some error regarding getting a WindowsFilePath versus a string

        dataTuple = (dateNband[0], dateNband[1], file_names[f], sat_data[f])
        data.append(dataTuple)

    sortedData = sorted(data, key = lambda x:(x[0], x[1]))
    grouped_data = groupby(sortedData, key=lambda x:x[0])

    npArrayToOverallStack = [] # the stacked nparray that will be stacked to have dimensions (timestamps, bands, height, width) (stackedByTime is the return value)

    tile = process_Tile_number(str(file_names[0])) # to get the tileID kept getting some error regarding getting a WindowsFilePath versus a string

    listOfMetadata = [] # list of the Metadata (return value)

    for key, group in grouped_data:
        # print(f"Key: {key}, Group: {list(group)}")
        groupAsList = list(group) # format of group == List[(date, band, filename, image as numpy array),...]
        nparrayToStack = []
        
        listOfBands = []
        listOfFilenames = []

        for item in groupAsList:
            nparrayToStack.append(item[3]) # append image
            listOfBands.append(item[1]) # append band
            listOfFilenames.append(item[2]) # append filename
        satellite_metadata = Metadata(satellite_type, listOfFilenames, tile, listOfBands, key)
        listOfMetadata.append(satellite_metadata)


        stackedByBand = np.stack(nparrayToStack, axis=0) # stacked the images by band

        npArrayToOverallStack.append(stackedByBand)

    stackedByTime = np.stack(npArrayToOverallStack, axis = 0) # stacked by time
            

    return (stackedByTime, listOfMetadata)

    '''
    Remember to edit process_<satellite>_filenames to regex
    
    '''
    raise NotImplementedError


def get_grouping_function(satellite_type: str): #Helper for stack_satellite_data
    """
    Return the function to group satellite files by date and band.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    function
        The function to group satellite files by date and band.
    """
    if satellite_type == 'viirs':
        return process_viirs_filename
    elif satellite_type == 'sentinel1':
        return process_s1_filename
    elif satellite_type == 'sentinel2':
        return process_s2_filename
    elif satellite_type == 'landsat':
        return process_landsat_filename
    return process_ground_truth_filename
    raise NotImplementedError


def get_unique_dates_and_bands(
        metadata_keys: Set[Tuple[str, str]]
        ) -> Tuple[Set[str], Set[str]]:
    """
    Extract unique dates and bands from satellite metadata keys.

    Parameters
    ----------
    metadata_keys : Set[Tuple[str, str]]
        A set of tuples containing the date and band for each satellite file.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        A tuple containing the unique dates and bands.
    """

    '''
    Notes:

    '''
    dateSet = set()
    bandSet = set()
    for dateNband in metadata_keys:
        dateSet.add(dateNband[0])
        bandSet.add(dateNband[1])

    return (dateSet, bandSet)
    raise NotImplementedError


def load_satellite(
        tile_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of tile files.
    Parameters
    ----------
    tile_dir : str or os.PathLike
        The Tile directory containing the satellite tiff files.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"
    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (time, band, height, width) and a list of the filenames.
    """

    '''
    Notes:
    - likely going to use stack_satellite_data <- input as get_satellite_files and read_satellite_files
    - 
    '''
    listOfSatelliteFiles = get_satellite_files(tile_dir, satellite_type) #To get the satellite files somewhere
    arrayOfSatelliteImages = read_satellite_files(listOfSatelliteFiles) # to turn those satellite files into np.arrays
    return stack_satellite_data(arrayOfSatelliteImages, listOfSatelliteFiles, satellite_type)
    raise NotImplementedError

def getTileNumber(filename: str | os.PathLike) -> int:
    '''
    Takes in a filename, find the tile number, and returns as an integer.
    '''
    tileStr = process_Tile_number(filename)
    pattern = re.compile(r'(\d{1,2})')
    match_obj = pattern.search(tileStr)

    return (int(match_obj.group(1)))

def load_satellite_dir(
        data_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[List[Metadata]]]:
    """
    Load all bands for a given satellite type fhttps://drive.google.com/file/d/FILE_ID/view?usp=sharing
rom a directory of multiple
    tile files.
    Parameters
    ----------
    data_dir : str or os.PathLike
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"
    Returns
    -------
    Tuple[np.ndarray, List[List[Metadata]]]
        A tuple containing the satellite data as a volume with
        dimensions (tile_dir, time, band, height, width) and a list of the
        Metadata objects.
    """
    # print(data_dir)
    if type(data_dir) == str:
        data_dir = Path(data_dir)
    stackByTilesArray = [] # The array we use to prep our 4D np.arrays to stack by tiles, introducing a 5th dimension
    listOfListsOfMetadata = [] # holds the lists of Metadata from stack_satellite_data
    dict_of_tiles_and_files = {} # To help in assigning and later iterating in order the data based on tile number
    biggestTileNum = 0 # also will help in iterating through the data in order of the tiles
    for file_path in data_dir.glob("*Tile*"):
        tileNum = getTileNumber(file_path)
        dict_of_tiles_and_files[tileNum] = file_path
        # print(file_path)
        if biggestTileNum < tileNum:
            biggestTileNum = tileNum
    # print(dict_of_tiles_and_files)
    for i in range(1, biggestTileNum+1):

        loadedData = load_satellite(dict_of_tiles_and_files[i], satellite_type)
        stackByTilesArray.append(loadedData[0])
        listOfListsOfMetadata.append(loadedData[1])

    # print(listOfListsOfMetadata)

    stackedByTiles = np.stack(stackByTilesArray, axis=0)
    return (stackedByTiles, listOfListsOfMetadata)
    raise NotImplementedError

# if __name__ == '__main__': #COMMENT OUT WHEN NOT TESTING LOCALLY
    # read_satellite_files
    # list_of_satellite_paths = [ "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\unit_test_data\\Tile1\\DNB_VNP46A1_A2020221.tif"]
    # result = read_satellite_files(list_of_satellite_paths)
    # print(result)

    # Stack_satellite_data
    # list_of_satellite_paths = [ "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\unit_test_data\\Tile1\\S1A_IW_GRDH_20200723_VH.tif",
    #                            "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\unit_test_data\\Tile1\\S1A_IW_GRDH_20200723_VV.tif",
    #                            "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\unit_test_data\\Tile1\\S1A_IW_GRDH_20200804_VH.tif",
    #                            "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\unit_test_data\\Tile1\\S1A_IW_GRDH_20200804_VV.tif"]
    # sat_data = [np.random.rand(800, 800)]*4

    # stack_satellite_data(sat_data, list_of_satellite_paths, 'sentinel1')

    # path = "C:\\Users\\micha\\Documents\\UCI_W24\\CS_175\\hw01-preprocessing-Wasabi-jpg\\data\\raw\\Train"
    # load_satellite_dir(Path(path), "sentinel1")
'''
Overall Notes:
- Tile: 1 area, multiple satellites take an image of said area with different bands with different times
- At a given time, multiple bands make up an image for that timestamp (unless DNB, then just 1 band for all timestamps)

- Channel: 
'''