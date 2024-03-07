""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""
import numpy as np
from scipy.ndimage import gaussian_filter
# import file_utils

faking_maxProjectionViirs = False
def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.
    The gaussian filter should be applied to each heightxwidth image individually.
    Parameters
    ----------
    img : np.ndarray
        The image to be filtered. The shape of the array is (time, band, height, width).
    sigma : float
        The sigma of the gaussian filter.
    Returns
    -------
    np.ndarray
        The filtered image. The shape of the array is (time, band, height, width).
    """
    '''
    Notes: 
    apply scipy's gaussian filter (library)
    '''
    # print(img.shape) # Using test cases, 2 times, 3 bands = 6 pictures
    # print("img")
    # print(img)
    # print(img[0].shape)
    # print("img[0]")
    # print(img[0])

    overallGaussianToStack = []
    for timestamps in range(img.shape[0]):
        bandGaussianToStack = []
        for bands in range(img.shape[1]):
            # print(str(timestamps) + "  --- " + str(bands))
            bandGaussianToStack.append(gaussian_filter(img[timestamps][bands], sigma))
        bandGaussianStacked = np.stack(bandGaussianToStack, axis=0)
        overallGaussianToStack.append(bandGaussianStacked)

    return np.stack(overallGaussianToStack, axis=0)
    # return gaussian_filter(img, sigma)
    raise NotImplementedError


def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  group_by_time=True
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.
    It calculates the top `clip_quantile` samples and the bottom `clip_quantile`
    samples, and sets any value above the top to the first value under the top value,
    and any value below the bottom to the first value above the top value.
    group_by_time affects how img_max and img_min are calculated, if
    group_by_time is true, the quantile limits are shared along the time dimension.
    Otherwise, the quantile limits are calculated individually for each image.
    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped. The shape of the array is (time, band, height, width).
    clip_quantile : float
        The quantile to clip the outliers by. Value between 0 and 0.5.
    Returns
    -------
    np.ndarray
        The clipped image stack. The shape of the array is (time, band, height, width).
    """
    '''
    Notes:
    some images have crazy values, so need to "clip" them within a reasonable range (check 1st discussion for info on clip)

    quantile function: if you input a value 'q' from 0 to 1, then quantile function returns a value that 100% - q% are values less than 
    k, while 100% - (100%-q%) are values greater than k. (returns q, whatever quantile it is)
    Trying to quantile q

    parameter group_by_time: [time, band, x, y], keep the time dimension, so when quantiling, if this param is true, just pass in last 2 parameters, otherwise, pass in first param of time, and last 2 params

    find quantile first, then clip
    '''
    if group_by_time:
        quantiledLower = np.quantile(img_stack, clip_quantile, axis=(-2,-1), keepdims=True)
        quantiledUpper = np.quantile(img_stack, 1- clip_quantile, axis=(-2,-1), keepdims=True)
        return np.clip(img_stack, quantiledLower, quantiledUpper)
    else:
        # listOfTimesToStack = []
        # for times in range(img_stack.shape[0]):
        #     listOfBandsToStack = []
        #     for bands in range(img_stack.shape[1]):
        #         quantiledLower = np.quantile(img_stack[times][bands], clip_quantile)
        #         quantiledUpper = np.quantile(img_stack[times][bands], 1-clip_quantile)
        #         img_clipped = np.clip(img_stack[times][bands], quantiledLower, quantiledUpper)
        #         listOfBandsToStack.append(img_clipped)
        #     stackedBands = np.stack(listOfBandsToStack, axis=0)
        #     listOfTimesToStack.append(stackedBands)
        # stackedTimes = np.stack(listOfTimesToStack, axis=0)
        # return stackedTimes
        quantiledLower = np.quantile(img_stack, clip_quantile, axis=(0,-2,-1), keepdims=True)
        quantiledUpper = np.quantile(img_stack, 1- clip_quantile, axis=(0,-2,-1), keepdims=True)
        return np.clip(img_stack, quantiledLower, quantiledUpper)
    raise NotImplementedError


def minmax_scale(img: np.ndarray, group_by_time=True): 
    """
    This function minmax scales the image stack to values between 0 and 1.
    This transforms any image to have a range between img_min to img_max
    to an image with the range 0 to 1, using the formula 
    (pixel_value - img_min)/(img_max - img_min).
    group_by_time affects how img_max and img_min are calculated, if
    group_by_time is true, the min and max are shared along the time dimension.
    Otherwise, the min and max are calculated individually for each image.
    
    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled. The shape of the array is (time, band, height, width).
    group_by_time : bool
        Whether to group by time or not.
    Returns
    -------
    np.ndarray
        The minmax scaled image stack. The shape of the array is (time, band, height, width).
    """

    '''
    Notes:
    check document from wednesday discussion
    group_by_time: same as above
    '''
    # print(f'Shape: {img.shape}')
    if group_by_time:
        minVal = np.min(img, axis=(-2,-1), keepdims=True)
        maxVal = np.max(img, axis=(-2,-1), keepdims=True)
        # listOfTimesToStack = []
        # for times in range(img.shape[0]):
        #     listOfBandsToStack = []
        #     for bands in range(img.shape[1]):
        #         img[times][bands] = np.divide(img[times][bands]-minVal, maxVal-minVal, out = img[times][bands], where=maxVal-minVal!=0)
        #         # for row in range(img.shape[2]):
        #         #     for pix in range(img.shape[3]):
        #         #         img[times][bands][row][pix] = (img[times][bands][row][pix] - minVal) / (maxVal - minVal)
        #         listOfBandsToStack.append(img[times][bands])
        #     stackedBands = np.stack(listOfBandsToStack, axis=0)
        #     listOfTimesToStack.append(stackedBands)
        # stackedTimes = np.stack(listOfTimesToStack, axis=0)
        # return stackedTimes
        return np.divide(img-minVal, maxVal-minVal)
    else:
        minVal = np.min(img, axis=(0,-2,-1), keepdims=True)
        maxVal = np.max(img, axis=(0,-2,-1), keepdims=True)
        # listOfTimesToStack = []
        # for times in range(img.shape[0]):
        #     listOfBandsToStack = []
        #     for bands in range(img.shape[1]):
        #         # minVal = np.min(img[times][bands])
        #         # maxVal = np.max(img[times][bands])
        #         img[times][bands] = np.divide(img[times][bands]-minVal, maxVal-minVal, out = img[times][bands], where=maxVal-minVal!=0)
        #         # for row in range(img.shape[2]):
        #         #     for pix in range(img.shape[3]):
        #         #         print(f'pixel:{img[times][bands][row][pix]}, minVal:{minVal}, maxVal:{maxVal}')
        #         #         img[times][bands][row][pix] = (img[times][bands][row][pix] - minVal) / (maxVal - minVal)
        #         # listOfBandsToStack.append(img[times][bands])
        #     # stackedBands = np.stack(listOfBandsToStack, axis=0)
        #     # listOfTimesToStack.append(stackedBands)
        # # stackedTimes = np.stack(listOfTimesToStack, axis=0)
        # # return img
        return np.divide((img-minVal), (maxVal-minVal))
    raise NotImplementedError


def brighten(img, alpha=0.13, beta=0):
    """
    This is calculated using the formula new_pixel = alpha*pixel+beta.
    If a value of new_pixel falls outside of the [0,1) range,
    the values are clipped to be 0 if the value is under 0 and 1 if the value is over 1.
    Parameters
    ----------
    img : np.ndarray
        The image to be brightened. The shape of the array is (time, band, height, width).
        The input values are between 0 and 1.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.
    Returns
    -------
    np.ndarray
        The brightened image. The shape of the array is (time, band, height, width).
    """
    for times in range(img.shape[0]):
        for bands in range(img.shape[1]):
            for row in range(img.shape[2]):
                for pix in range(img.shape[3]):
                    new_pixel = alpha*img[times][bands][row][pix]+beta
                    img[times][bands][row][pix] = new_pixel

    return np.clip(img, 0, 1)
    raise NotImplementedError


def gammacorr(band, gamma=2):
    """
    This function applies a gamma correction to the image.
    This is done using the formula pixel^(1/gamma)
    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected. The shape of the array is (time, band, height, width).
        The input values are between 0 and 1.
    gamma : float
        The gamma parameter of the gamma correction.
    Returns
    -------
    np.ndarray
        The gamma corrected image. The shape of the array is (time, band, height, width).
    """
    for times in range(band.shape[0]):
        for bands in range(band.shape[1]):
            for row in range(band.shape[2]):
                for pix in range(band.shape[3]):
                    new_pixel = band[times][bands][row][pix]**(1/gamma)
                    band[times][bands][row][pix] = new_pixel

    return band
    raise NotImplementedError


def maxprojection_viirs(
        viirs_stack: np.ndarray,
        clip_quantile: float = 0.01
        ) -> np.ndarray: 
    """
    This function takes a stack of VIIRS tiles and returns a single
    image that is the max projection of the tiles.
    The output value of the projection is such that 
    output[band,i,j] = max_time(input[time,band,i,j])
    i.e, the value of a pixel is the maximum value over all time steps.
    Parameters
    ----------
    tile_dir : str (WRONG)
        The directory containing the VIIRS tiles. The shape of the array is (time, band, height, width).
    Returns
    -------
    np.ndarray
        Max projection of the VIIRS stack, of shape (band, height, width)
    """
    # HINT: use the time dimension to perform the max projection over.
    # retImage = np.zeros(viirs_stack.shape[1],viirs_stack.shape[2],viirs_stack.shape[3])
    # clippedVIIRS_Stack = quantile_clip(viirs_stack, clip_quantile, False)
    # minMaxedVIIRS_Stack = minmax_scale(clippedVIIRS_Stack, False)
    # return np.max(minmax_scale(quantile_clip(viirs_stack, clip_quantile, False), False), axis=0)
    
    # tempQuant = np.quantile(a=viirs_stack, q=clip_quantile, keepdims=True)
    
    # if not faking_maxProjectionViirs:
    for times in range(viirs_stack.shape[0]):
        quantiledLower = np.quantile(viirs_stack[times], clip_quantile)
        quantiledUpper = np.quantile(viirs_stack[times], 1 - clip_quantile)
        viirs_stack[times] = np.clip(viirs_stack[times], quantiledLower, quantiledUpper)

    maxedViirs = np.max(viirs_stack, axis= 0)
    
    return minmax_scale(maxedViirs, False)
    # for times in range(viirs_stack.shape[0]):
    #     for bands in range(viirs_stack.shape[1]):
    #         quantiledLower = np.quantile(viirs_stack[times][bands], clip_quantile)
    #         quantiledUpper = np.quantile(viirs_stack[times][bands], 1-clip_quantile)
    #         viirs_stack[times][bands] = np.clip(viirs_stack[times][bands], quantiledLower, quantiledUpper)
    #         # viirs_stack[times][bands] = np.where(viirs_stack[times][bands] < quantiledLower, quantiledLower, viirs_stack[times][bands])
    #         # viirs_stack[times][bands] = np.where(viirs_stack[times][bands] > quantiledUpper, quantiledUpper, viirs_stack[times][bands])

    # # return np.max(minmax_scale(viirs_stack, False), axis = 0)
    # new_viirs_img = viirs_stack[0]
    # minmaxed = minmax_scale(viirs_stack, False)
    # for row in range(viirs_stack.shape[2]):
    #     for col in range(viirs_stack.shape[3]):
    #         currMax = float("-inf")
    #         for times in range(viirs_stack.shape[0]):
    #             if currMax < viirs_stack[times][0][row][col]:
    #                 currMax = viirs_stack[times][0][row][col]
    #         new_viirs_img[0][row][col] = currMax
    
    # return new_viirs_img


    raise NotImplementedError
    # else:
        # return viirs_stack[0]


def preprocess_sentinel1(
        sentinel1_stack: np.ndarray,
        clip_quantile: float = 0.01,
        sigma=1
        ) -> np.ndarray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
    """
    base10S1Stack = np.log10(sentinel1_stack)
    quantClipS1Stack = quantile_clip(base10S1Stack, clip_quantile)
    gaussian_filtered_S1 = per_band_gaussian_filter(quantClipS1Stack, sigma)
    return minmax_scale(gaussian_filtered_S1)
    raise NotImplementedError


def preprocess_sentinel2(sentinel2_stack: np.ndarray,
                         clip_quantile: float = 0.05,
                         gamma: float = 2.2
                         ) -> np.ndarray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    quantClipS2Stack = quantile_clip(sentinel2_stack, clip_quantile)
    gamma_corrected_S2 = gammacorr( quantClipS2Stack,gamma)
    return minmax_scale(gamma_corrected_S2)
    raise NotImplementedError


def preprocess_landsat(
        landsat_stack: np.ndarray,
        clip_quantile: float = 0.05,
        gamma: float = 2.2
        ) -> np.ndarray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    quantClipL8Stack = quantile_clip(landsat_stack, clip_quantile)
    gamma_corrected_L8 = gammacorr( quantClipL8Stack,gamma)
    return minmax_scale(gamma_corrected_L8)

    raise NotImplementedError


def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    quantClipViirs = quantile_clip(viirs_stack, clip_quantile)
    return minmax_scale(quantClipViirs)
    raise NotImplementedError

# if __name__ == '__main__':
'''
Q's

quantile clip, quantile() issue? told not to use?

maxprojection_viirs inquiry
'''