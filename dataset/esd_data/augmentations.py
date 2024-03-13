""" Place your augmentations.py code here"""

""" Augmentations Implemented as Callable Classes."""
import cv2
import numpy as np
import torch
import random
from typing import Dict

def apply_per_band(img, transform):
    """
    Helpful function to allow you to more easily implement
    transformations that are applied to each band separately.
    Not necessary to use, but can be helpful.
    """
    result = np.zeros_like(img)
    for band in range(img.shape[0]):
        transformed_band = transform(img[band].copy())
        result[band] = transformed_band

    return result


class Blur(object):
    """
        Blurs each band separately using cv2.blur

        Parameters:
            kernel: Size of the blurring kernel
            in both x and y dimensions, used
            as the input of cv.blur

        This operation is only done to the X input array.
    """
    def __init__(self, kernel=3):
        self.kernel = kernel

    def __blur_method(self, x):
        # print("Blur shape:")
        # print(x.shape)
        return cv2.blur(x, (self.kernel, self.kernel))

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
            Performs the blur transformation.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        # sample must have X and y in a dictionary format
        if list(sample.keys()) != ['X', 'y']:
            raise ValueError("Sample must have X and y in a dictionary format")
        # dimensions of img: (t, bands, tile_height, tile_width)
        # blur X
        x_img = sample['X']
        # x_blur = []
        # print(f'X_img shape BEFORE BLUR: {x_img.shape}')
        x_blurred = apply_per_band(x_img, self.__blur_method)
        # for time in range(x_img.shape[0]):
        #     time_stack = []
        #     for band in range(x_img.shape[1]):
        #         blurred_band = cv2.blur(x_img[time][band], (self.kernel, self.kernel))
        #         print(f'x blurred_band shape: {blurred_band.shape}')
        #         time_stack.append(blurred_band)
        #     x_blur.append(time_stack)
        
        # blur y
        y_img = sample['y']
        # y_blur = []
        # print(f'y_img shape BEFORE BLUR: {y_img.shape}')
        y_blurred = apply_per_band(y_img, self.__blur_method)
        # for time in range(y_img.shape[0]):
        #     time_stack = []
        #     for band in range(y_img.shape[1]):
        #         blurred_band = cv2.blur(y_img[time][band], (self.kernel, self.kernel))
        #         time_stack.append(blurred_band)
        #     y_blur.append(time_stack)
        
        return {'X': x_blurred, 'y': np.array(y_blurred)}
        # return {'X': np.array(x_blur), 'y': np.array(y_blur)}


class AddNoise(object):
    """
        Adds random gaussian noise using np.random.normal.

        Parameters:
            mean: float
                Mean of the gaussian noise
            std_lim: float
                Maximum value of the standard deviation
    """
    def __init__(self, mean=0, std_lim=0.):
        self.mean = mean
        self.std_lim = std_lim

    def __call__(self, sample):
        """
            Performs the add noise transformation.
            A random standard deviation is first calculated using
            random.uniform to be between 0 and self.std_lim

            Random noise is then added to each pixel with
            mean self.mean and the standard deviation
            that was just calculated

            The resulting value is then clipped using
            numpy's clip function to be values between
            0 and 1.

            This operation is only done to the X array.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        # sample must have X and y in a dictionary format
        if list(sample.keys()) != ['X', 'y']:
            raise ValueError("Sample must have X and y in a dictionary format")
        x_img = sample['X']
        y_img = sample['y']

        # calculate random standard deviation
        rsd = random.uniform(0., self.std_lim)

        # add random noise to each pixel for x and y
        x_noise = x_img + np.random.normal(self.mean, rsd, x_img.shape)
        # y_noise = y_img + np.random.normal(self.mean, rsd, y_img.shape)

        # clip the noise images
        x_clip = np.clip(x_noise, 0, 1)
        # y_clip = np.clip(y_noise, 0, 1)

        return {'X': x_clip, 'y': y_img}


class RandomVFlip(object):
    """
        Randomly flips all bands vertically in an image with probability p.

        Parameters:
            p: probability of flipping image.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
            Performs the random flip transformation using cv.flip.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        # sample must have X and y in a dictionary format
        if list(sample.keys()) != ['X', 'y']:
            raise ValueError("Sample must have X and y in a dictionary format")
        x_img = sample['X']
        y_img = sample['y']

        # flip x
        x_flip = []
        for time in range(x_img.shape[0]):
            # TODO: not sure if self.p should be used inside the loop or outside the loop
            # print(f"Starting X shape BEFORE Vert flip: {x_img[time].shape}")
            if np.random.rand() < self.p:
                flip_allband = cv2.flip(x_img[time], 0)
                # print(f"CV2 VertiFlipped: {flip_allband.shape}")
            else:
                flip_allband = x_img[time]
                # print(f"NOT CV2 VertiFlipped: {flip_allband.shape}")
            x_flip.append(flip_allband)
        
        # for item in x_flip:
        #     print(f"X item shape: {item.shape}")

        # flip y
        y_flip = []
        for time in range(y_img.shape[0]):
            if np.random.rand() < self.p:
                flip_allband = cv2.flip(y_img[time], 0)
            else:
                flip_allband = y_img[time]
            y_flip.append(flip_allband)

        # for item in x_flip:
        #     print(f"X Vert item shape: {item.shape}")

        # for itemY in y_flip:
        #     print(f"y Vert item shape: {itemY.shape}")
        return {'X': np.stack(x_flip), 'y': np.stack(y_flip)}


class RandomHFlip(object):
    """
        Randomly flips all bands horizontally in an image with probability p.

        Parameters:
            p: probability of flipping image.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
            Performs the random flip transformation using cv.flip.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        # sample must have X and y in a dictionary format
        if list(sample.keys()) != ['X', 'y']:
            raise ValueError("Sample must have X and y in a dictionary format")
        x_img = sample['X']
        y_img = sample['y']

        # flip x
        x_flip = []
        for time in range(x_img.shape[0]):
            # print(f"Starting X shape BEFORE Hori flip: {x_img[time].shape}")
            if np.random.rand() < self.p:
                flip_allband = cv2.flip(x_img[time], 1)
                # print(f"CV2 HoriFlipped: {flip_allband.shape}")
            else:
                flip_allband = x_img[time]
                # print(f"NOT CV2 HoriFlipped: {flip_allband.shape}")
            x_flip.append(flip_allband)
        
        # flip y
        y_flip = []
        for time in range(y_img.shape[0]):
            # print(f"Starting Y shape BEFORE Horiflip: {y_img[time].shape}")
            if np.random.rand() < self.p:
                flip_allband = cv2.flip(y_img[time], 1)
                # print(f"CV2 HoriFlipped YYY: {flip_allband.shape}")
            else:
                flip_allband = y_img[time]
                # print(f"NOT CV2 HoriFlipped YYY: {flip_allband.shape}")
            y_flip.append(flip_allband)
                
        # for item in x_flip:
        #     print(f"X Hori item shape: {item.shape}")

        # for itemY in y_flip:
        #     print(f"y Hori item shape: {itemY.shape}")
        
        return {'X': np.stack(x_flip), 'y': np.stack(y_flip)}


class ToTensor(object):
    """
        Converts numpy.array to torch.tensor
    """
    def __call__(self, sample):
        """
            Transforms all numpy arrays to tensors

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, torch.Tensor]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        # sample must have X and y in a dictionary format
        if list(sample.keys()) != ['X', 'y']:
            raise ValueError("Sample must have X and y in a dictionary format")
        x_img = sample['X']
        y_img = sample['y']

        # convert to tensor
        x_convert = torch.from_numpy(x_img)
        y_convert = torch.from_numpy(y_img)

        return {'X': x_convert, 'y': y_convert}
    
