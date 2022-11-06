# -*- coding: utf-8 -*-
"""
Author: Elias Mindlberger
    The classes below are required for loading and storing data
    and feeding data in a fitting shape to a CNN model.
"""
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import os
from random import randint
import torch

device = "cuda:0"


class Images(Dataset):
    """
    Class for Image-Dataset initialization.
    """

    def __init__(self, image_directory: str, normalize: bool = True, crop: bool = True, im_shape: int = 100):
        """
            :param image_directory: A path containing .jpg-Images.
            :param normalize: Boolean-value to check whether data shall be normalized or not. Default=True
        """
        self.images = glob.glob(os.path.join(image_directory, "**", "*.jpg"), recursive=True)
        self.normalize = normalize
        self.crop = crop
        self.im_shape = im_shape
        self.resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape),
            transforms.CenterCrop(size=(im_shape, im_shape))
        ])

    def __getitem__(self, idx) -> np.array:
        """
            :param idx: Index of the image to be queried
            :return: the image as Numpy Float Array of shape (3, 100, 100)
        """
        # Open Image as PIL Image / crop if specified
        image = self._crop_image(idx, self.im_shape) if self.crop else Image.open(self.images[idx])
        return image, idx

    def __len__(self) -> int:
        """
        :return: Number of images in dataset
        """
        return len(self.images)

    def _crop_image(self, idx: int, im_shape: int = 100):
        """
            :param idx: Index of the image to be cropped
            :param im_shape: Output-height and -width the image should be cropped to
            :return: returns cropped PIL-Image of size (im_shape, im_shape)
        """
        # Open image
        image = Image.open(self.images[idx])
        # Resize and center-crop image
        return self.resize_transforms(image)


class TransformedImages(Dataset):
    """
    Class for Image-Dataset transformation.
    """
    def __init__(self, images: Dataset, transforms_chain: transforms.Compose = None,
                 offset: tuple = None, spacing: tuple = None, normalize: bool = True):
        self.images = images
        self.offset = offset
        self.spacing = spacing
        self.transforms_chain = transforms_chain
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get a random offset and spacing if none is defined
        if self.spacing is None: spacing = (randint(2, 6), randint(2, 6))
        if self.offset is None: offset = (randint(0, 8), randint(0, 8))

        # get image as PIL-Image
        image, idx = self.images[idx]

        # execute transforms if specified
        if self.transforms_chain: image = self.transforms_chain(image)

        # Convert to Numpy and transpose to right shape
        image = np.array(image, dtype=np.float32) if self.normalize else np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))

        # create mask-array
        known_array = np.zeros_like(image)
        known_array[:, offset[1]:len(known_array[0]):spacing[1], offset[0]:len(known_array[0][0]):spacing[0]] = 1

        # multiply image with mask element-wise -> returns 1*pixel if mask is True else 0
        input_array = image * known_array

        # use mask to return the 1D-Array of R,G,B values which are off-grid
        target_array = image[known_array < 1]

        return input_array, known_array, target_array, image, idx
