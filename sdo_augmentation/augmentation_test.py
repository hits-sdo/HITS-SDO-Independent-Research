'''
# TODO revise module docstring
Team objective:
1. Create a list of augmentation functions for training data
2. Apply a set of augmentations selected in random order or by the user

Important details on the project:
1. The model takes in a fixed size image and outputs a vector representation
    of the image
2. The model uses the vector representation to return similar images

Plan for Mar 10:
1. Perform a combination of augmentations depicted by a target dictionary
2. Work on the documentation of the code
'''

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from augmentation_list import AugmentationList
from augmentation import Augmentations
from image_utils import read_image, stitch_adj_imgs

DATA_DIR = './sdo_augmentation/data/euv/tiles/'
len_ = len("tile_20230206_000634_1024_0171_")

EXISTING_FILES = []

for root, dir, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".p"):
            EXISTING_FILES.append(file)


class Tests_on_Augmentations(unittest.TestCase):
    """
    numerical and visual verfication of invidual augmentation methods
    and combinations
    """

    def setUp(self):
        """define augmentation object"""
        self.DATA_DIR = './sdo_augmentation/data/euv/tiles/'
        self.FILE_NAME = EXISTING_FILES[59]
        self.image = read_image(self.DATA_DIR + self.FILE_NAME, 'p')
        self.superimage = stitch_adj_imgs(self.DATA_DIR, self.FILE_NAME,
                                          EXISTING_FILES)
        self.augument_list = AugmentationList(instrument='euv')

        augment_list = self.augument_list.randomize()

        self.augmentations = Augmentations(self.superimage, augment_list)

    def test_rotate(self):
        """visually check if
        superimage helps filling voids after image rotation
        """
        image_ro = self.augmentations.rotate(self.superimage, rotation := 45)
        plt.subplot(1, 2, 1)
        plt.imshow(
            self.superimage[self.image.shape[0]:2*self.image.shape[0],
                            self.image.shape[0]:2*self.image.shape[0]],
            vmin=0,
            vmax=1)
        plt.title('original image')

        plt.subplot(1, 2, 2)
        plt.imshow(
            image_ro[self.image.shape[0]:2*self.image.shape[0],
                     self.image.shape[0]:2*self.image.shape[0]],
            vmin=0,
            vmax=1)

        plt.title(f'rotated image {rotation} deg')
        plt.show()

    def test_dim(self):
        """
        test image dimension
        """
        self.assertEqual(len(self.image.shape), 2)

    def test_resolution(self):
        """
        check if input and augmented images
        are of same size
        """
        image_tr = self.augmentations.translate(self.image, translate=(5, 5))
        self.assertEqual(self.image.shape, (64, 64))
        self.assertEqual(image_tr.shape, (64, 64))

    def test_range(self):
        """
        check that input is not a constant image
        """
        image = self.augmentations.brighten(self.image)
        self.assertNotEqual(np.max(image), np.min(self.image))

    def test_tile_start(self):
        """
        check that tile location is correctly identified
        """
        self.assertEqual(int(self.FILE_NAME[len_:len_+4]), 640)

    def test_adjacent_valid(self):
        """
        check if the superimage correcty
        combines the adjacent tiles
        """
        superImage = stitch_adj_imgs(self.DATA_DIR, self.FILE_NAME)
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(superImage, vmin=0, vmax=1)
        plt.title('superimage')
        plt.show()

    def test_flip(self):
        """
        check if the image flip works
        """
        image_tr = self.augmentations.v_flip(self.image)

        # ensuring same size
        self.assertEqual(image_tr.shape, self.image.shape)

        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(image_tr, vmin=0, vmax=1)
        plt.title('Vertically flipped image')
        plt.show()

    def test_blur(self):
        """
        check if an image is blur kernel size is adequate
        """
        image_tr = self.augmentations.blur(self.image, blur=(2, 2))
        # ensuring same size
        self.assertEqual(image_tr.shape, self.image.shape)

        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(image_tr, vmin=0, vmax=1)
        plt.title('blurred image')
        plt.show()

    def test_pole_flip(self):
        """
        test polarity flip for magnetograms
        """
        # flip and ensure different image
        image_tr = self.augmentations.p_flip(self.superimage)
        image_tr = image_tr[self.image.shape[0]:2*self.image.shape[0],
                            self.image.shape[0]:2*self.image.shape[0]]
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, vmin=0, vmax=1, cmap="gray")
        plt.title('original image')
        plt.subplot(1, 3, 2)
        plt.imshow(image_tr, vmin=0, vmax=1, cmap="gray")
        plt.title('Polarity flipped image')
        plt.subplot(1, 3, 3)
        plt.plot(image_tr[30, :], label='transformed image')
        plt.plot(self.image[30, :], label='original image')
        plt.plot([0, 63], [0.5, 0.5], '-k')
        plt.legend(frameon=False)
        plt.show()

    def test_zoom(self):
        """visually check zooming in and out"""
        z = 2
        image_tr = self.augmentations.zoom(self.image, zoom=z)
        s = image_tr.shape
        s1 = self.image.shape
        y1 = s[0]//2 - s1[0]//2
        y2 = s[0]//2 + s1[0]//2
        x1 = s[1]//2 - s1[1]//2
        x2 = s[1]//2 + s1[1]//2
        image_tr = image_tr[y1:y2, x1:x2]
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(image_tr, vmin=0, vmax=1)
        if z < 1:
            plt.title('zoomed out image')
        else:
            plt.title('zoomed in image')
        plt.show()

    def test_augmentations(self):
        """
        check if combination of augmentations work as expected
        """
        # applied on superimage
        augmented_img, title = self.augmentations.perform_augmentations()

        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(augmented_img[self.image.shape[0]:
                                 2*self.image.shape[0],
                                 self.image.shape[0]:
                                 2*self.image.shape[0]], vmin=0, vmax=1)
        plt.title(title)
        plt.show()

        self.assertNotEqual(np.max(augmented_img), np.min(self.image))


if __name__ == '__main__':
    unittest.main()
