import sys
import random
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from augmentation import Augmentations
from augmentation_list import AugmentationList



def main():

    # Read main euv image as normalized grayscale
    euv_image = Image.open('./data/euv/fd/20120403_000036_aia.lev1_euv_12s_4k.jpg')
    euv_image = np.array(euv_image.convert('L')) / 255

    # Select random pickle image from dataset and normalize it
    pickle_paths = glob.glob('./data/euv/tiles/*.p')
    pickle_path = random.choice(pickle_paths)
    pickle_image = pickle.load(imfile := open(pickle_path, 'rb'))
    imfile.close()
    pickle_image = pickle_image.astype(float) / 255

    # Perform random set of augmentations on the image
    augmentation_list = AugmentationList('euv')
    augmentation = Augmentations(pickle_image, augmentation_list.randomize())
    aug_img, aug_title = augmentation.perform_augmentations()

    # Fast fourier for augmented image
    faug_img = np.fft.fftshift(np.fft.fft2(aug_img))
    faug_mgn_spect = 20*np.log(np.abs(faug_img))

    # Invert back to normal image
    ishift_img = np.fft.ifftshift(faug_img)
    ifft_img = np.fft.ifft2(ishift_img)
    abs_img = np.abs(ifft_img)

    # Show Augmented Image
    cv.imshow("idft:", np.uint8(abs_img))


if __name__ == '__main__':
    main()