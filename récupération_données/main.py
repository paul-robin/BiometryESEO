#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io, filters
from skimage.color import rgb2hed, hed2rgb, rgb2gray, gray2rgb, hsv2rgb, rgb2hsv

from skimage.exposure import rescale_intensity
from skimage import exposure
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import segmentation
from skimage import morphology

from skimage.filters import rank
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage.transform import probabilistic_hough_line


def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    hsv = rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return hsv2rgb(hsv)

def main():
    print("hello world")
    palm = io.imread("palm.jpg")
    img = rgb2gray(palm)
    img_rescaled = rescale(img, 0.25, anti_aliasing=False)[200:800, 30:630]

    rows, cols = img.shape

    noise = np.ones_like(img_rescaled) * 0.2 * (img_rescaled.max() - img_rescaled.min())
    rng = np.random.default_rng()
    noise[rng.random(size=noise.shape) > 0.5] *= -1

    img_noise = img_rescaled + noise
    img_const = img_rescaled + abs(noise)

    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img_rescaled, 4)

    img_rob = filters.roberts(logarithmic_corrected)
    img_sob = filters.sobel(logarithmic_corrected)

    edges = canny(1 - img_sob, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=1, line_length=2, line_gap=3)


    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=False, sharey=False)
    ax = axes.ravel()

    ax[0].imshow(palm)
    ax[0].set_title("Original image")

    ax[1].imshow(img_rescaled, cmap=plt.cm.gray)
    ax[1].set_title("grayscale")

    ax[2].imshow(edges)
    ax[2].set_title('canny')  # Note that there is no Eosin stain in this image

    ax[3].imshow(img_rescaled)
    ax[3].set_title("grayscale rgb")

    ax[4].imshow(img_noise, cmap=plt.cm.gray)
    ax[4].set_title("noisy")  # Note that there is no Eosin stain in this image

    ax[5].imshow(img_const, cmap=plt.cm.gray)
    ax[5].set_title("rebuilt")

    ax[6].imshow(rescale_intensity(1 - img_rob), cmap=plt.cm.gray)
    ax[6].set_title("robert")  # Note that there is no Eosin stain in this image

    ax[7].imshow(rescale_intensity(1 - img_sob), cmap=plt.cm.gray)
    ax[7].set_title("sobel")

    ax[8].imshow(rescale_intensity(1 - logarithmic_corrected), cmap=plt.cm.gray)
    ax[8].set_title("log")


    plt.show()




if __name__ == "__main__":main()
