#!/bin/python3

from skimage import io
from skimage.color import rgb2gray
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import exposure

palm = io.imread("palm.jpg")
image = rgb2gray(palm)
image_rescaled = rescale(image, 0.25, anti_aliasing=False)[200:800, 30:630]

print(image_rescaled)
cmap = plt.cm.gray


# Logarithmic
logarithmic_corrected = exposure.adjust_log(image_rescaled, 10)

image_corr = logarithmic_corrected

kwargs = {'sigmas': [1], 'mode': 'reflect'}

fig, axes = plt.subplots(2, 4)
for i, black_ridges in enumerate([1, 0]):
    for j, func in enumerate([meijering, sato, frangi, hessian]):
        kwargs['black_ridges'] = black_ridges
        result = func(image_corr, **kwargs)
        axes[i, j].imshow(result, cmap=cmap, aspect='auto')
        if i == 0:
            axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
                                  'Sato\ntubeness', 'Frangi\nvesselness',
                                  'Hessian\nvesselness'][j])
        if j == 0:
            axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
