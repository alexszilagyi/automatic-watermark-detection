
import math
import os
import warnings

import cv2
import numpy
import numpy as np
import scipy
import scipy.fftpack

import matplotlib.pyplot as plt

import tensorflow as tf

from estimate_watermark import *
from watermark_reconstruct import *

from constants import constants

FOLDERNAME = 'D:/Images/TL;DR/BumbleBee/1_RW/25.3/'

# assert os.path.exists(FOLDERNAME), "Folder does not exist."

original = []
images = []
for r, dirs, files in os.walk(FOLDERNAME):
    # Get all the images
    for file in files:
        img = cv2.imread(os.sep.join([r, file]))
        if img is not None:
            
            height, width = img.shape[:2]
            watermarkSection = img[(height-constants.wmY):height, (width-constants.wmX):width]
            images.append(watermarkSection)
            original.append(img)
        else:
            print("%s not found." % (file))
images = np.array(images)
# images.shape


gx, gy = estimate_watermark(images)

est = poisson_reconstruct(gx, gy)

gx_manual_crop = gx
gy_manual_crop = gy
est_manual_crop = poisson_reconstruct(gx_manual_crop, gy_manual_crop)

cropped_gx, cropped_gy = crop_watermark(gx_manual_crop, gy_manual_crop)
est_auto_crop = poisson_reconstruct(cropped_gx, cropped_gy)

with open('cropped.npz', 'wb') as f:
    np.savez(f, cropped_gx=cropped_gx, cropped_gy=cropped_gy)


img = cv2.imread('D:/Images/TL;DR/BumbleBee/1_RW/25.3/25_3_001.jpg')
start, rect = watermark_detector(img, cropped_gx, cropped_gy)

im = img.copy()
cv2.rectangle(im, (start[1], start[0]), (start[1] + rect[1], start[0] + rect[0]), (255, 0, 0))

plt.figure(figsize=(12, 12), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(im)
plt.show()

# images_cropped = images[:, start[0]:start[0] + rect[0], start[1]:start[1] + rect[1]]
images_cropped = images


# Print some random indices extracted
N = 4
random_indices = np.random.randint(images_cropped.shape[0], size=(N*N,))
fig, axes = plt.subplots(N, N, figsize=(12, 8))
for i, val in enumerate(random_indices):
    axes[i//N, i%N].imshow(images_cropped[val])


J = images_cropped
W_m = est_auto_crop

# Wm = (255*PlotImage(W_m))
Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]

Wm = Wm + alpha * est_Ik

W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]


img = cv2.imread('images/fotolia_processed/fotolia_137840787.jpg')[None]
Jt = img[:, start[0]:start[0] + rect[0], start[1]:start[1] + rect[1]]


# now we have the values of alpha, Wm, J
# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)

img[:, start[0] + 2: start[0] + rect[0] - 2, start[1] + 2: start[1] + rect[1] - 2] = Ik[:, 2:-2, 2:-2]
img = img.astype(np.uint8)

plt.figure(figsize=(12, 12), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(img[0])
plt.show()