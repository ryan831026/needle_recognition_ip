from scipy.cluster.vq import kmeans
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

path = 'needle_images/'
output_path = 'image_processing_output/'

for i, filename in enumerate(glob.glob(os.path.join(path, '*.png'))):
    fname = filename.split('/')[1]

    # read image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find phantom
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    threshold = 200
    mask_phantom2_raw = img_hls[:, :, 2] > threshold
    kernel = np.ones((3, 3), np.uint8)
    mask_phantom2 = mask_phantom2_raw.astype(np.uint8)
    mask_phantom2 = cv2.morphologyEx(
        mask_phantom2, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_phantom2 = cv2.erode(mask_phantom2, kernel, iterations=8)

    # find needle
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(img_grey, cv2.CV_64F, ksize=5)
    threshold = 150
    laplacian = laplacian > 150

    # find needle in phantom
    needle_raw = np.bitwise_and(laplacian, mask_phantom2)
    needle = cv2.morphologyEx(needle_raw, cv2.MORPH_OPEN, kernel, iterations=2)

    # plot
    img_needle = np.zeros(img.shape, dtype=int)
    img_needle[:, :, 0] = needle * 255
    overlayed_img = 0.3 * img + 0.7 * img_needle
    overlayed_img = overlayed_img.astype(np.uint8)

    # save img

    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    # overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2GRAY)

    cv2.imwrite(os.path.join(output_path, fname), overlayed_img)
    print(f'save image #{i}')
