from scipy.cluster.vq import kmeans
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans

# read image
# img = cv2.imread('needle_images/right_0.10502971240232001.png')
img = cv2.imread('needle_images/left_0.040011234283447265.png')
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
overlayed_img = overlayed_img.astype(int)
plt.imshow(overlayed_img)
plt.show()
