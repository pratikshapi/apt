import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread, imsave
from skimage.color import rgb2hsv, hsv2rgb
import cv2

# plt.figure(num=None, figsize=(8, 6), dpi=80)


pacman_image = imread('graphics/pac-man-3.jpeg')
pacman_image_filtered_blue = (pacman_image[:,:,2] > 150 ) & ( pacman_image[:,:,1] < 150)
pacman_image_filtered_white = pacman_image[:,:,1] > 150

print(pacman_image_filtered_white)

# pacman_image_filtered_white = pacman_image_filtered_white and pacman_image_filtered_blue
# pacman_image_filtered_white = cv2.bitwise_and(pacman_image_filtered_white, pacman_image_filtered_blue, mask = None)
plt.figure(num=None, figsize=(8, 6), dpi=80)
imsave('pac-man-white.jpg', pacman_image_filtered_white)
imsave('pac-man-blue.jpg', pacman_image_filtered_blue)
pacman_image_filtered_blue = pacman_image_filtered_blue * 1
pacman_image_filtered_white = pacman_image_filtered_white * 2
pacman_image_filtered_blue = pacman_image_filtered_blue + pacman_image_filtered_white
print(pacman_image_filtered_blue)

np.savetxt('pac-man-processed.txt', pacman_image_filtered_blue, fmt='%i')