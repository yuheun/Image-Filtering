"""
Image Cropping
"""

import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

img = cv.imread('lena_color.png')

print('image size: ', img.shape)
h, w, c = img.shape # height, width, channel

print('pixel intensity value: ', img[100, 70]) # position of (100, 70)
print('pixel in: ', img[511, 511]) # 512 = out of range

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # opencv: bgr [012]

roi = img[50:300, 30:280] # crop - ROI(Region of Interest) extraction by slicing

cv.imshow(img)
cv.imshow(roi)

cv.waitKey()
cv.destroyAllWindows()