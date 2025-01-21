import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

"""
Image Smoothing
"""

"""
Average Blur
"""
img = cv.imread('lena_color.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.float32)/25 # kernel setting
conv_img = cv.filter2D(img, -1, kernel) # convolution 2D smoothing
conv_gray_img = cv.filter2D(gray, -1, kernel)

cv.imshow(img)
cv.imshow(conv_img)
cv.imshow(gray)
cv.imshow(conv_gray_img)

cv.waitKey()
cv.destroyAllWindows()

"""
Gaussian Blur
"""

gau_sigma1_img = cv.GaussianBlur(gray, (5, 5), 1)
gau_sigma3_img = cv.GaussianBlur(gray, (5, 5), 3)
gau_sigma57_img = cv.GaussianBlur(gray, (5, 5), 7)
gau_sigma07_img = cv.GaussianBlur(gray, (0, 0), 7) 
# If you set kernel 0 x 0, it will be automatically determined based on the sigma value

cv.imshow(gray)
cv.imshow(gau_sigma1_img)
cv.imshow(gau_sigma3_img)
cv.imshow(gau_sigma57_img)
cv.imshow(gau_sigma07_img)

cv.waitKey()
cv.destroyAllWindows()

"""
Median Blur for Salt-and-Pepper Noise
"""

noise1 = cv.imread('noise_p1.png')
noise2 = cv.imread('noise_p2.png')
noise3 = cv.imread('noise_p3.png')

noise1_gau = cv.GaussianBlur(noise1, (5, 5), 5)
noise2_gau = cv.GaussianBlur(noise2, (5, 5), 5)
noise3_gau = cv.GaussianBlur(noise3, (5, 5), 5)

noise1_med = cv.medianBlur(noise1, 5, 5)
noise2_med = cv.medianBlur(noise2, 5, 5)
noise3_med = cv.medianBlur(noise3, 5, 5)

plt.subplot(231), plt.imshow(noise1_gau)
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(noise2_gau)
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(noise3_gau)
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(noise1_med)
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(noise2_med)
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(noise3_med)
plt.xticks([]), plt.yticks([])

plt.show()