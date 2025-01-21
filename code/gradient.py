import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

"""
Image Gradients
"""

"""
Sobel

for x           for y

-1 0 1          -1 -2 -1
-2 0 2           0  0  0
-1 0 1           1  2  1
"""

img = cv.imread('lena_color.png')

sobelx = cv.Sobel(img, -1, 1, 0, 3) # ddepth=-1, dx=1, dy=0, 3x3 ksize
sobely = cv.Sobel(img, -1, 0, 1, 3)

abs_grad_x = cv.convertScaleAbs(sobelx)
abs_grad_y = cv.convertScaleAbs(sobely)

sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) #alpha 0.5, gamma 0

plt.subplot(141), plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(sobelx)
plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(sobely)
plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(sobel)
plt.xticks([]), plt.yticks([])

plt.show()

"""
Canny
"""

thres1 = 0
thres2 = 280

canny_img = cv.Canny(img, thres1, thres2)

cv.imshow(canny_img)

cv.waitKey()
cv.destroyAllWindows()

"""
Laplacian
"""

mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
mask2 = np.array([[1,1,1], [1,-8,1], [1,1,1]])
mask3 = np.array([[-1,-1,-1], [-1, 8, -1], [-1, -1, -1]])

laplacian1 = cv.filter2D(img, -1, mask1)
laplacian2 = cv.filter2D(img, -1, mask2)
laplacian3 = cv.filter2D(img, -1, mask3)
laplacian = cv.Laplacian(img, -1)

plt.figure(figsize=(20, 15))
plt.subplot(141), plt.imshow(laplacian1)
plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(laplacian2)
plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(laplacian3)
plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(laplacian)
plt.xticks([]), plt.yticks([])

plt.show()
