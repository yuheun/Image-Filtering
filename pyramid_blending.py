import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

"""
Imgae Pyramid & Blending
"""

apple = cv.imread('apple.png', cv.IMREAD_COLOR) # Read the image in color, ignoring transparent parts
orange = cv.imread('orange.png')

print(apple.shape)
print(orange.shape)

apple_pyrdown = cv.pyrDown(apple)
apple_pyrup = cv.pyrUp(apple)

cv.imshow(apple_pyrup)
cv.imshow(apple)
cv.imshow(apple_pyrdown)

cv.waitKey()
cv.destroyAllWindows()

apple_orange = np.hstack((apple[:, :256], orange[:, 256:])) # horizontal crop and attach

apple_cp = apple.copy()
gaupyr_apple = [apple_cp] # list to save scales

for i in range(6):
  apple_cp = cv.pyrDown(apple_cp)
  gaupyr_apple.append(apple_cp)

apple_cp = gaupyr_apple[5] # smallest
lappry_apple = [apple_cp] 

for i in range(5, 0, -1):
  gau_expanded = cv.pyrUp(gaupyr_apple[i])
  lap_diff = cv.subtract(gaupyr_apple[i-1], gau_expanded)
  lappry_apple.append(lap_diff) 

orange_copy = orange.copy()
gaupyr_orange = [orange_copy]

for i in range(6):
  orange_copy = cv.pyrDown(orange_copy)
  gaupyr_orange.append(orange_copy)

orange_copy = gaupyr_orange[5]
lappry_orange = [orange_copy]

for i in range(5, 0, -1):
  gau_expanded = cv.pyrUp(gaupyr_orange[i])
  lap_diff = cv.subtract(gaupyr_orange[i-1], gau_expanded)
  lappry_orange.append(lap_diff)

apple_orange_pyramid = []

n = 0

for apple_lap, orange_lap in zip(lappry_apple, lappry_orange):
  n += 1
  cols, rows, ch = apple_lap.shape
  laplacian = np.hstack((apple_lap[:, :int(cols/2)], orange_lap[:, int(cols/2):]))
  apple_orange_pyramid.append(laplacian)

apple_orange_reconstruct = apple_orange_pyramid[0] 

for i in range(1, 6):
  apple_orange_reconstruct = cv.pyrUp(apple_orange_reconstruct)
  apple_orange_reconstruct = cv.add(apple_orange_pyramid[i], apple_orange_reconstruct) # 차이 더해줘서 up 만드는..

cv.imshow(apple_orange)
cv.imshow(apple_orange_reconstruct)

cv.waitKey()
cv.destroyAllWindows()