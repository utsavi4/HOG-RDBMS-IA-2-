from skimage import feature
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('login3.png')
#feature.hog function to calculate the HOG features
(hog, hog_image) = feature.hog(image, orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', visualize=True, transform_sqrt=True)
# it returns two values


cv2.imshow('HOG Image', hog_image)
cv2.imwrite('hog.jpg', hog_image*255.)
cv2.waitKey(0)
