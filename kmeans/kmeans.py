import numpy as np
import cv2

image = cv2.imread('images/food.png')
x, y, z = image.shape
image2D = image.reshape(x*y, z)
image2D = np.float32(np.float32(image2D))


criteria = (cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)

num_of_clusters = 5

ret, label, center = cv2.kmeans(image2D, num_of_clusters, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
result = center[label.flatten()]
result = result.reshape((image.shape))

cv2.imwrite("segmented_image.png", result)

cv2.imshow('color segmented image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
