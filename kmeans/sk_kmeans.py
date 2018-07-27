from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/food.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# show image
plt.figure()
plt.axis("off")
plt.imshow(image)

x, y, z = image.shape
image2D = image.reshape(x*y, z)

kmeans = KMeans(init='k-means++', n_clusters=5, n_init=20, max_iter=250).fit(image2D)

# plot histogram of all colors
labels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
(hist, _) = np.histogram(kmeans.labels_, bins = labels)
hist = hist.astype("float")
hist = hist/hist.sum()

hist_bar = np.zeros((100, 200, 3), dtype = "uint8")
index_1 = 0
for (percent, color) in zip(hist, kmeans.cluster_centers_):
	index_2 = index_1 + (percent * 200)
	cv2.rectangle(hist_bar, (int(index_1), 0), (int(index_2), 100), color.astype("uint8").tolist(), -1)
	index_1 = index_2 

plt.figure()
plt.axis("off")
plt.imshow(hist_bar)
plt.show()
