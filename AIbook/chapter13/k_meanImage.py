import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 10
image = plt.imread("view.jpg")
image = np.array(image, dtype=np.float64) / 255
w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, d))
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)
newImage=np.zeros((w, h, d))
label_index = 0
for i in range(w):
    for j in range(h):
        newImage[i][j] = kmeans.cluster_centers_[labels[label_index]]
        label_index += 1


plt.figure()
sub1=plt.subplot(1,2,1)
plt.axis('off')
plt.title('Original image (full colors)')
plt.imshow(image)

sub2=plt.subplot(1,2,2)
plt.axis('off')
plt.title('Quantized image (10 colors, K-Means)')
plt.imshow(newImage)
plt.show()

