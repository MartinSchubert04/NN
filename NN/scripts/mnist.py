import numpy as np
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_images.tofile("NN\\data\\train_images.mat")
train_labels.tofile("NN\\data\\train_labels.mat")
test_images.tofile("NN\\data\\test_images.mat")
test_labels.tofile("NN\\data\\test_labels.mat")

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)