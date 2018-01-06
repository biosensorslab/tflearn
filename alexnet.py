# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from PIL import Image
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17
'''
img = Image.open("17flowers/jpg/0/image_0006.jpg")
img = img.resize((227, 227), Image.ANTIALIAS)
img = np.asarray(img, dtype="float32")

img2 = Image.open("17flowers/jpg/1/image_0096.jpg")
img2 = img2.resize((227, 227), Image.ANTIALIAS)
img2 = np.asarray(img2, dtype="float32")
imgs = np.asarray([img, img2])
'''
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(incoming=network, nb_filter=96, filter_size=11,  strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
#sigmoid
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                max_checkpoints=1, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True,
      show_metric=True, batch_size=64, snapshot_step=200,
      snapshot_epoch=False, run_id='alexnet_oxflowers17')

# prediction = model.predict([X[0]])
# print("Prediction: %s" % str(prediction[0]))
# print(np.argmax(prediction[0]))
# print(np.amax(prediction[0]))
model.evaluate(X, Y, 128)

results = model.predict(imgs)
print(results)
print(np.argmax(results[0]))
