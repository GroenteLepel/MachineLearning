#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:15:59 2019

@author: laurens
"""

# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import datasets, layers, models, regularizers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
# Download and prepare the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
#%%
#We made a custom activation function
from keras.backend import sigmoid
def swish(x, beta = 14):
    return (x * sigmoid(beta * x))

#%%

#We tried different configurations with different number of layers, activation functions, dropout, l1/l2 regularization.
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(60,activation=swish,activity_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.15),
    layers.BatchNormalization(),
    layers.Dense(30,activation=swish,activity_regularizer=regularizers.l2(0.001)),
   # layers.Dropout(0.15),
    layers.BatchNormalization(),
##    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


#%%

#train_images2=(train_images-np.mean(train_images))/np.std(train_images)
#test_images2=(test_images-np.mean(train_images))/np.std(train_images)


history2 = model.fit(train_images, train_labels_one_hot, epochs=50,
                    validation_data=(test_images, test_labels_one_hot),batch_size=1000)

test_loss, test_acc = model.evaluate(test_images, test_labels_one_hot)

#%%

plt.plot(history2.history['accuracy'], label='accuracy')
plt.plot(history2.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history2.history['loss'], label='accuracy')
plt.plot(history2.history['val_loss'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()

