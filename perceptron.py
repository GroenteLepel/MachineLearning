# Adapted from the code on https://www.tensorflow.org/tutorials/images/cnn 
from __future__ import absolute_import, division, print_function, \
    unicode_literals
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import time

# Download and prepare the CIFAR10 dataset
(train_images, train_labels), (
    test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create the convolutional base
n_hidden = 0
epochs = 500
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dropout(0.2))
# model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

time_start = time.time()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs,
                    validation_data=(test_images, test_labels),
                    batch_size=1000)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('done in {0:3.2f} seconds'.format(time.time() - time_start))


def plot_history():
    fig, ax = plt.subplots(1, 2, sharex='all', figsize=(10, 4))

    ax[0].set_title("Loss")
    ax[0].plot(history.history['loss'], label="train data")
    ax[0].plot(history.history['val_loss'], label="test data")
    ax[0].legend()
    ax[0].set_xlim((-10, epochs))

    ax[1].set_title("Accuracy (fraction)")
    ax[1].plot(history.history['acc'], label="train data")
    ax[1].plot(history.history['val_acc'], label="test data")

    fig.savefig("1hlayers150_dropout02_500epochs_1000batchsize.png")


plot_history()
