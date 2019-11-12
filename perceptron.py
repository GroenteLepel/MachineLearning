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
n_hidden = 1
n_nodes1 = 2000
# n_nodes2 = 750
epochs = 600
batch = 5000
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_nodes1, activation='relu'))
model.add(layers.Dropout(0.3))
# model.add(layers.Dense(n_nodes2, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

time_start = time.time()
model.compile(optimizer='nadam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs,
                    validation_data=(test_images, test_labels),
                    batch_size=batch)

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

    fig.savefig("{0:d}hlayers{1:d}_{2:d}epochs_{3:d}batchsize_dropout0503_nadam.png".format(
        n_hidden, n_nodes1, epochs, batch))

    fig.show()


plot_history()
