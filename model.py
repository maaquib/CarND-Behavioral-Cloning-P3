### Import libraries
import csv
import cv2 as cv
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import __version__ as keras_version
print('KERAS version {}'.format(keras_version))

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

### Constants
TOP_CROP    = 70
BOTTOM_CROP = 25
INPUT_SHAPE = (160, 320, 3)
STEERING_CORRECTION = 0.2

### Image preprocessing utils
def histogram_eq_transform(image):
    """
    http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    Takes the image data to be transformed
    Return: Histogram equalized image data
    """
    image[:,:,0] = cv.equalizeHist(image[:,:,0])
    image[:,:,1] = cv.equalizeHist(image[:,:,1])
    image[:,:,2] = cv.equalizeHist(image[:,:,2])
    return image

def preprocess(images):
    """
    Preprocess the images
    Returns: Processed image data
    """
    images = np.array([histogram_eq_transform(image) for image in images])
    return images

def load_simulation_files():
    """
    Loads simulation images and labels using the metadata csv file
    Returns: Loaded file names
    """
    lines = []
    with open('../data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # Skip the header row
        for line in reader:
            lines.append(line)
    return lines

def augment_data(images, steerings):
    """
    Augment data by flipping the images and negating the steering angles
    Returns: augmented dataset
    """
    np.concatenate((images, np.fliplr(images)))
    np.concatenate((steerings, -1 * steerings))
    return images, steerings

def generator(samples, batch_size=32):
    """
    Generator for loading images for model training
    Returns: shuffled training images
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = '../data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv.imread(center_name)
                # left_name = '../data/IMG/' + batch_sample[1].split('/')[-1]
                # left_image = cv.imread(left_name)
                # right_name = '../data/IMG/' + batch_sample[2].split('/')[-1]
                # right_image = cv.imread(right_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                # images.append(left_image)
                # images.append(right_image)
                angles.append(center_angle)
                # angles.append(center_angle + STEERING_CORRECTION)
                # angles.append(center_angle - STEERING_CORRECTION)

            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = augment_data(X_train, y_train)
            # X_train = preprocess(X_train)
            yield shuffle(X_train, y_train)


### Model
def build_model(dropout):
    """
    Using NVIDIA model
    https://arxiv.org/pdf/1604.07316v1.pdf
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(cropping=((TOP_CROP, BOTTOM_CROP), (0, 0))))
    model.add(Convolution2D(24, 5, 5,subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    model.add(Flatten())
    # model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(1))
    return model

def load_previous_model_if_exists(dropout):
    """
    Load weights from the last trained model if the model file exists
    rather than starting from scratch
    Returns: model
    """
    model = build_model(dropout)
    model_path = './model.h5'
    if os.path.exists(model_path):
        model.load_weights(model_path)
    return model


### Model training constants
BATCH_SIZE    = 32
DROPOUT       = 0.75
EPOCHS        = 5
LEARNING_RATE = 0.001

samples = load_simulation_files()
train_samples, validation_samples = train_test_split(samples, test_size=0.3)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = build_model(DROPOUT)

optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=optimizer)


checkpointer = ModelCheckpoint('./model.h5', monitor='val_loss', verbose=1, save_best_only=True)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 6, \
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples) * 6, \
                                     nb_epoch=EPOCHS, verbose=1, callbacks=[checkpointer])

model.save('model.h5')
