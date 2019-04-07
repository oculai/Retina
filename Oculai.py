import numpy as np
import pandas as pd
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2 as cv

def prep_for_model(img):
    return cv.resize(np.array(img), (360, 360)).reshape(1, 360, 360, 3)

def build_model(weights_file):
    model = Sequential()

    model.add(Conv2D(3, (11, 11), activation='relu', input_shape=(360, 360, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(96, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(192, (3, 3)))

    model.add(Flatten())
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.load_weights(weights_file)

    return model

def get_prediction(img, model):
    return model.predict(prep_for_model(img))

"""
def get_next_ten(input, model, scaler):
    output_arr = []
    preds_arr = []
    first_pred = model.predict(input)
    output_arr = output_arr.append(scaler.inverse_transform(first_pred))
    for i in range(10):
        next_pred = model.predict(preds_arr[i])
        pred_arr = preds_arr.append(next_pred)
        output_arr = output_arr.append(scaler.inverse_transform(next_pred))
    return output_arr# models.py
"""
