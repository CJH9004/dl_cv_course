#%%
import numpy as np
from tensorflow.keras.models import Sequential
import csv
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, optimizers, losses
from matplotlib import pyplot
import tensorflow as tf
import pandas as pd
import random

SEED = 13
random.seed(SEED)

def get_model(shape):
    model = Sequential([
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=shape),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(1164, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-2), 
        loss='mean_squared_error'
    )

    return model

def load_data():
    d = pd.read_csv('driving_log.csv')[['center', 'steering']]