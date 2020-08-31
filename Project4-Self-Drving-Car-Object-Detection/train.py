#%%
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers, losses, Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import random

SEED = 13
random.seed(SEED)

AUTOTUNE = tf.data.experimental.AUTOTUNE

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

    return model

def preprocessing(path, label):
    img = tf.io.decode_and_crop_jpeg(tf.io.read_file(path), [60, 0, 80, 320])
    img = tf.image.random_brightness(img, 0.2)
    if random.randint(0,1) == 0:
        img = tf.image.flip_left_right(img)
        label = -label
    img = tf.image.random_contrast(img, 0.2, 0.5)
    return img/255.0, label

def load_data():
    d = pd.read_csv('driving_log.csv')[['center', 'steering']]
    df = d.query('steering != 0').append(d.query('steering == 0').sample(frac=0.05)).sample(frac=1)
    return tf.data.Dataset.from_tensor_slices((df['center'], df['steering'])).map(preprocessing, num_parallel_calls=AUTOTUNE)

def show(ds):
    sample = next(iter(ds))
    print(sample[1])
    plt.figure()
    plt.imshow(sample[0])
    plt.axis('off')

def train():
    model = get_model((80, 320, 3))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-2), 
        loss='mean_squared_error',
        metrics=['acc']
    )
    ds = load_data()
    model.fit(ds.prefetch(AUTOTUNE).batch(64), epochs=2)
    for i in range(10):
        predict(model, ds)

def predict(model, ds):
    sample = next(iter(ds))
    print(model.predict([sample[0]])[0], ', ', sample[1])
