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
INPUT_SHAPE = (128, 128, 3)

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
    img = tf.io.decode_and_crop_jpeg(tf.io.read_file(path), [80, 0, 60, 320], channels=3)
    img = tf.image.resize(img, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
    img = tf.image.random_brightness(img, 0.2)
    if random.randint(0,1) == 0:
        img = tf.image.flip_left_right(img)
        label = -label
    img = tf.image.random_contrast(img, 0.2, 0.5)
    img = tf.cast(img, tf.float32) / 255.0 - 0.5
    return img, label

def load_data():
    d = pd.read_csv('driving_log.csv')[['center', 'steering']]
    df = d.query('steering != 0').append(d.query('steering == 0').sample(frac=0.05)).sample(frac=1)
    return tf.data.Dataset.from_tensor_slices((df['center'], df['steering'])).map(preprocessing, num_parallel_calls=AUTOTUNE)

def show(ds):
    sample = next(iter(ds))
    print(sample[0], sample[1])
    plt.figure()
    plt.imshow(sample[0])
    plt.axis('off')

def train(ds):
    model = get_model(INPUT_SHAPE)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4), 
        loss='mean_squared_error',
    )
    model.fit(ds.prefetch(AUTOTUNE).batch(64), epochs=5)
    return model

def predict(model, ds):
    it = iter(ds)
    for i in range(10):
        sample = next(it)
        print(tf.squeeze(model.predict(tf.expand_dims(sample[0], axis=0))).numpy(), ', ', sample[1].numpy())
