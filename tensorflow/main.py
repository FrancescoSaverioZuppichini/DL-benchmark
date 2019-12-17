
import tensorflow as tf
import logging
from tensorflow import keras
import os
from pathlib import Path

root = Path(__file__).parent.absolute()
logging.basicConfig(filename=root / 'app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

fashion_mnist = keras.datasets.fashion_mnist

try:
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # rescale between 0 - 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=30, 
    callbacks=[keras.callbacks.CSVLogger(root / 'logs.csv')])
except Exception as e:
    logging.exception("Exception occurred")
logging.info("Done")
