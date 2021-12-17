import time
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

import mfcc_model1
import mfcc_model2


def plot(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train")
    axs[0].plot(history.history["val_accuracy"], label="test")

    axs[1].plot(history.history["loss"], label="train")
    axs[1].plot(history.history["val_loss"], label="test")

    plt.show()


def prepare_dataset(x, y, test_size, val_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size)

    x_train = tf.expand_dims(x_train, axis=-1)
    x_val = tf.expand_dims(x_val, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    with open('data.json', "r") as fp:
        data = json.load(fp)

    x = np.array(data["data"])
    y = np.array(data["labels"])

    x_train, x_val, x_test, y_train, y_val, y_test = prepare_dataset(x, y, 0.25, 0.2)

    model = mfcc_model1.build_model(x_train.shape[1:])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=500)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\ntest:', test_acc)


    class TimeCallback(keras.callbacks.Callback):
        def on_predict_begin(self, logs=None):
            self.start_time = time.time()

        def on_predict_end(self, logs=None):
            print(f"{time.time() - self.start_time} seconds")


    model.predict(x_test, callbacks=[TimeCallback()])

    # plot(history)


