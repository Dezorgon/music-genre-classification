import tensorflow.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, \
    BatchNormalization, Flatten, SeparableConv2D


def build_model(input_shape):
    model = keras.Sequential([
        Conv2D(256, (4, 13), activation='relu', input_shape=input_shape),
        MaxPooling2D((4, 1), 4),
        Dropout(0.3),

        SeparableConv2D(256, (4, 1), activation='relu'),
        MaxPooling2D((2, 1), 2),
        Dropout(0.3),

        SeparableConv2D(512, (4, 1), activation='relu'),
        MaxPooling2D((2, 1), 2),
        Dropout(0.3),

        Flatten(),
        Dense(2 ** 13, activation='relu'),
        Dropout(0.3),

        Dense(10, activation='softmax'),
    ])

    return model
