import tensorflow.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, \
    BatchNormalization, Flatten, SeparableConv2D


def build_model(input_shape):
    model = keras.Sequential([
        Conv2D(256, (4, 13), activation='relu', kernel_regularizer=keras.regularizers.l2(0.04),
               input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        SeparableConv2D(256, (4, 1), activation='relu', kernel_regularizer=keras.regularizers.l2(0.04)),
        BatchNormalization(),
        MaxPooling2D((2, 1), 2),

        SeparableConv2D(128, (3, 1), activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        SeparableConv2D(128, (3, 1), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 1), 2),

        SeparableConv2D(64, (2, 1), activation='relu', kernel_regularizer=keras.regularizers.l2(0.04)),
        Dropout(0.2),
        BatchNormalization(),
        SeparableConv2D(64, (2, 1), activation='relu', kernel_regularizer=keras.regularizers.l2(0.04)),
        BatchNormalization(),
        MaxPooling2D((2, 1), 2),

        Flatten(),
        Dense(2 ** 10, activation='relu', kernel_regularizer=keras.regularizers.l2(0.08)),
        Dropout(0.2),

        Dense(10, activation='softmax'),
    ])

    return model
