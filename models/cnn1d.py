import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)


def build_cnn1d(input_shape=(50, 1)):
    """
    Build and return the 1D Convolutional Neural Network.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample.

    Returns
    -------
    tensorflow.keras.Model
        Compiled-ready CNN1D model.
    """

    model = Sequential([
        Input(shape=input_shape),

        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    return model