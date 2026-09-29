"""
Modular 1D Convolutional Neural Network (1D CNN) for phishing website detection.

Provides build_cnn1d() to generate parameterized models for systematic
hyperparameter tuning and controlled experimentation.
"""

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
from tensorflow.keras.optimizers import Adam


def build_cnn1d(
    input_shape=(50, 1),
    conv1_filters=32,
    conv2_filters=64,
    conv3_filters=None,
    kernel_size=3,
    dense_units=64,
    dropout_rate=0.3,
    learning_rate=0.001
):
    """
    Build, compile, and return a parameterized 1D Convolutional Neural Network.

    Parameters
    ----------
    input_shape : tuple, default=(50, 1)
        Input shape per sample (timesteps, channels).
    conv1_filters : int, default=32
        Number of filters in the first Conv1D layer.
    conv2_filters : int, default=64
        Number of filters in the second Conv1D layer.
    conv3_filters : int or None, default=None
        Optional number of filters in a third Conv1D layer (e.g. 128).
    kernel_size : int, default=3
        Convolution window length across features.
    dense_units : int, default=64
        Number of hidden units in the fully connected dense layer.
    dropout_rate : float, default=0.3
        Dropout regularization rate before output.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.

    Returns
    -------
    tensorflow.keras.Model
        Compiled Keras Sequential model ready for training.
    """
    layers = [
        Input(shape=input_shape),

        Conv1D(
            filters=conv1_filters,
            kernel_size=kernel_size,
            activation="relu"
        ),
        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=conv2_filters,
            kernel_size=kernel_size,
            activation="relu"
        ),
        MaxPooling1D(pool_size=2)
    ]

    if conv3_filters is not None:
        layers.extend([
            Conv1D(
                filters=conv3_filters,
                kernel_size=kernel_size,
                activation="relu"
            ),
            MaxPooling1D(pool_size=2)
        ])

    layers.extend([
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model = Sequential(layers)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model