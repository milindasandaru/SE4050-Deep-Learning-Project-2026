"""
MLP (Multi-Layer Perceptron) model for phishing website detection.

This module contains ONE function: build_mlp(). It builds and returns a
compiled Keras model. All training/evaluation logic lives in the notebook
(notebooks/03_MLP.ipynb) so this file stays a small, reusable "architecture
definition" -- exactly like models/cnn1d.py does for the CNN.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_mlp(input_dim, learning_rate=0.001):
    """
    Build and compile the MLP architecture.

    Architecture (matches the team's agreed diagram):
        Input(input_dim)
          -> Dense(128) -> ReLU -> Dropout(0.3)
          -> Dense(64)  -> ReLU -> Dropout(0.3)
          -> Dense(32)  -> ReLU -> Dropout(0.2)
          -> Dense(1)   -> Sigmoid   (phishing probability)

    Parameters
    ----------
    input_dim : int
        Number of input features (50 for our processed PhiUSIIL data).
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        A compiled, ready-to-train Keras model.
    """

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
