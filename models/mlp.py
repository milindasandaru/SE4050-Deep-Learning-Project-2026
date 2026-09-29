"""
MLP (Multi-Layer Perceptron) model for phishing website detection.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_mlp(input_dim, learning_rate=0.001):
    """
    3 hidden layers, shrinking each time
    Dropout after each one to stop it from just memorizing the training data
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