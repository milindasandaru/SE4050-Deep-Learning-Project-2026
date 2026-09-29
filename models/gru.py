"""
GRU (Gated Recurrent Unit) model for phishing website detection.

This module contains ONE function: build_gru(). It builds and returns a
compiled Keras model. All training/evaluation logic lives in the notebook
(notebooks/06_GRU.ipynb), so this file remains a small and reusable
architecture definition.
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_gru(input_shape=(50, 1), learning_rate=0.001):
    """
    Build and compile the GRU architecture.

    Architecture:
        Input(50, 1)
          -> GRU(64)
          -> Dropout(0.3)
          -> Dense(32) -> ReLU
          -> Dropout(0.3)
          -> Dense(1) -> Sigmoid

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample.
        For this project, the 50 numerical features are reshaped to (50, 1).

    learning_rate : float
        Learning rate used by the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        A compiled GRU model ready for training.
    """

    # Create the Sequential GRU model
    model = Sequential([

        # Input contains 50 timesteps with 1 feature value at each timestep
        Input(shape=input_shape),

        # GRU layer with 64 hidden units
        # This layer processes the input feature sequence
        GRU(64),

        # Dropout helps reduce overfitting by randomly disabling 30% of neurons
        Dropout(0.3),

        # Fully connected layer used to learn higher-level patterns
        Dense(32, activation="relu"),

        # Additional dropout for regularization
        Dropout(0.3),

        # Final output layer
        # Sigmoid returns a probability between 0 and 1
        Dense(1, activation="sigmoid")
    ])

    # Compile the model
    # Adam is used as the optimizer
    # Binary cross-entropy is used because this is a binary classification problem
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Return the compiled model to be trained in the notebook
    return model