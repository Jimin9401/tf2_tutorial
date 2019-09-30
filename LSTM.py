import tensorflow as tf
from tensorflow import keras
from tensorflow import keras


import tra


class LSTM(keras.Model):
    def __init__(self, hidden_size, vocab_size):
        super(LSTM, self).__init__()
        self.embedder = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.lstm1 = keras.layers.LSTM(hidden_size, return_state=True, return_sequences=True)
        self.fc = keras.layers.Dense(vocab_size, input_shape=[hidden_size, ])

    def call(self, X, training=None, mask=None):
        x = self.embedder(X)
        x, _, _ = self.lstm1(x)
        return self.fc(x)