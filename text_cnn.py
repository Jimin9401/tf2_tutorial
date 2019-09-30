import tensorflow as tf
from tensorflow.keras import layers,optimizers
from tensorflow import keras

class textCNN(keras.Model):
    def __init__(self, hidden_size, vocab_size, filter_size, N_gram_list, n_class):
        super(textCNN, self).__init__()
        self.embedder = layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.convlayers = []
        for kernel in N_gram_list:
            self.convlayers.append(layers.Conv1D(activation='relu', filters=30, kernel_size=kernel, strides=1))

        self.concat_hidden = len(N_gram_list * filter_size)

        self.fc = layers.Dense(n_class, activation='sigmoid', input_shape=[self.concat_hidden, ])

    @tf.function
    def call(self, x):
        x_embed = self.embedder(x)
        x_convolved = [c(x_embed) for c in self.convlayers]

        x_concatenated = None
        for xc in x_convolved:
            x_mapped = tf.reduce_max(xc, axis=1)
            if x_concatenated == None:
                x_concatenated = x_mapped
            else:
                x_concatenated = tf.concat([x_concatenated, x_mapped], axis=1)

        return self.fc(x_concatenated)



tf.keras.backend.set_learning_phase(1)

tf.keras.backend.learning_phase()
#