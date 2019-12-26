import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model


class GatedConvLayer(keras.layers.Layer):

    def __init__(self,hidden_dim,n_gram=3,decode=True):
        super(GatedConvLayer, self).__init__()
        self.hidden_dim=hidden_dim

        if decode:
            self.conv=layers.Conv1D(filters=hidden_dim*2,kernel_size=n_gram,padding="valid")
        else:
            self.conv=layers.Conv1D(filters=hidden_dim*2,kernel_size=n_gram,padding="same")

        self.conv.build(input_shape=(hidden_dim,hidden_dim,hidden_dim))

    def call(self,inputs):

        input_convolved=self.conv(inputs)

        gate,hidden=tf.split(input_convolved,num_or_size_splits=2,axis=-1)

        gate=tf.nn.sigmoid(gate)

        outputs=gate*hidden

        return outputs


class Embedding(layers.Layer):
    def __init__(self, vocab_size:int, embedding_dim:int, init_std:float=0.02, padding_idx:int=0):
        super(Embedding, self).__init__()

        self.vocab_size=vocab_size
        # tf.Variable(shape=(vocab_size,embedding_dim),ini)
        self.embedding_weight = self.add_weight(shape=(vocab_size,embedding_dim),
                                                initializer=keras.initializers.TruncatedNormal(stddev=init_std),
                                                name='embeddings')
        # padding_weights = self.weights[0][padding_idx]
        # padding_weights.assign(tf.zeros_like(padding_weights))


    def call(self, x: tf.Tensor, oracle: tf.Tensor = None, decay:float =None):


        x_one_hot = tf.one_hot(x, self.vocab_size)

        if oracle is None:
            return tf.matmul(x_one_hot,self.embedding_weight)

        else:
            oracle_one_hot=tf.one_hot(x, self.vocab_size)
            inp=x_one_hot*decay+(1-decay)*oracle_one_hot

            return tf.matmul(inp,self.embedding_weight)




class LayerNorm(layers.Layer):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)


    # @tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None]),))
    def call(self, inputs):

        return self.norm_1(inputs)



class LSTM(keras.Model):
    def __init__(self, hidden_size, vocab_size):
        super(LSTM, self).__init__()
        self.embedder=Embedding(vocab_size,hidden_size)
        self.lstm1 = keras.layers.LSTM(hidden_size, return_state=True, return_sequences=True,name="lstm")
        self.fc = keras.layers.Dense(vocab_size, input_shape=[hidden_size, ],name="classifier")

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None],dtype=tf.int64),tf.TensorSpec(shape=[None, None],dtype=tf.int64),),
                 experimental_relax_shapes=True)
    def call(self, x,y=None):

        x_embed = self.embedder(x)
        x_encoded, _, _ = self.lstm1(x_embed)

        # predict=tf.matmul(x_encoded,self.embedder.weights[0],transpose_b=True)
        predict=self.fc(x_encoded)
        return predict

