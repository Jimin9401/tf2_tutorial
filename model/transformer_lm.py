from .layers.layers_util import *
from .layers.layer import LayerNorm,Embedding
import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow import keras

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0

        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.wq = self.add_weight(shape=[d_model,d_model],initializer=keras.initializers.TruncatedNormal(stddev=0.02))
        self.wk = self.add_weight(shape=[d_model,d_model],initializer=keras.initializers.TruncatedNormal(stddev=0.02))
        self.wv = self.add_weight(shape=[d_model,d_model],initializer=keras.initializers.TruncatedNormal(stddev=0.02))
        self.dense = self.add_weight(shape=[d_model, d_model], initializer=keras.initializers.TruncatedNormal(stddev=0.02))


    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None]),))
    def split_heads(self, x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        batch_size=tf.shape(x)[0]
        x_split = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x_split, perm=[0, 2, 1, 3])


    # @tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None],dtype=tf.float32),tf.TensorSpec(shape=[None,None,None],dtype=tf.float32),
    #                               tf.TensorSpec(shape=[None,None,None],dtype=tf.float32),tf.TensorSpec(shape=[None,None],dtype=tf.float32)))
    # @tf.function
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = tf.matmul(q,self.wq)  # (batch_size, seq_len, d_model)
        k = tf.matmul(k,self.wk)  # (batch_size, seq_len, d_model)
        v = tf.matmul(v,self.wv)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention,attention_weights = scale_dot_product(q, k, v, mask)


        scaled_attention = tf.transpose(scaled_attention,perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)


        output = tf.matmul(concat_attention,self.dense)  # (batch_size, seq_len_q, d_model)


        return output


class PositionwiseFF(layers.Layer):
    def __init__(self,d_ff,d_model):
        super(PositionwiseFF, self).__init__()
        # self.a1=layers.Dense(d_ff,input_shape=(d_model,))
        # self.a2=layers.Dense(d_model,input_shape=(d_model,))
        #
        self.a1 = self.add_weight(shape=[d_model,d_ff],initializer=keras.initializers.TruncatedNormal(stddev=0.02))
        self.a2 = self.add_weight(shape=[d_ff,d_model],initializer=keras.initializers.TruncatedNormal(stddev=0.02))

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None]),))
    def call(self,inputs):

        return tf.matmul(tf.matmul(inputs,self.a1),self.a2)





class DecoderBlock_lm(layers.Layer):
    def __init__(self,d_model,d_ff,h,p_rate=0):
        super(DecoderBlock_lm, self).__init__()
        self.multi1=MultiheadAttention(d_model,h)
        self.multi2=MultiheadAttention(d_model,h)
        self.norm1=LayerNorm()
        self.norm2=LayerNorm()

        self.ffn=PositionwiseFF(d_ff,d_model)


        self.dropout1=layers.Dropout(p_rate)
        self.dropout2=layers.Dropout(p_rate)
        self.dropout3=layers.Dropout(p_rate)

    def call(self, inputs,y_mask):

        z=self.multi1(inputs,inputs,inputs,y_mask)
        z=self.dropout1(z)
        z=self.norm1(inputs+z)

        return self.dropout3(self.ffn(z))



class Decoder_lm(layers.Layer):
    def __init__(self,d_model,d_ff,h,num_layers):
        super(Decoder_lm, self).__init__()
        self.decoder_blocks=[ DecoderBlock_lm(d_model,d_ff,h)  for _ in range(num_layers)]
        self.num_layers=num_layers


    def call(self,dec_states,y_mask):

        for i in range(self.num_layers):
            dec_states = self.decoder_blocks[i](dec_states,y_mask)

        return dec_states



class Transformer_LM_Base(keras.Model):
    def __init__(self,num_layers,d_model,d_ff,h,vocab_size):
        super(Transformer_LM_Base, self).__init__()


        self.embedder=Embedding(vocab_size,d_model)
        self.decoder=Decoder_lm(d_model=d_model,d_ff=d_ff,h=h,num_layers=num_layers)

        # self.classifer=layers.Dense(input_dim=d_model,units=vocab_size)
        # self.embedder.weights([0])

    def call(self,inp):

        inp_seq_lens=tf.shape(inp)[1]
        tril = tf.linalg.band_part(tf.ones((inp_seq_lens, inp_seq_lens), dtype=tf.float32), -1, 0)
        inp_mask=create_mask(inp)

        inp_embed=self.embedder(inp)

        dec_state=self.decoder(inp_embed,inp_mask,tril)
        predict=tf.matmul(dec_state,self.embedder.weights[0],transpose_b=True)

        return predict

class Transformer_LM_Oracle(Transformer_LM_Base):
    def __init__(self,num_layers,d_model,d_ff,h,vocab_size):
        super(Transformer_LM_Oracle, self).__init__(num_layers,d_model,d_ff,h,vocab_size)


    def call(self,inp,oracle=None,decay:float=None):

        inp_seq_lens=tf.shape(inp)[1]
        tril = tf.linalg.band_part(tf.ones((inp_seq_lens, inp_seq_lens), dtype=tf.float32), -1, 0)

        if oracle is None:
            inp_embed=self.embedder(inp)
        else:
            inp_embed = self.embedder(inp,oracle,decay)

        dec_state=self.decoder(inp_embed,tril)

        predict=tf.matmul(dec_state,self.embedder.weights[0],transpose_b=True)


        return predict




