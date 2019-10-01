from __future__ import absolute_import, division, print_function, unicode_literals


from utils import get_model
from preprocessing import preprocessing
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras import layers,optimizers,backend
import os

from tensorflow import keras



def create_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)



class Embedder(layers.Layer):
    def __init__(self,d_model,vocab_size):
        super(Embedder, self).__init__()
        self.d_model=d_model
        self.embedder=layers.Embedding(input_dim=vocab_size,output_dim=d_model)

    def call(self, inputs):
        input_embed=self.embedder(inputs)
        seq_lens=input_embed.shape[1]
        dim=input_embed.shape[2]

        input_embed+=positional_encoding(seq_lens,dim)

        return input_embed


def scale_dot_product(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)


    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output,attention_weights




# class Scaled_dot_product(layers.Layer):
#     def __init__(self):
#         super(Scaled_dot_product, self).__init__()
#
#
#     def call(self,query,key,value,mask):
#         batch_size = query.shape[0]
#         h = query.shape[1]
#         d_h = query.shape[-1]
#         score = tf.matmul(query, key, transpose_b=True)
#         score = score / tf.math.sqrt(tf.cast(d_h, dtype=tf.float32))
#
#         print(mask)
#         print(score)
#         if mask is not None:
#             score += (mask*-1e9)
#
#         attn_score = tf.nn.softmax(score, axis=-1)
#
#         weighted_sums = tf.matmul(attn_score, value)
#
#
#
#         return weighted_sums
#

# class MultiheadAttention(layers.Layer):
#     def __init__(self,d_model,h):
#         super(MultiheadAttention, self).__init__()
#         self.Qw=layers.Dense(units=d_model)
#         self.Kw=layers.Dense(units=d_model)
#         self.Vw=layers.Dense(units=d_model)
#         self.h=h
#         self.d_head=d_model//h
#         self.d_model=d_model
# #        self.attn=Scaled_dot_product()
#
#
#     def split_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, -1, self.h,self.d_head))
#         return tf.transpose(x, perm=[0, 2, 1, 3])
#
#
#     def call(self, query,key,value,mask):
#         batch_size=query.shape[0]
#         query = self.Qw(query)
#         key = self.Kw(key)
#         value = self.Vw(value)
#
#         query=self.split_heads(query,batch_size)
#         key=self.split_heads(key,batch_size)
#         value=self.split_heads(value,batch_size)
#
#         #z=scale_dot_product(query,key,value,mask)
#
#         scaled_attention = scale_dot_product(query, key, value, mask)
#
#         scaled_attention = tf.transpose(scaled_attention,
#                                         perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
#
#         z = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
#         return z

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention,attention_weights = scale_dot_product(q, k, v, mask)


        scaled_attention = tf.transpose(scaled_attention,perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)


        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)


        return output



class LayerNorm(layers.Layer):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs):

        return self.norm_1(inputs)


class PositionwiseFF(layers.Layer):
    def __init__(self,d_ff,d_model):
        super(PositionwiseFF, self).__init__()
        self.a1=layers.Dense(d_ff)
        self.a2=layers.Dense(d_model)


    def call(self,inputs):

        return self.a2(self.a1(inputs))



class EncoderBlock(layers.Layer):
    def __init__(self,d_model,h,d_ff,p_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.multihead=MultiheadAttention(d_model,h)
        self.norm1=LayerNorm()
        self.norm2=LayerNorm()
        self.ffn=PositionwiseFF(d_ff,d_model)

        self.dropout1=layers.Dropout(p_rate)
        self.dropout2=layers.Dropout(p_rate)
        self.dropout3=layers.Dropout(p_rate)



    def call(self,inputs,mask):

        z=self.multihead(inputs,inputs,inputs,mask)
        z=self.dropout1(z)
        out1=self.norm1(z+inputs)

        z=self.dropout2(self.ffn(out1))


        return self.norm2(z+out1)


class Encoder(layers.Layer):
    def __init__(self,num_layer,h,d_model,d_ff):
        super(Encoder, self).__init__()
        self.num_layer=num_layer
        self.blocks=[EncoderBlock(d_model,h,d_ff) for _ in range(num_layer)]


    def call(self,z,mask):

        for block in self.blocks:
            z=block(z,mask)


        return z


class DecoderBlock(layers.Layer):
    def __init__(self,d_model,d_ff,h,p_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.multi1=MultiheadAttention(d_model,h)
        self.multi2=MultiheadAttention(d_model,h)
        self.norm1=LayerNorm()
        self.norm2=LayerNorm()

        self.ffn=PositionwiseFF(d_ff,d_model)


        self.dropout1=layers.Dropout(p_rate)
        self.dropout2=layers.Dropout(p_rate)
        self.dropout3=layers.Dropout(p_rate)


    def call(self, inputs,enc_outputs,x_mask,y_mask):

        z=self.multi1(inputs,inputs,inputs,y_mask)

        z=self.dropout1(z)

        output1=self.norm1(inputs+z)

        output2=self.multi2(output1,enc_outputs,enc_outputs,x_mask)
        output2=self.dropout2(output2)

        output3=self.norm2(output1+output2)


        return self.dropout3(self.ffn(output3))


class Decoder(layers.Layer):
    def __init__(self,d_model,d_ff,h,num_layers):
        super(Decoder, self).__init__()
        self.decoder_blocks=[ DecoderBlock(d_model,d_ff,h)  for _ in range(num_layers)]
        self.num_layers=num_layers


    def call(self,dec_states,enc_outputs,x_mask,y_mask):

        for i in range(self.num_layers):
            dec_states=self.decoder_blocks[i](dec_states,enc_outputs,x_mask,y_mask)

        return dec_states






class Transformer(keras.Model):
    def __init__(self,num_layers,d_model,d_ff,h,vocab_size):
        super(Transformer, self).__init__()
        self.embedder= Embedder(d_model,vocab_size)
        self.encoder=Encoder(num_layer=num_layers,h=h,d_model=d_model,d_ff=d_ff)

        self.decoder=Decoder(d_model=d_model,d_ff=d_ff,h=h,num_layers=num_layers)

        self.classifer=layers.Dense(units=vocab_size)


    def call(self,src,trg,decode=False):

        batch_size=src.shape[0]
        trg_seq_lens=trg.shape[1]
        tril = tf.linalg.band_part(tf.ones((trg_seq_lens, trg_seq_lens), dtype=tf.float32), -1, 0)

        src_mask=create_mask(src)

        src_embed=self.embedder(src)
        src_state=self.encoder(src_embed,src_mask)

        sequence_predicted=None

        trg_embed=self.embedder(trg)


        dec_state=self.decoder(trg_embed,src_state,src_mask,tril)
        predict=self.classifer(dec_state)

        if sequence_predicted is None:
            sequence_predicted=predict
        else:
            sequence_predicted=tf.concat([sequence_predicted,predict],axis=1)

        return sequence_predicted[:,:-1,:]


## for debugging

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser=argparse.ArgumentParser()
    parser.add_argument("-f_name",type=str,default="./data/token_data.pickle")
    parser.add_argument("-d_model",type=int,default=256)
    parser.add_argument("-d_ff",type=int,default=1024)
    parser.add_argument("-head",type=int,default=4)
    parser.add_argument("-epoch",type=int,default=30)
    parser.add_argument("-batch_size",type=int,default=64)
    parser.add_argument("-num_layer",type=int,default=4)
    parser.add_argument("-eval_step",type=int,default=500)
    parser.add_argument("-eval_after",type=int,default=1000)
    parser.add_argument("-lr",type=float,default=3e-3)
    parser.add_argument("-output_dir", type=str, default="./output")
    parser.add_argument("-use_half_precision",type=bool, default=False)


    args=parser.parse_args()

    if args.use_half_precision:
        dtype = 'float16'
        backend.set_floatx(dtype)

    train_ds,eval_ds, word2idx, idx2word = preprocessing(f_name="./data/token_data.pickle",batch_size=args.batch_size)

    transformer=Transformer(num_layers=args.num_layer,d_model=args.d_model,d_ff=args.d_ff,vocab_size=len(word2idx),h=args.head)

    model=get_model(model=transformer,eval_after=args.eval_after,eval_step=args.eval_step,output_dir=args.output_dir,model_name="Transformer")


    model.train(epoch=args.epoch,train_data=train_ds,eval_data=eval_ds,lr=args.lr)

if __name__=="__main__":
    main()


