import tensorflow as tf
from tensorflow import keras
from layers.layer import GatedConvLayer,Embedding
from layers.layers_util import *
from tensorflow.keras import layers,optimizers
import copy
import argparse
import os
import numpy as np


class ConvEncoder(layers.Layer):
    def __init__(self,vocab_size,hidden_dim,num_layers,n_gram=3):
        super(ConvEncoder, self).__init__()
        # self.embedder=layers.Embedding(input_dim=vocab_size,output_dim=hidden_dim)
        #self.pos_enc=positional_encoding(vocab_size,hidden_dim)
        self.conlayers=[GatedConvLayer(hidden_dim=hidden_dim,n_gram=n_gram,decode=False)\
                        for _ in range(num_layers)]


    def call(self,src):


        for c in self.conlayers:
            src=c(src)

        return src


class ConvDecoder(layers.Layer):

    def __init__(self,vocab_size,hidden_dim,num_layers,n_gram=3):
        super(ConvDecoder, self).__init__()
        self.embedder=layers.Embedding(input_dim=vocab_size,output_dim=hidden_dim)
        self.hidden_dim=hidden_dim
        self.conlayers=[GatedConvLayer(hidden_dim=hidden_dim,n_gram=n_gram,decode=True)\
                        for _ in range(num_layers)]


        #self.pos_enc = positional_encoding(vocab_size, hidden_dim)

        self.n_gram=n_gram
        self.num_layers=num_layers


    def call(self,trg,src_embed,mask):

        batch_size = tf.shape(trg)[0]
        # seq_lens = trg_embed.shape[1]
        # hidden_dim = trg_embed.shape[2]

        pad = tf.zeros(shape=[batch_size, self.n_gram - 1, self.hidden_dim], dtype=tf.float32)

        for c in self.conlayers:
            trg+=c(tf.concat([pad,trg],axis=1))

        weighted_sum = dot_product_attention(query=trg, key=src_embed, mask=mask)
        info = tf.concat([weighted_sum, trg], axis=-1)

        return info



class ConvSeq2Seq(layers.Layer):
    def __init__(self,vocab_size,num_layeres,hidden_size,vocab,n_gram=3):
        super(ConvSeq2Seq, self).__init__()
        self.vocab_size=vocab_size
        self.vocab=vocab
        self.embedder=Embedding(vocab_size,hidden_size)
        # self.embedder=Embedding(vocab_size,hidden_size)

        self.encoder=ConvEncoder(vocab_size=vocab_size,hidden_dim=hidden_size,\
                             num_layers=num_layeres,n_gram=n_gram)
        self.decoder =ConvDecoder(vocab_size=vocab_size, hidden_dim=hidden_size, \
                               num_layers=num_layeres, n_gram=n_gram)

        # self.classifier=layers.Dense(units=vocab_size,input_shape=(hidden_size,))

        self.classifier = self.add_weight(shape=(hidden_size*2, vocab_size),
                                                initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                                name='classifier')


    def call(self,src,trg):

        # assert ~(trg is None and training==True)

        # batch_size=src.shape[0]
        src_embed=self.embedder(src)
        src_encoded=self.encoder(src_embed)

        src_lens = tf.shape(src)[1]
        trg_lens = tf.shape(trg)[1]

        src_mask = tf.cast(tf.equal(src, 0), dtype=tf.float32) * -1e7
        trg_mask = tf.cast(tf.equal(trg, 0), dtype=tf.float32) * -1e7

        mask = tf.tile(tf.expand_dims(src_mask, axis=1), multiples=[1, trg_lens, 1]) \
               * tf.tile(tf.expand_dims(trg_mask, axis=-1), multiples=[1, 1, src_lens])

        mask = tf.cast(tf.equal(mask, 0), tf.float32) * -1e5
        trg_embed=self.embedder(trg)
        dec_state=self.decoder(trg_embed,src_encoded,mask)
        predict=tf.matmul(dec_state,self.classifier)

        return predict[:,:-1,:]

