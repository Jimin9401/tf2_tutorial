
from __future__ import absolute_import, division, print_function, unicode_literals

from preprocessing import preprocessing
import tensorflow as tf
from tensorflow.keras import layers,optimizers
import copy

import argparse
import os
from tensorflow import keras
from utils import get_model

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

@tf.function(experimental_relax_shapes=True)
def compute_loss(predict,true):
    pre_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict,labels=true)
    pre_loss*=tf.cast(tf.logical_not(tf.equal(x=0,y=true)),tf.float32)

    return tf.reduce_mean(pre_loss)

@tf.function(experimental_relax_shapes=True)
def compute_accuracy(predict,true):
    predictions=tf.cast(tf.argmax(predict,axis=-1),tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions,true),tf.float32))

@tf.function(experimental_relax_shapes=True)
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predict = model(x,y,True)
        loss = compute_loss(predict=predict[:,:-1],true= y[:,1:])

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(predict=predict[:,:-1],true= y[:,1:])

    return loss, accuracy

def train(epoch,model,optimizer,train_data):
    loss=0.0
    accuracy=0.0
    for e in range(epoch):
        for step,(x,y) in enumerate(train_data):
            loss,accuracy=train_one_step(model,optimizer,x,y)
            if((step+1)%50==0):
                print("epoch:",e+1," step:",step+1," loss: {:0.5}".format(loss.numpy())," accuracy:  {:0.4}".format(accuracy.numpy()))
        print()
        print("epoch:",e+1," loss: {:0.5}".format(loss.numpy())," accuracy:  {:0.4}".format(accuracy.numpy()))
        print()
    return loss,accuracy


class Convlayer(keras.layers.Layer):
    def __init__(self,hidden_dim,n_gram=3,decode=True):
        super(Convlayer, self).__init__()
        self.hidden_dim=hidden_dim

        if decode:
            self.conv=layers.Conv1D(filters=hidden_dim*2,kernel_size=n_gram,padding="valid")
        else:
            self.conv=layers.Conv1D(filters=hidden_dim*2,kernel_size=n_gram,padding="same")

    def call(self,inputs):

        input_convolved=self.conv(inputs)

        gate,hidden=tf.split(input_convolved,num_or_size_splits=2,axis=-1)

        gate=tf.nn.sigmoid(gate)

        outputs=gate*hidden

        return outputs

class ConvEncoder(keras.layers.Layer):
    def __init__(self,vocab_size,hidden_dim,num_layers,n_gram=3):
        super(ConvEncoder, self).__init__()
        self.embedder=layers.Embedding(input_dim=vocab_size,output_dim=hidden_dim)
        #self.pos_enc=positional_encoding(vocab_size,hidden_dim)
        self.conlayers=[Convlayer(hidden_dim=hidden_dim,n_gram=n_gram,decode=False)\
                        for _ in range(num_layers)]

    def call(self,src):

        src_embed=self.embedder(src)

        seq_lens=src_embed.shape[1]
        hidden_dim=src_embed.shape[2]
        #src_embed+=positional_encoding(seq_lens,hidden_dim)
        for c in self.conlayers:
            src_embed=c(src_embed)

        return src_embed

#        src_embed+=self.pos_enc[:,:seq_lens,:]



def dot_product_attention(query, key, mask=None):
    score = tf.matmul(query, key, transpose_b=True)
    if mask is not None:
        score += mask

    attn = tf.nn.softmax(score, axis=-1)
    weighted_sum = tf.matmul(attn, key)

    return weighted_sum


class ConvDecoder(keras.layers.Layer):

    def __init__(self,vocab_size,hidden_dim,num_layers,n_gram=3):
        super(ConvDecoder, self).__init__()
        self.embedder=layers.Embedding(input_dim=vocab_size,output_dim=hidden_dim)
        self.hidden_dim=hidden_dim
        self.conlayers=[Convlayer(hidden_dim=hidden_dim,n_gram=n_gram,decode=True)\
                        for _ in range(num_layers)]

        #self.pos_enc = positional_encoding(vocab_size, hidden_dim)

        self.n_gram=n_gram
        self.num_layers=num_layers

    def call(self,trg,src_embed,mask,training=False):

        trg_embed=self.embedder(trg)
        batch_size = trg_embed.shape[0]
        seq_lens = trg_embed.shape[1]
        hidden_dim = trg_embed.shape[2]

        pad = tf.zeros(shape=[batch_size, self.n_gram - 1, self.hidden_dim], dtype=tf.float32)

        if training:
            for c in self.conlayers:
                trg_embed+=c(tf.concat([pad,trg_embed],axis=1))

            weighted_sum = dot_product_attention(query=trg_embed, key=src_embed, mask=mask)
            info = tf.concat([weighted_sum, trg_embed], axis=-1)
            return info

        else:
            for c in self.conlayers:
                trg_embed += c(tf.concat([pad, trg_embed], axis=1))

            weighted_sum = dot_product_attention(query=trg_embed, key=src_embed, mask=mask)
            info = tf.concat([weighted_sum, trg_embed], axis=-1)

            return info[:,-1,:]


class ConvSeq2Seq(keras.Model):
    def __init__(self,vocab_size,num_layeres,hidden_size,vocab,n_gram=3):
        super(ConvSeq2Seq, self).__init__()
        self.vocab_size=vocab_size
        self.vocab=vocab
        self.encoder=ConvEncoder(vocab_size=vocab_size,hidden_dim=hidden_size,\
                             num_layers=num_layeres,n_gram=n_gram)
        self.decoder =ConvDecoder(vocab_size=vocab_size, hidden_dim=hidden_size, \
                               num_layers=num_layeres, n_gram=n_gram)
        self.classifier=layers.Dense(units=vocab_size)
    def call(self,src,trg=None,training=False,decoding_step=20):

        assert ~(trg is None and training==True)

        batch_size=src.shape[0]


        src_encoded=self.encoder(src)

        if training:
            src_lens = src.shape[1]
            trg_lens = trg.shape[1]

            src_mask = tf.cast(tf.equal(src, 0), dtype=tf.float32) * -1e7
            trg_mask = tf.cast(tf.equal(trg, 0), dtype=tf.float32) * -1e7

            mask = tf.tile(tf.expand_dims(src_mask, axis=1), multiples=[1, trg_lens, 1]) \
                   * tf.tile(tf.expand_dims(trg_mask, axis=-1), multiples=[1, 1, src_lens])

            mask = tf.cast(tf.equal(mask, 0), tf.float32) * -1e5
            dec_state=self.decoder(trg,src_encoded,mask,training)

            return self.classifier(dec_state)[:,:-1,:]

        else:
            decoding_tokens=self.vocab["<start>"]
            decoding_tokens=tf.tile(tf.expand_dims([decoding_tokens],axis=0),multiples=[batch_size,1])

            seq_generated=copy.deepcopy(decoding_tokens)
            src_mask = tf.cast(tf.equal(src, 0), dtype=tf.float32) * -1e7

            mask = tf.cast(tf.equal(src_mask, 0), tf.float32) * -1e5
            mask=tf.expand_dims(mask,axis=1)

            for t in range(decoding_step):

                dec_info=self.decoder(seq_generated,src_encoded,mask,training)
                predict=self.classifier(dec_info)
                decoding_tokens=tf.math.argmax(predict,axis=-1,output_type=tf.int32)
                seq_generated=tf.concat([seq_generated,tf.expand_dims(decoding_tokens,axis=-1)],axis=-1)
            return seq_generated[:,1:]

# for debugging
def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("-f_name", type=str, default="./data/token_data.pickle", )
    parser.add_argument("-hidden_size", type=int, default=300)
    parser.add_argument("-num_layer", type=int, default=6)
    parser.add_argument("-epoch", type=int, default=30)

    ##
    parser.add_argument("-eval_step",type=int,default=3000)
    parser.add_argument("-eval_after",type=int,default=3000)
    parser.add_argument("-lr",type=float,default=1e-4)
    parser.add_argument("-output_dir", type=str, default="./output")
    parser.add_argument("-batch_size",type=int , default=100)


    args = parser.parse_args()


    train_ds,eval_ds, word2idx, idx2word = preprocessing(f_name=args.f_name,batch_size=args.batch_size)

    convseq2seq=ConvSeq2Seq(vocab_size=len(word2idx),hidden_size=args.hidden_size,num_layeres=args.num_layer,vocab=word2idx)


    model=get_model(convseq2seq,eval_step=args.eval_step,eval_after=args.eval_after,output_dir=args.output_dir,model_name="convseq2seq")

    model.train(args.epoch,train_ds,eval_ds,args.lr)
    #train(epoch=args.epoch,model=convseq2seq,optimizer=optimizer,train_data=train_ds)



if __name__=="__main__":
    main()
