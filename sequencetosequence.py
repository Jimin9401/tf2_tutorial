from __future__ import absolute_import, division, print_function, unicode_literals

from utils import get_model
from tensorflow import keras
import tensorflow as tf
from preprocessing import preprocessing
import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()

        self.embedder = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True))

    @tf.function
    def call(self, src):
        src_embed = self.embedder(src)
        h = self.lstm(src_embed)

        return h[0], h[1:]

def dot_product_attention(query,value,mask=None):
    score = tf.matmul(a=query, b=value, transpose_b=True)

    if mask is not None:
        score += tf.cast(tf.expand_dims(mask, axis=1), tf.float32) * -1e9

    attn_score=tf.nn.softmax(score,axis=-1)

    return attn_score



class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        self.embedder = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)

        self.lstm = keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True)

        self.attention = dot_product_attention



    @tf.function
    def call(self, src_hidden, trg, mask,previous_hidden):

        trg_embed = self.embedder(trg)

        trg_embed=tf.expand_dims(trg_embed,axis=1)

        trg_hidden,hidden,cell = self.lstm(trg_embed,previous_hidden)



        attn_score = self.attention(query=trg_hidden, value=src_hidden, mask=mask)

        weighted_sum = tf.matmul(attn_score, src_hidden)


        return tf.concat([weighted_sum, trg_hidden], axis=-1),(hidden,cell)


class Seq2Seq(tf.keras.Model):
    def __init__(self, src_size, trg_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size=src_size, hidden_size=hidden_size)
        self.decoder = Decoder(vocab_size=trg_size, hidden_size=hidden_size)

        self.fc = keras.layers.Dense(units=hidden_size)
        self.classifier = keras.layers.Dense(units=trg_size)
        self.trg_size = trg_size


    @tf.function
    def call(self, src, trg, training=False):

        n_batch = trg.shape[0]
        trg_seq_lens = trg.shape[1]

        src_hidden, hc = self.encoder(src)


        src_hidden_ = self.fc(src_hidden)

        mask = tf.equal(0, src)
        previous_hidden=self.fc(tf.concat([hc[0],hc[2]],axis=-1))
        previous_cell=self.fc(tf.concat([hc[1],hc[3]],axis=-1))

        previous_=(previous_hidden,previous_cell)

        logits = None

        seq_generated=tf.expand_dims(trg[:,0],axis=1)

        if training:
            for t in range(trg_seq_lens-1):
                dec_state,previous_= self.decoder.call(trg=trg[:, t], src_hidden=src_hidden_, mask=mask,previous_hidden=previous_)

                if logits is not None:
                    logits=tf.concat([logits,self.classifier(dec_state)],axis=1)
                else:
                    logits= self.classifier(dec_state)

            return logits

        else:
            for t in range(trg_seq_lens-1):
                dec_state,previous_ = self.decoder.call(trg=seq_generated[:, t], src_hidden=src_hidden_, mask=mask,previous_hidden=previous_)

                if logits is not None:
                    logits=tf.concat([logits,self.classifier(dec_state)],axis=1)
                else:
                    logits= self.classifier(dec_state)

                seq_generated=tf.concat([seq_generated,tf.expand_dims(tf.cast(tf.argmax(logits[:, t, :], axis=-1),dtype=tf.int32),axis=-1)],axis=-1)

            return logits,seq_generated






#for debugging
def main():


    parser=argparse.ArgumentParser()
    parser.add_argument("-f_name",type=str,default="./data/token_data.pickle")

    parser.add_argument("-d_model",type=int,default=512)
    parser.add_argument("-d_ff",type=int,default=2048)
    parser.add_argument("-head",type=int,default=8)
    parser.add_argument("-epoch",type=int,default=30)
    parser.add_argument("-batch_size",type=int,default=100)
    parser.add_argument("-num_layer",type=int,default=8)
    parser.add_argument("-eval_step",type=int,default=500)
    parser.add_argument("-eval_after",type=int,default=1000)
    parser.add_argument("-output_dir", type=str, default="./output/")

    args=parser.parse_args()

    train_ds,eval_ds, word2idx, idx2word = preprocessing(f_name=args.f_name,batch_size=100)

    seq2seq = Seq2Seq(src_size=len(word2idx), trg_size=len(word2idx), hidden_size=20)

    train_model=get_model(model=seq2seq,eval_step=args.eval_step,eval_after=args.eval_after,output_dir=args.output_dir,model_name="seq2seq")

    train_model.train(epoch=10,train_data=train_ds,eval_data=eval_ds)

if __name__ =="__main__":
    main()