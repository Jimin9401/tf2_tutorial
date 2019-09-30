
from seq2seq import Seq2Seq
import tensorflow as tf
from tensorflow import keras
import json
from konlpy.tag import Kkma
from itertools import chain
import pickle

import copy


with open("./token_data.pickle",'rb') as f:
    data=pickle.load(f)

question=data["question"]
answer=data["answer"]
word2idx=data['word2idx']

X_train= [a[:-1] for a in question]
y_train= [[1]+a for a in question]



X_train=tf.cast(keras.preprocessing.sequence.pad_sequences(X_train,maxlen=20),tf.int64)
y_train=tf.cast(keras.preprocessing.sequence.pad_sequences(y_train,maxlen=20),tf.int64)

train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))

train_ds=train_ds.shuffle(3000).batch(300)


for x,y in train_ds:
    print()


model=Seq2Seq(src_size=len(word2idx),trg_size=len(word2idx),hidden_size=300)

a=model.call(src=x,trg=y,training=True)



import numpy as np


rand_number=int(np.random.geometric(p=0.2)/2+1)

