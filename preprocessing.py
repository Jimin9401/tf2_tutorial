import pickle
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers,optimizers
from sklearn.model_selection import train_test_split
from utils import Batchfier
import pandas as pd

def preprocessing(f_name,batch_size,test_size=0.2):
    with open(f_name,'rb') as f:
        data=pickle.load(f)

    question=data["question"]
    word2idx=data["word2idx"]
    idx2word=data["idx2word"]


    X_data= [a[0:-1] for a in question]
    y_data= [[1]+a[1:] for a in question]

    X_max_seq_lens = max([len(i) for i in X_data])
    y_max_seq_lens = max([len(i) for i in y_data])

    X_train,X_eval,y_train,y_eval=train_test_split(X_data,y_data,test_size=test_size)

    X_train = tf.cast(keras.preprocessing.sequence.pad_sequences(X_train, maxlen=X_max_seq_lens), tf.int32)
    y_train = tf.cast(keras.preprocessing.sequence.pad_sequences(y_train, maxlen=y_max_seq_lens), tf.int32)
    X_eval = tf.cast(keras.preprocessing.sequence.pad_sequences(X_eval, maxlen=X_max_seq_lens), tf.int32)
    y_eval = tf.cast(keras.preprocessing.sequence.pad_sequences(y_eval, maxlen=y_max_seq_lens), tf.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    eval_ds = tf.data.Dataset.from_tensor_slices((X_eval, y_eval))

    train_ds=train_ds.shuffle(3000).batch(batch_size)
    eval_ds=eval_ds.shuffle(3000).batch(batch_size)

    # declare witg dynamic padded sequence respectly
    # train_ds = pd.DataFrame({'src':X_train,'trg':y_train})
    # eval_ds = pd.DataFrame({'src':X_eval,'trg':y_eval})
    #
    # train_batchfier = Batchfier(train_ds,batch_size=batch_size)
    # eval_batchfier=Batchfier(eval_ds,batch_size=batch_size)
    #
    # train_ds=train_batchfier.tf_data()
    # eval_ds=eval_batchfier.tf_data()
    #
    # train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # eval_ds = eval_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds,eval_ds,word2idx,idx2word

@tf.function
def f(x):
    print(x)


def main():
    preprocessing(f_name="./data/token_data.pickle",batch_size=32)


# for debugging
if __name__=="__main__":
    main()