import pickle
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers,optimizers
from sklearn.model_selection import train_test_split
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


    train_ds = pd.DataFrame({'src':X_train,'trg':y_train})
    eval_ds = pd.DataFrame({'src':X_eval,'trg':y_eval})


    return train_ds,eval_ds,word2idx,idx2word





def main():
    preprocessing(f_name="./data/token_data.pickle",batch_size=32)


# for debugging
if __name__=="__main__":
    main()