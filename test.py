from tensorflow import keras
import tensorflow as tf
from preprocessing import preprocessing
from sequencetosequence import Seq2Seq

from tensorflow import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


f_name="./data/token_data.pickle"

train_ds,word2idx,idx2word=preprocessing(f_name=f_name,batch_size=100)


@tf.function
def compute_loss(predict,true):
    pre_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict,labels=true)
    pre_loss*=tf.cast(tf.logical_not(tf.equal(x=0,y=true)),tf.float32)

    return tf.reduce_mean(pre_loss)

@tf.function
def compute_accuracy(predict,true):
    predictions=tf.cast(tf.argmax(predict,axis=-1),tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions,true),tf.float32))



train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predict = model.call(src=x,trg=y,training=True)
        loss = compute_loss(predict=predict,true= y[:,1:])

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(predict=predict,true= y[:,1:])

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





model=Seq2Seq(src_size=len(word2idx),trg_size=len(word2idx),hidden_size=30)





train(epoch=30,model=model,optimizer=keras.optimizers.Adam(),train_data=train_ds)




