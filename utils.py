
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os



#compute_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#compute_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()


# @tf.function(input_signature=(tf.TensorSpec(shape=[None,None],dtype=tf.int32),
#                               tf.TensorSpec(shape=[None,None,None],dtype=tf.float32)))
def compute_loss(true,predict):
    pre_loss=tf.keras.losses.sparse_categorical_crossentropy(y_pred=predict,y_true=true[:,1:],from_logits=True)
    mask=tf.logical_not(tf.equal(x=0, y=true[:,1:]))


    num_pad=tf.reduce_sum(tf.cast(mask,tf.float32))

    pre_loss*=tf.cast(mask,tf.float32)



    return tf.reduce_sum(pre_loss)/num_pad
#
#
# @tf.function(input_signature=(tf.TensorSpec(shape=[None,None],dtype=tf.int32),
#                               tf.TensorSpec(shape=[None,None,None],dtype=tf.float32)))
def compute_accuracy(true,predict):
    predictions=tf.cast(tf.argmax(predict,axis=-1),tf.int32)

#    mask = tf.logical_not(tf.equal(x=0, y=true[:, 1:]))

    return tf.reduce_mean(tf.cast(tf.equal(predictions,true[:,1:]),tf.float32))

#
# # @tf.function(input_signature=(tf.TensorSpec(shape=[None,None],dtype=tf.int32),
# #                               tf.TensorSpec(shape=[None,None],dtype=tf.int32)))

@tf.function()
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predict = model(x,y,True)
        loss = compute_loss(y,predict)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy=compute_accuracy(y,predict)

    return loss, accuracy


# @tf.function(input_signature=(tf.TensorSpec(shape=[None,None],dtype=tf.int32),
#                               tf.TensorSpec(shape=[None,None],dtype=tf.int32)))

@tf.function
def eval_step(model, x, y):
    predict = model.call(x,y,True)
    loss = compute_loss(y[:,1:],predict)

    accuracy = compute_accuracy(predict=predict,true= y)

    return loss, accuracy




class get_model(object):

    def __init__(self,model,eval_step,eval_after,model_name,output_dir):
        self.model=model
        self.eval_step=eval_step
        self.eval_after=eval_after
        self.output_dir=output_dir
        self.global_step = 0
        self.model_name=model_name

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.train_log_dir = self.output_dir+"/summary/"+str(self.model_name)+"/train/"
        self.eval_log_dir = self.output_dir+"/summary/"+str(self.model_name)+"/eval/"

    def train(self,epoch,train_data,eval_data,lr=1e-4):
        tf.keras.backend.set_learning_phase(1)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

        self.get_optimizer(lr)

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(self.eval_log_dir)

        print("start train!!")
        loss = 0.0
        accuracy = 0.0

        for e in range(epoch):

            for x, y in train_data:
                self.global_step += 1
                loss, accuracy = train_one_step(self.model, self.optimizer, x, y)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.global_step)
                    tf.summary.scalar('accuracy', accuracy, step=self.global_step)

                print("epoch:", e + 1, " step:",self.global_step , " loss: {:0.5}".format(loss.numpy())," perplexity: {:0.5}".format(tf.exp(loss.numpy())),
                      " accuracy:  {:0.4}".format(accuracy.numpy()))

                if self.global_step % self.eval_step == 0 and self.global_step > self.eval_after:
                    self.eval(eval_data=eval_data)

        return loss, accuracy


    def eval(self,eval_data):
        tf.keras.backend.set_learning_phase(0)
        eval_loss=[]
        eval_accuracy=[]

        for step, (x, y) in enumerate(eval_data):

            loss,accuracy=eval_step(self.model,x,y)

            eval_loss.append(loss.numpy())
            eval_accuracy.append(accuracy.numpy())

        with self.eval_summary_writer.as_default():
            tf.summary.scalar('eval_loss', np.mean(eval_loss), step=self.eval_step)
            tf.summary.scalar('eval_accuracy', np.mean(eval_accuracy), step=self.eval_step)

        print("evaluation loss: {:0.5}".format(np.mean(eval_loss)),
              " perplexity: {:0.5}".format(tf.exp(np.mean(eval_loss))),
              "accuracy:  {:0.5}".format(np.mean(eval_accuracy)))

        self.save()

    def get_optimizer(self,lr):
        self.optimizer=keras.optimizers.Adam(lr)

    def save(self):
        path=self.output_dir+"/ckpt/model_"+str(self.global_step)
        self.model.save_weights(path)
        print("saved at ",path)

    def restore(self,ckpt_dir,latest_ckpt=True):

        if latest_ckpt:
            latest=tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
            print(latest)
            self.model.load_weights(latest)
        else:
            self.model.load_weights(latest_ckpt)
        print("loaded from  ",latest)




class Batchfier:
    def __init__(self, df:pd.DataFrame,batch_size:int=32, maxlen=None, criteria:str='lens'):
        self.df = df
        self.df["lens"]=[len(i) for i in df["src"]]

        self.maxlen = maxlen
        self.size = batch_size
        self.num_buckets = len(df) //batch_size + 1
        self.criteria = criteria
        if maxlen:
            self.truncate_text()

        # self.size = len(self.df) / num_buckets
        self.sort(criteria)
        self.shuffle()

    def __len__(self):
        return len(self.df)

    def truncate_text(self):
        for idx, i in self.df.iterrows():
            if i['lens'] > self.maxlen:
                self.df.at[idx, 'src'] = self.df.at[idx, 'src'][:self.maxlen]
                self.df.at[idx, 'trg'] = self.df.at[idx, 'trg'][:self.maxlen]
                self.df.at[idx, 'lens'] = self.maxlen

    def shuffle(self):
        dfs = []
        for bucket in range(self.num_buckets):
            new_df = self.df.loc[bucket * self.size: (bucket + 1) * self.size - 1]
            new_df = new_df.sample(frac=1).reset_index(drop=True)
            dfs.append(new_df)
        random.shuffle(dfs)
        df = pd.concat(dfs)
        self.df = df

    def sort(self,criteria):
        self.df = self.df.sort_values(criteria).reset_index(drop=True)

    def iterator(self):
        for _, i in self.df.iterrows():
            yield i['src'], i['trg'] # 나와야 하는 variable의 개수

    def tf_data(self):
        dataset = tf.data.Dataset.from_generator(self.iterator,(tf.int32,tf.int32))
        # 들어가는 feature의 개수 --> 나중에 multitask learning이나 3개 이상의 feature가 들어갈 수 있기에

        return dataset.padded_batch(batch_size=self.size,padded_shapes=([None],[None]))
        # shape=[batch_size, ] * 위에 선언한 generator의 아웃풋 개수
