from __future__ import absolute_import, division, print_function, unicode_literals


import argparse
from tensorflow.keras import layers,optimizers
from tensorflow import keras
import tensorflow as tf
from preprocessing import preprocessing
import pandas as pd



class textCNN(keras.Model):
    def __init__(self, hidden_size, vocab_size, filter_size, N_gram_list, n_class):
        super(textCNN, self).__init__()
        self.embedder = layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.concat_hidden = len(N_gram_list * filter_size)

        self.fc = layers.Dense(n_class, input_shape=[self.concat_hidden, ])

        self.convlayers = []
        for kernel in N_gram_list:
            self.convlayers.append(layers.Conv1D(activation='relu', filters=filter_size, kernel_size=kernel, strides=1))

    def call(self, x):
        x_embed = self.embedder(x)
        x_convolved = [c(x_embed) for c in self.convlayers]
        x_concatenated = None

        for xc in x_convolved:
            x_mapped = tf.reduce_max(xc, axis=1)
            if x_concatenated is None:
                x_concatenated = x_mapped
            else:
                x_concatenated = tf.concat([x_concatenated, x_mapped], axis=1)

        logits=self.fc(x_concatenated)

        return logits


@tf.function(experimental_relax_shapes=True)
def compute_loss(predict, true):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(true, predict))

@tf.function(experimental_relax_shapes=True)
def compute_accuracy(predict, true):
    predictions = tf.argmax(predict, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, true), tf.float32))


@tf.function(experimental_relax_shapes=True)
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predict = model(x)
        loss = compute_loss(predict, y)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(predict, y)

    return loss, accuracy

@tf.function(experimental_relax_shapes=True)
def eval_step(model, x, y):
    predict = model.call(x)
    loss = compute_loss(predict=predict,true= y)

    accuracy = compute_accuracy(predict=predict,true= y)

    return loss, accuracy

class get_model(object):

    def __init__(self,model,eval_step,eval_after,ckpt_path,model_name):


        self.model=model
        self.eval_step=eval_step
        self.eval_after=eval_after
        self.ckpt_path=ckpt_path
        self.global_step = 0
        self.model_name=model_name
    def train(self,epoch,train_data,eval_data,lr=1e-4):
        self.get_optimizer(lr)
        self.train_log_dir = "output/summary/"+str(self.model_name)+"/train/"
        self.test_log_dir = "output/summary/"+str(self.model_name)+"/test/"

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

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

                print("epoch:", e + 1, " step:",self.global_step , " loss: {:0.5}".format(loss.numpy()),
                      " accuracy:  {:0.4}".format(accuracy.numpy()))

                if self.global_step % self.eval_step == 0 and self.global_step > self.eval_after:
                    self.eval(eval_data=eval_data)

        return loss, accuracy


    def eval(self,eval_data):

        eval_loss=[]
        eval_accuracy=[]

        for step, (x, y) in enumerate(eval_data):

            loss,accuracy=eval_step(self.model,x,y)

            eval_loss.append(loss.numpy())
            eval_accuracy.append(accuracy.numpy())

        with self.eval_summary_writer.as_default():
            tf.summary.scalar('loss', np.mean(eval_loss), step=self.eval_step)
            tf.summary.scalar('accuracy', np.mean(eval_accuracy), step=self.eval_step)

        print("evaluation loss: {:0.5}".format(np.mean(eval_loss)),
              "accuracy:  {:0.5}".format(np.mean(eval_accuracy)))
        self.save()

    def get_optimizer(self,lr):
        self.optimizer=optimizers.Adam(lr)

    def save(self):
        path=self.ckpt_path+"/_model_"+str(self.global_step)
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



#for debug

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("-f_name",type=str,default="./data/token_data.pickle")
    parser.add_argument("-d_model",type=int,default=512)
    parser.add_argument("-d_ff",type=int,default=2048)
    parser.add_argument("-head",type=int,default=8)
    parser.add_argument("-epoch",type=int,default=3)
    parser.add_argument("-batch_size",type=int,default=10)
    parser.add_argument("-num_layer",type=int,default=8)
    parser.add_argument("-eval_step",type=int,default=500)
    parser.add_argument("-eval_after",type=int,default=1000)
    parser.add_argument("-ckpt_path", type=str, default="./output/ckpt/")

    args=parser.parse_args()
    train_ds,eval_ds, word2idx, idx2word = preprocessing(f_name=args.f_name,batch_size=args.batch_size)


    cnn = textCNN(N_gram_list=[2, 3, 4], filter_size=30, hidden_size=300, vocab_size=len(word2idx), n_class=2)

    train_ds,eval_ds, word2idx, idx2word = preprocessing(f_name="./data/token_data.pickle",batch_size=args.batch_size)



    model=get_model(model=cnn,eval_after=args.eval_after,eval_step=args.eval_step,ckpt_path=args.ckpt_path,model_name="txtcn")

    model.train(epoch=args.epoch,train_data=train_ds,eval_data=eval_ds)


if __name__=="__main__":
    main()
