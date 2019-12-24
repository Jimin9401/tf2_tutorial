from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
from abc import *


class ModelWrapper(object):

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

    @abstractmethod
    def train(self):
        """

        will implement at distinct tasks

        """

        return NotImplementedError

    @abstractmethod
    def eval(self,eval_data):

        return NotImplementedError

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


class S2SWrapper(ModelWrapper):

    def __init__(self,model,eval_step,eval_after,model_name,output_dir):
        super(S2SWrapper,self).__init__(model,eval_step,eval_after,model_name,output_dir)

    def train(self, epoch,batchfier, lr,train_step=100,padding_index=0, batch_seqlen=512):

        keras.backend.set_learning_phase(1)

        self.get_optimizer(lr)
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(self.eval_log_dir)

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.int64),
                                      tf.TensorSpec(shape=[None, None,None], dtype=tf.float32)),
                     experimental_relax_shapes=True)
        def compute_loss(true, predict):
            batch_loss = tf.keras.losses.sparse_categorical_crossentropy(y_pred=predict, y_true=true,
                                                                         from_logits=True)
            loss_mask = tf.cast(tf.not_equal(x=true, y=0), dtype=tf.float32)

            # pre_loss*=tf.cast(mask,tf.float32)

            return tf.reduce_sum(batch_loss * loss_mask) / tf.reduce_sum(loss_mask)

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.int64),
                                      tf.TensorSpec(shape=[None, None], dtype=tf.int64)),
                     experimental_relax_shapes=True)
        def train_one_step(x, y):
            with tf.GradientTape() as tape:
                predict = self.model(x, y)
                loss = compute_loss(y, predict)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        accuracy = 0.0
        step_loss=0
        n_iter = 0

        for e in range(epoch):
            pbar = tf.keras.utils.Progbar(100)
            pbar_cnt = 0
            for batch_src,batch_trg in batchfier:
                loss=train_one_step(batch_src,batch_trg)

                pbar_cnt+=1
                step_loss+=loss
                pbar.update(pbar_cnt,[['loss',step_loss/pbar_cnt],['perplexity',tf.exp(step_loss/pbar_cnt)],['n_iter',int(n_iter)]])

                if pbar_cnt==100:
                    # with self.train_summary_writer.as_default():
                    #     tf.summary.scalar('loss', step_loss/pbar_cnt, step=self.global_step)
                    #     tf.summary.scalar('perplexity', math.exp(step_loss/pbar_cnt), step=self.global_step)
                    n_iter+=1
                    pbar = tf.keras.utils.Progbar(100)
                    pbar_cnt = 0
                    step_loss = 0

                # if self.global_step % self.eval_step == 0 and self.global_step > self.eval_after:
                #     self.eval(eval_data=eval_data)

        return loss, accuracy


class LMWrapper(ModelWrapper):

    def __init__(self,model,eval_step,eval_after,model_name,output_dir):
        super(LMWrapper,self).__init__(model,eval_step,eval_after,model_name,output_dir)

    def train(self, epoch,batchfier, lr,train_step=100,padding_index=0, batch_seqlen=512):

        keras.backend.set_learning_phase(1)

        self.get_optimizer(lr)
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(self.eval_log_dir)

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.int64),
                                      tf.TensorSpec(shape=[None, None,None], dtype=tf.float32)),
                     experimental_relax_shapes=True)
        def compute_loss(true, predict):
            batch_loss = tf.keras.losses.sparse_categorical_crossentropy(y_pred=predict, y_true=true,
                                                                         from_logits=True)
            loss_mask = tf.cast(tf.not_equal(x=true, y=0), dtype=tf.float32)

            # pre_loss*=tf.cast(mask,tf.float32)

            return tf.reduce_sum(batch_loss * loss_mask) / tf.reduce_sum(loss_mask)

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.int64),
                                      tf.TensorSpec(shape=[None, None], dtype=tf.int64)),
                     experimental_relax_shapes=True)
        def train_one_step(x, y):
            with tf.GradientTape() as tape:
                predict = self.model(x)
                loss = compute_loss(y, predict)

            grads = tape.gradient(loss, self.model.trainable_variables)

            # grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return loss

        accuracy = 0.0
        step_loss=0
        n_iter = 0

        for e in range(epoch):
            pbar = tf.keras.utils.Progbar(100)
            pbar_cnt = 0
            for batch_x,_ in batchfier:
                x = text[:, :-1]
                y = text[:, 1:]
                loss=train_one_step(x,y)

                pbar_cnt+=1
                step_loss+=loss

                pbar.update(pbar_cnt,[['loss',step_loss/pbar_cnt],['perplexity',tf.exp(step_loss/pbar_cnt)],['n_iter',int(n_iter)]])

                if pbar_cnt==100:
                    n_iter+=1
                    pbar = tf.keras.utils.Progbar(100)
                    pbar_cnt = 0
                    step_loss = 0

                #
                # if self.global_step % self.eval_step == 0 and self.global_step > self.eval_after:
                #     self.eval(eval_data=eval_data)

        return loss, accuracy

