
import tensorflow as tf



class LabelSmoothing(object):
    def __init__(self,model,interpolate=1e-4):
        self.model=model
        self.interpolate=interpolate

    def smoothing_function(self,y_hat,y):

        dim = tf.shape(y_hat)[-1]
        y_one_hot=tf.one_hot(y,dim)

        y_hat=tf.nn.softmax(y_hat,-1)
        logits=y_one_hot*(1-self.interpolate)+self.interpolate*y_hat
        log_logits=tf.nn.log_softmax(logits,-1)

        self.interpolate+=1e-4

        return log_logits

