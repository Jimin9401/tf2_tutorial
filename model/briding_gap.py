import tensorflow as tf


def gumbel_reparam(logits):
    bsz,lens,depth=tf.shape(logits)
    u=tf.random.uniform(shape=[bsz,lens,depth],minval=0,maxval=1,dtype=tf.float32)
    noise=-tf.math.log(-tf.math.log(u))

    return (noise+logits)/2




class oracle(object):
    def __init__(self,model,temperature):

        self.model=model
        self.temperature=temperature

    def get_oracle(self,inp):

        logits=self.model(inp)

        pertubed_logits=gumbel_reparam(logits)

        # pertubed_logits=tf.stop_gradient(pertubed_logits)

        return tf.argmax(pertubed_logits,-1)




