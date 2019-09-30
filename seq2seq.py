
from tensorflow import keras
import tensorflow as tf
import copy
from utils import compute_accuracy,compute_loss
###
import pickle
import copy






with open("./token_data.pickle",'rb') as f:
    data=pickle.load(f)

question=data["question"]
answer=data["answer"]
word2idx=data['word2idx']

X_train= [a[:-1] for a in question]
y_train= [[1]+a for a in question]


X_max_seq_lens= max([len(i) for i in X_train])
y_max_seq_lens= max([len(i) for i in y_train])


X_train=tf.cast(keras.preprocessing.sequence.pad_sequences(X_train,maxlen=X_max_seq_lens,padding='post'),tf.int32)
y_train=tf.cast(keras.preprocessing.sequence.pad_sequences(y_train,maxlen=y_max_seq_lens,padding='post'),tf.int32)

train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))

train_ds=train_ds.shuffle(3000).batch(100)


for x,y in train_ds:
    print()

###


class Encoder(keras.Model):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()

        self.embedder = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True))

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


class Decoder(keras.Model):
    def __init__(self, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        self.embedder = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.lstm = keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True)

        self.attention = dot_product_attention

    def call(self, src_hidden, trg, mask,previous_hidden):

        trg_embed = self.embedder(trg)

        trg_embed=tf.expand_dims(trg_embed,axis=1)

        trg_hidden,hidden,cell = self.lstm(trg_embed,previous_hidden)



        attn_score = self.attention(query=trg_hidden, value=src_hidden, mask=mask)

        weighted_sum = tf.matmul(attn_score, src_hidden)


        return tf.concat([weighted_sum, trg_hidden], axis=-1),(hidden,cell)


class Seq2Seq(keras.Model):
    def __init__(self, src_size, trg_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size=src_size, hidden_size=hidden_size)
        self.decoder = Decoder(vocab_size=trg_size, hidden_size=hidden_size)

        self.fc = keras.layers.Dense(units=hidden_size)
        self.classifier = keras.layers.Dense(units=trg_size)
        self.trg_size = trg_size

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

                seq_generated=tf.concat([seq_generated,tf.expand_dims(tf.argmax(logits[:, t, :], axis=-1),axis=-1)],axis=-1)

            return logits,seq_generated




#####################################



model=Seq2Seq(src_size=len(word2idx),trg_size=len(word2idx),hidden_size=300)


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predict = model(x,y,training=True)
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
            if((step+1)%10==0):
                print("epoch:",e+1," step:",step+1," loss: {:0.5}".format(loss.numpy())," accuracy:  {:0.4}".format(accuracy.numpy()))
        print()
        print("epoch:",e+1," loss: {:0.5}".format(loss.numpy())," accuracy:  {:0.4}".format(accuracy.numpy()))
        print()
    return loss,accuracy



train(epoch=1,model=model,optimizer=keras.optimizers.Adam(),train_data=train_ds)



