from layers.layer import Embedding
from tensorflow import keras
from layers.layers_util import dot_product_attention


class Encoder(keras.Model):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()

        self.embedder = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size)
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size, return_sequences=True, return_state=True))

    def call(self, src):
        src_embed = self.embedder(src)
        h = self.lstm(src_embed)

        return h[0], h[1:]


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

        for t in range(trg_seq_lens-1):
            dec_state,previous_= self.decoder.call(trg=trg[:, t], src_hidden=src_hidden_, mask=mask,previous_hidden=previous_)

            if logits is not None:
                logits=tf.concat([logits,self.classifier(dec_state)],axis=1)
            else:
                logits= self.classifier(dec_state)

        return logits
