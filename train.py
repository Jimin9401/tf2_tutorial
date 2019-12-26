from util.data_utils.preprocessing import preprocessing
from model.transformer_lm import Transformer_LM_Oracle,Transformer_LM_Base
from util.scheduler.batch import Batchfier,Batchfier_LM
from model.wrapper import LMWrapper_Oracle,LMWrapper
from util.data_utils.preprocessing import *
from util.encoder.encoder import get_encoder



if __name__=="__main__":

    d_model=512
    h=8
    d_ff=2048
    n_layer=12
    eval_step=10000
    eval_after=1000

    # train_ds, test_ds, word2idx,idx2word=preprocessing("../data/token_data.pickle",4,0.2)

    ds=pd.read_pickle("../data/korean_lyrics_0_index.pkl")

    train_batch=Batchfier_LM(ds,8,200)

    train_batch=train_batch.tf_data()
    # test_batch=test_batch.tf_data()

    encoder=get_encoder("util/encoder/")


    vocab_size=len(encoder.encoder)+1

    #training pure ground truth
    # transformer=Transformer_LM_Base(num_layers=n_layer,d_model=d_model,d_ff=d_ff,h=h,vocab_size=vocab_size)
    #
    # model=LMWrapper(transformer,eval_step,eval_after,"transformer","log")


    #training with oracle
    transformer=Transformer_LM_Oracle(num_layers=n_layer,d_model=d_model,d_ff=d_ff,h=h,vocab_size=vocab_size)

    model=LMWrapper_Oracle(transformer,eval_step,eval_after,"transformer","log")

    model.train(5,train_batch,1e-5)









