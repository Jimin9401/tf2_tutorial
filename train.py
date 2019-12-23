from util.data_utils.preprocessing import preprocessing
from model.transformer import Transformer
from util.scheduler.batch import Batchfier
from model.wrapper import S2SWrapper
from util.data_utils.preprocessing import preprocessing




if __name__=="__main__":

    d_model=512
    h=8
    d_ff=64
    n_layer=12

    eval_step=10000
    eval_after=1000

    train_ds, test_ds, word2idx,idx2word=preprocessing("../data/token_data.pickle",4,0.2)

    train_batch=Batchfier(train_ds,4)
    test_batch=Batchfier(test_ds,4)

    train_batch=train_batch.tf_data()
    test_batch=test_batch.tf_data()


    transformer=Transformer(num_layers=n_layer,d_model=d_model,d_ff=d_ff,h=h,vocab_size=len(word2idx))

    model=S2SWrapper(transformer,eval_step,eval_after,"transformer","log")


    model.train(10,train_batch,1e-5)









