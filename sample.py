from model.wrapper import S2SWrapper_Oracle
from model.transformer_lm import Transformer_LM_Oracle
from util.encoder.encoder import get_encoder



input_text=""



enc_path="util/encoder/"

d_model = 512
h = 8
d_ff = 2048
n_layer = 12
eval_step = 10000
eval_after = 1000
vocab_size=len(enc.encoder)+1
enc=get_encoder(enc_path)



indexed=enc.encode(input_text)

length=

transformer=Transformer_LM_Oracle(n_layer,d_model,d_ff,h,vocab_size)



transformer.restore("")
