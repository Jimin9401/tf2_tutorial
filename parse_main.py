from util.encoder.encoder import get_encoder
import glob
import pandas as pd
from util.data_utils.preprocessing import load_pickle
from multiprocessing import Pool
from functools import partial
from contextlib import contextmanager
from util.encoder.multi_processing import poolcontext


def indexing_cache(f_name,encoder):

    print("start preprocess document : ",f_name)
    df=pd.read_pickle(f_name)

    paragraphs_list=df.paragraphs.tolist()

    korean_idx=[]
    for paragraph in paragraphs_list:
        korean_idx.append(encoder.encode(paragraph))


    df=pd.DataFrame({"paragraphs":korean_idx})
    df.to_pickle(f_name[:-4]+"_index.pkl")


enc=get_encoder("util/encoder")
pkl_list=glob.iglob("../data/korean*.pkl")

num_process=3

with poolcontext(processes=num_process) as pool:
    pool.map(partial(indexing_cache, encoder=enc),pkl_list )







