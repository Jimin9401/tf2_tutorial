import pandas as pd
import random
import tensorflow as tf


class Batchfier:

    def __init__(self, df:pd.DataFrame,batch_size:int=32, maxlen=None, criteria:str='lens'):

        df["lens"]=[len(i) for i in df.paragraphs]

        self.df = df
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
                self.df.at[idx, 'paragraphs'] = self.df.at[idx, 'paragraphs'][:self.maxlen]

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
            yield i['paragraphs']#, i['trg']

    def tf_data(self):
        dataset = tf.data.Dataset.from_generator(self.iterator,(tf.int64,tf.int64))
        # 들어가는 feature의 개수 --> 나중에 multitask learning이나 3개 이상의 feature가 들어갈 수 있기에
        return dataset.padded_batch(batch_size=self.size,padded_shapes=([None],[None]) )
        # shape=[batch_size, ] * 위에 선언한 generator의 아웃풋 개수




class Batchfier_LM:

    def __init__(self, df:pd.DataFrame,batch_size:int=32, maxlen=None, criteria:str='lens'):

        df["lens"]=[len(i) for i in df.paragraphs]

        self.df = df
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
                self.df.at[idx, 'paragraphs'] = self.df.at[idx, 'paragraphs'][:self.maxlen]

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
            yield i['paragraphs']#, i['trg']

    def tf_data(self):
        dataset = tf.data.Dataset.from_generator(self.iterator,(tf.int64))
        # 들어가는 feature의 개수 --> 나중에 multitask learning이나 3개 이상의 feature가 들어갈 수 있기에
        return dataset.padded_batch(batch_size=self.size,padded_shapes=([None]) )
        # shape=[batch_size, ] * 위에 선언한 generator의 아웃풋 개수