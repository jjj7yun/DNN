import numpy as np
import torch.nn as nn

class BatchSampler:
    def __init__(self, data_size, batch_size, drop_remain=False, shuffle=False):
        self.data_size = data_size
        self.batch_size = batch_size
        self.drop_remain = drop_remain
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.data_size)
        else:
            perm = range(self.data_size)

        batch_idx = []
        for idx in perm:
            batch_idx.append(idx)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []
        if len(batch_idx) > 0 and not self.drop_remain:
            yield batch_idx

    def __len__(self):
        if self.drop_remain:
            return self.data_size // self.batch_size
        else:
            return int(np.ceil(self.data_size / self.batch_size))

class DataBatcher:
    def __init__(self, *data_source, batch_size, drop_remain=False, shuffle=False):
        self.data_source = list(data_source)
        self.batch_size = batch_size
        self.drop_remain = drop_remain
        self.shuffle = shuffle

        for i, d in enumerate(self.data_source):
            if isinstance(d, list):
                self.data_source[i] = np.array(d)

        self.data_size = len(self.data_source[0])
        if len(self.data_source)> 1:
            flag = np.all([len(src) == self.data_size for src in self.data_source])
            if not flag:
                raise ValueError("All elements in data_source should have same lengths")

        self.sampler = BatchSampler(self.data_size, self.batch_size, self.drop_remain, self.shuffle)
        self.iterator = iter(self.sampler)

        self.n=0

    def __next__(self):
        batch_idx = next(self.iterator)
        batch_data = tuple([data[batch_idx] for data in self.data_source])

        if len(batch_data) == 1:
            batch_data = batch_data[0]
        return batch_data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.sampler)

def nsmc_preprocess(train, test):
    train['document'] = train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    test['document'] = test['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train['document'] = train['document'].str.replace('^ +', "")
    test['document'] = test['document'].str.replace('^ +', "")

    train['document'].replace('', np.nan, inplace=True)
    test['document'].replace('', np.nan, inplace=True)

    train = train.dropna()
    test = test.dropna()

    X, y = train['document'].tolist(), train['label'].tolist()
    X_test, y_test = test['document'].tolist(), test['label'].tolist()

    return X, X_test, y, y_test

def kor_eng_preprocess(train, test):
    
    train['ko'] = train['ko'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    test['ko'] = test['ko'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

    train['ko'] = train['ko'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    test['ko'] = test['ko'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train['ko'] = train['ko'].str.replace('^ +', "")
    test['ko'] = test['ko'].str.replace('^ +', "")

    train['ko'].replace('', np.nan, inplace=True)
    test['ko'].replace('', np.nan, inplace=True)

    train = train.dropna()
    test = test.dropna()

    X, y = train['ko'].tolist()[:40000], train['en'].tolist()[:40000]
    X_test, y_test = test['ko'].tolist()[:10000], test['en'].tolist()[:10000]

    return X, X_test, y, y_test

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)