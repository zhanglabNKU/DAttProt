import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinSequenceDataSet(Dataset):
    def __init__(self, data, max_len=1024, shuffle=False):
        if isinstance(data, str):
            self.data = np.load(data, allow_pickle=True)
        else:
            self.data = np.array(data)
        if shuffle:
            np.random.shuffle(self.data)
        self.max_len = max_len
        self.zeros = torch.zeros(max_len, dtype=torch.long)

    def preprocess(self, x: torch.LongTensor):
        # B:24 -> D:10 or N:12
        x[x == 24] = torch.randint_like(x[x == 24], 2) * 2 + 10
        # Z:25 -> E:6 or Q:14
        x[x == 25] = torch.randint_like(x[x == 25], 2) * 8 + 6
        if len(x) > self.max_len:
            start_i = np.random.randint(len(x) - self.max_len)
            x = x[start_i: start_i + self.max_len]
        elif len(x) < self.max_len:
            x = torch.cat([x, self.zeros[:self.max_len - len(x)]], dim=-1)
        return x

    def __len__(self):
        return len(self.data)


class UnsupervisedDataset(ProteinSequenceDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        x = torch.LongTensor(self.data[index])
        l = min(len(x), self.max_len)
        x = self.preprocess(x)
        return x, l


class SupervisedDataset(ProteinSequenceDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        one_data = self.data[index]
        x = torch.LongTensor(one_data[:-1])
        x = self.preprocess(x)
        label = np.long(one_data[-1])
        return x, label
