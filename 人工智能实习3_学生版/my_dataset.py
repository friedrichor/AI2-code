import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tool.Global import *


class MyDataSet(Dataset):
    def __init__(self, inputs, enc_vocab2id):
        self.inputs = inputs
        self.enc_vocab2id = enc_vocab2id

    def __getitem__(self, item):
        def enc2id(enc):
            try:
                return self.enc_vocab2id[enc]
            except:
                return 3  # <?>

        inputs = self.inputs[item].split(char_space)

        inputs_vocab2id = []
        for w in inputs:
            inputs_vocab2id.append(enc2id(w))

        return inputs_vocab2id

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def collate_fn(batch):
        inputs = batch
        inputs_list = [torch.tensor(input) for input in inputs]
        inputs_pad = pad_sequence(inputs_list, batch_first=True, padding_value=0)
        return torch.tensor(inputs_pad)
