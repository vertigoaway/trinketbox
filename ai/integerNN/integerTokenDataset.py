import torch
from torch.utils.data import Dataset
import numpy.typing as npt
import numpy as np
import math as m

#varient of sparsetokendataset, for integer tokens

class textDataset(Dataset):
    def __init__(self, inSize:int, outSize:int, tokenizedData, vocSize:int):
        # tokenizedData is expected to be a flat list/array of integer tokens
        ct : int = int(len(tokenizedData)/(inSize+outSize))-1
        self.ct = ct
        self.vocSize = vocSize
        self.tokenizedData = torch.LongTensor(tokenizedData)
        self.inSize = inSize
        self.outSize = outSize
        
    def __len__(self):
        return self.ct

    def __getitem__(self, idx):
        # return token id sequences directly
        start = self.inSize * idx
        inp = self.tokenizedData[start:start+self.inSize]
        out = self.tokenizedData[start+self.inSize:start+self.inSize+self.outSize]
        return inp, out
