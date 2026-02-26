import torch
from torch.utils.data import Dataset
import numpy.typing as npt
import numpy as np
import math as m

#varient of sparsetokendataset, for integer tokens

class textDataset(Dataset):
    def __init__(self, inSize:int, outSize:int, tokenizedData, vocSize:int)->None:
        # tokenizedData is expected to be a flat list/array of integer tokens
        self.ct = len(tokenizedData) - outSize+inSize
        self.vocSize = vocSize
        self.tokenizedData = torch.LongTensor(tokenizedData) #optimization can be made here
        self.inSize = inSize
        self.outSize = outSize
        
    def __len__(self)->int:
        return self.ct

    def __getitem__(self, idx : int):#shifting window
        # return token id sequences
        endPos : int = idx+self.inSize
        inp = self.tokenizedData[idx : endPos]
        out = self.tokenizedData[endPos : endPos + self.outSize]

        return inp, out
