import torch
from torch.utils.data import Dataset
import numpy.typing as npt
import numpy as np
import math as m

import torch.nn.functional as F
class textDataset(Dataset):
    def __init__(self, inSize:int, outSize:int, tokenizedData, vocSize:int):
        self.ct = len(tokenizedData) - (outSize+inSize+1)
        self.vocSize = vocSize
        tokenizedData = torch.LongTensor(tokenizedData)
        inp = []
        out = []

        for x in range(self.ct):
            inp.append(F.one_hot(tokenizedData[inSize*x:inSize*(x+1)],self.vocSize).to_sparse())
            out.append(F.one_hot(tokenizedData[inSize*(x+1):(inSize*(x+1))+outSize],self.vocSize).to_sparse())
        self.inp = inp
        self.out = out
        
        return

    def __len__(self):
        return self.ct

    def __getitem__(self, idx):
        #input, expected output
        return self.inp[idx], self.out[idx]