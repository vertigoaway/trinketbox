import torch
from torch.utils.data import Dataset
import numpy.typing as npt
import numpy as np
import math as m

class textDataset(Dataset):
    def __init__(self, inSize:int, outSize:int, tokenizedData, vocSize:int,denom=1)->None:
        # tokenizedData is expected to be a flat list/array of integer tokens
        device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ct = (len(tokenizedData) - (outSize+inSize+1))//denom
        self.vocSize = vocSize
        self.inp = []
        self.out = []
        self.inSize = inSize
        self.outSize = outSize
        for idx in range(self.ct):
            endPos : int = idx+self.inSize
            self.inp.append(torch.LongTensor(tokenizedData[idx : endPos],device=device))
            self.out.append(torch.LongTensor(tokenizedData[endPos:endPos+self.outSize],device=device))


        
    def __len__(self)->int:
        return self.ct

    def __getitem__(self, idx : int):#shifting window
        # return token id sequences


        return self.inp[idx], self.out[idx]

class lazyTextDataset(Dataset):
    def __init__(self, inSize:int, outSize:int, tokenizedData, vocSize:int)->None:
        # tokenizedData is expected to be a flat list/array of integer tokens
        self.device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ct = len(tokenizedData) - (outSize+inSize+1)
        self.vocSize = vocSize
        self.tokenizedData = torch.LongTensor(tokenizedData) #optimization can be made here
        self.inSize = inSize
        self.outSize = outSize
        
    def __len__(self)->int:
        return self.ct

    def __getitem__(self, idx : int):#shifting window
        # return token id sequences
        endPos : int = idx+self.inSize
        inp = self.tokenizedData[idx : endPos].to(self.device)
        out = self.tokenizedData[endPos : endPos + self.outSize].to(self.device)

        return inp, out