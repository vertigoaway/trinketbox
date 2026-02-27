import torch
from torch.utils.data import Dataset


import torch.nn.functional as F
class textDataset(Dataset):
    def __init__(self, inSize:int, outSize:int, tokenizedData, vocSize:int):
        self.ct = len(tokenizedData) - (inSize+outSize)
        self.vocSize = vocSize
        self.tokenizedData = torch.LongTensor(tokenizedData)
        self.inSize = inSize
        self.outSize = outSize
        return

    def __len__(self):
        return self.ct

    def __getitem__(self, x):
        #input, expected output
        return F.one_hot(self.tokenizedData[x:self.inSize+x],self.vocSize).to_sparse(),F.one_hot(self.tokenizedData[self.inSize+x:self.inSize+x+self.outSize],self.vocSize).to_sparse()