import torch
from torch import nn
from torch.utils.data import DataLoader
import trinketbox.ai.utils.sparseNNLoops as loops
import trinketbox.ai.utils.sparseTokenDataset as sparseDataset
from griot import char
from griot import tool
import csv
from trinketbox.ai.utils.sparseTensorCollate import sparse_collate_fn as sparseCollate 


learning_rate : float = 1e-3
batch_size : int = 1
epochs : int = 10
device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inSize : int = 256
outSize : int = 1
trainingData = 'data.csv'

#0 is null
#1 is end of sent

voc : char.Vocab = char.Vocab()
voc.addCharacters(list('abcdefghijklmnopqrstuvwxyz \\;:.,[]()'))

vocSize : int = len(voc)




###load and tokenize
with open(trainingData, "r") as csvfile:
    readout = list(csv.reader(csvfile))[1:]
    out = []
    for r in readout:
        if len(r[3]) > 3:
            out.append(r[3].strip().lower())
readout = out

###create dataloaders

x = tool.flattenTokenizedLines(voc.tokenizeLines(readout))
train_dataSet = sparseDataset.textDataset(inSize=inSize,outSize=outSize,
                                 tokenizedData=x[0:len(x)//2],
                                 vocSize=vocSize)
test_dataSet = sparseDataset.textDataset(inSize=inSize,outSize=outSize,
                                tokenizedData=x[len(x)//2:],
                                vocSize=vocSize)
train_dataloader = DataLoader(train_dataSet, batch_size=batch_size, 
                              shuffle=True,
                              collate_fn=sparseCollate)
test_dataloader = DataLoader(test_dataSet, batch_size=batch_size,
                              shuffle=True,
                              collate_fn=sparseCollate)


###
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256*vocSize,vocSize*256),
            nn.Linear(vocSize*256,vocSize*128),
            nn.Linear(vocSize*128, outSize*vocSize),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loopdeloop = loops.trainAndTest(train_dataloader,
                                test_dataloader,
                                model,
                                loss_fn,
                                optimizer)

print(f"Approx training steps per epoch:{len(train_dataSet)//batch_size}")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loopdeloop.train_loop()
    loopdeloop.test_loop()

