import torch
from torch import nn
from torch.utils.data import DataLoader
import ai.utils.sparseNNLoops as loops
import ai.utils.sparseTokenDataset as sparseDataset
import ai.utils.charTokenizer as cT
import csv
from ai.utils.sparseTensorCollate import sparse_collate_fn as sparseCollate 
learning_rate : float = 1e-3
batch_size : int = 1
epochs : int = 10
device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inSize : int = 256
outSize : int = 1
#0 is null
#1 is end of sent
#2 is link tok (unused fo now)
#3 is secret,,,, :3
voc : dict[str,int]= {'a':4,'b':5,'c':6,'d':7,'e':8,'f':9,'g':10,
       'h':11,'i':12,'j':13,'k':14,'l':15,'m':16,
       'n':17,'o':18,'p':19,'q':20,'r':21,'s':22,
       't':23,'u':24,'v':25,'w':26,'x':27,'y':28,
       'z':29,' ':30,'.':31,',':32,'\'':33,'/':34,
       '\"':35,':':36,';':37,'1':38,'2':39,'3':40,
       '4':41,'5':42,'6':43,'7':44,'8':45,'9':46,
       '0':47}
vocSize : int = len(voc)+4 #acct for special toks

###load and tokenize
file : str = "../data/data.csv"
csvfile = open(file, "r")
    
readout = list(csv.reader(csvfile))
goongagas = []
for r in readout:
    goongagas.append(r[3])
readout = goongagas
goongagas = None

###create dataloaders
x = cT.dynamicTokenize(readout,tokDict=voc)

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


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loopdeloop.train_loop()
    loopdeloop.test_loop()

print(model(test_dataSet[1][0]))
print("Done!")
