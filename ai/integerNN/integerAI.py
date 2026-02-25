import torch
from torch import nn
from torch.utils.data import DataLoader
import integerNNLoops as loops
import integerTokenDataset as sparseDataset
import charTokenizer as cT
import csv
learning_rate : float = 1e-2
batch_size : int = 8
epochs : int = 10
device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inSize : int = 2048
outSize : int = 3
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
                              shuffle=True,)
test_dataloader = DataLoader(test_dataSet, batch_size=batch_size,
                              shuffle=True,)


### LSTM Architecture Parameters
embedding_dim : int = 128  # Embedding dimension for vocabulary
hidden_size : int = 384    # Hidden size for each LSTM layer
num_layers : int = 2       # Number of LSTM layers
dropout : float = 0.2      # Dropout for regularization between LSTM layers

###
class NeuralNetwork(nn.Module):
    def __init__(self, vocSize, inSize, outSize, embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocSize, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, outSize * vocSize)
        self.outSize = outSize
        self.vocSize = vocSize
        
    def forward(self, x):
        # x shape: (batch_size, inSize)
        x = self.embedding(x)  # (batch_size, inSize, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, inSize, hidden_size)
        # Use the last timestep output
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        logits = self.linear(x)  # (batch_size, outSize * vocSize)
        logits = logits.view(-1, self.vocSize)  # Reshape for loss function
        return logits

model = NeuralNetwork(vocSize=vocSize, inSize=inSize, outSize=outSize, 
                      embedding_dim=embedding_dim, hidden_size=hidden_size, 
                      num_layers=num_layers, dropout=dropout).to(device)
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
