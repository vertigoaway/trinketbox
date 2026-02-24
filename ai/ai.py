import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import nnLoops as loops
import numpy as np
learning_rate = 1e-3
batch_size = 12
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
def preprocess(x, y):
    return x.to(device), y.to(device)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))



train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
test_dataloader = WrappedDataLoader(test_dataloader, preprocess) #yeah this is getting ugly

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024), #in, out
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loopdeloop = loops.trainAndTest(train_dataloader,test_dataloader,model,loss_fn,optimizer)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loopdeloop.train_loop()
    loopdeloop.test_loop()
print("Done!")
