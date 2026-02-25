import torch
from torch import nn
from torch.utils.data import DataLoader



batch_size : int = 12


class trainAndTest():
    def __init__(self,train_dataloader,test_dataloader,model,loss_fn,optimizer):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)


    def train_loop(self):
        dataloader = self.train_dataloader
        model = self.model
        loss_fn = self.loss_fn 
        optimizer = self.optimizer

        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            #X is the input 
            #y is the intended output
            X, y = X.to(self.device).float(), y.to(self.device).float()

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}")


    def test_loop(self):
        dataloader = self.test_dataloader
        model = self.model
        loss_fn = self.loss_fn 
        optimizer = self.optimizer
        
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device).float(), y.to(self.device).float()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        print(f"Avg loss: {test_loss:>8f} \n")


