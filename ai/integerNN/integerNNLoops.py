import torch
from torch import nn
from torch.utils.data import DataLoader

# nnloops adapted for integer token datasets

batch_size : int = 8

class trainAndTest():
    def __init__(self,train_dataloader,test_dataloader,model,loss_fn,optimizer)-> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)

    def train_loop(self) -> None:
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
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)  
            # Reshape pred from (batch_size, outSize, vocSize) to (batch_size*outSize, vocSize)
            # Reshape y from (batch_size, outSize) to (batch_size*outSize)
            pred_flat = pred.view(-1, pred.shape[-1])
            y_flat = y.view(-1)
            loss = loss_fn(pred_flat, y_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"loss: {loss.item():>7f}")

    def test_loop(self) -> None:
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
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                # Reshape for loss calculation
                pred_flat = pred.view(-1, pred.shape[-1])
                y_flat = y.view(-1)
                test_loss += loss_fn(pred_flat, y_flat).item()
                correct += (pred_flat.argmax(1) == y_flat).type(torch.float).sum().item()

        test_loss /= num_batches if num_batches>0 else 1
        print(f"Avg loss: {test_loss:>8f}\n")



