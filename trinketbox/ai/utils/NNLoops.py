import torch
from torch.nn.functional import one_hot

class trainAndTest():
    def __init__(self,train_dataloader,test_dataloader,model,loss_fn,optimizer,)-> None:
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
        model.train()
        stepCt = len(dataloader)
        for batch, (X, y) in enumerate(dataloader):
            #X is the input 
            #y is the intended output
            X, y_flat = X.to(self.device), y.to(self.device).view(-1)
            # Reshape y from (batch_size, outSize, vocSize) to (batch_size*outSize)
            pred = model(X)  
            pred = pred.view(-1, pred.shape[-1])            # Reshape pred from (batch_size, outSize, vocSize) to (batch_size*outSize, vocSize)

            loss = loss_fn(pred, y_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"step {batch}/{stepCt} loss: {loss.item():>7f}")

    def test_loop(self) -> None:
        dataloader = self.test_dataloader
        model = self.model
        loss_fn = self.loss_fn 
        
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        model.eval()
        num_batches = len(dataloader)
        test_loss, correct = 0.0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device).view(-1)
                pred = model(X)
                pred = pred.view(-1, pred.shape[-1])
                # Reshape for loss calculation
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches if num_batches>0 else 1
        print(f"Avg loss: {test_loss:>8f}\n")