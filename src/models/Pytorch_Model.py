import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.insert(0, 'C:/Users/enioh/Documents/Github/MMA-ML-Model')
from src.features.load_data import load_data

writer = SummaryWriter('C:/Users/enioh/Documents/Github/MMA-ML-Model/runs')

# Device setting to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset loading class
class MMADataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        super(MMADataset, self).__init__()
        
        # Dataloading
        self.X, self.y = load_data(as_numpy=True)
        self.X = np.nan_to_num(self.X)
        self.X, self.y = torch.from_numpy(self.X), torch.from_numpy(self.y)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index,:]
        y = self.y[index]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y

class StandardScaler:
    def __call__(self, X):
        X -= self.mean
        X /= (self.std + 1e-7)
        return X
    def fit(self, X):
        self.mean = X.mean(0, keepdim=True)
        self.std = X.std(0, unbiased=False, keepdim=True)


# Hyperparameters
input_size = 1083
hidden1_size = 500
hidden2_size = 100
hidden3_size = 25
output_size = 1
batch_size = 50
lr = 1e-3
epochs = 400

# Dataset loading and Fitting Scaler 
dataset = MMADataset()

train, val, test = random_split(dataset=dataset, 
                           lengths=[0.8, 0.1, 0.1])

sc = StandardScaler()
sc.fit(train[:][0])

train_loader = DataLoader(dataset=train, 
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(dataset=val, 
                          batch_size=batch_size,
                          shuffle=False)

test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=False)

loaders = {'train': train_loader,
           'val': val_loader,
           'test': test_loader}

datasets = {'train': train,
            'val': val,
            'test': test}

# Network architecture
class MMANet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super(MMANet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1_size)
        self.lrelu = nn.LeakyReLU(0.1)
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        self.l3 = nn.Linear(hidden2_size, hidden3_size)
        self.drop1 = nn.Dropout(p=0.2)
        self.l4 = nn.Linear(hidden3_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.lrelu(out)
        out = self.l2(out)
        out = self.lrelu(out)
        out = self.l3(out)
        out = self.lrelu(out)
        out = self.drop1(out)
        out = self.l4(out)
        return out

# Model init
model = MMANet(input_size=input_size,
                hidden1_size=hidden1_size,
                hidden2_size=hidden2_size,
                hidden3_size=hidden3_size,
                output_size=output_size).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Tensorboard
writer.add_graph(model, iter(train_loader).__next__()[0].to(device))
writer.close()

# Training
n_total_steps = len(train_loader)
for epoch in range(epochs):

    for phase in ['train', 'val']:
        # Set model to proper mode
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        
        for i, (X, y) in enumerate(loaders[phase]):
            # Scale X
            X = sc(X).to(device)
            y = y.to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # Forward
                outputs = model(X)

                # Loss
                loss = criterion(outputs, y)

                if phase == 'train':
                    # Backwards
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            threshold = torch.tensor([0.5]).to(device)
            pred = torch.sigmoid(outputs)
            results = (pred > 0.5).float()
            
            running_loss += loss.item() * X.size(0)
            running_corrects += (results == y).sum().item()

        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects / len(datasets[phase])

        print(f'Current Phase: {phase}, Epoch: {epoch+1}, Current Loss: {epoch_loss:.6f}, Current Acc: {epoch_acc:.6f}')
        
        writer.add_scalar(f'{phase} Loss', epoch_loss, epoch + 1)
        writer.add_scalar(f'{phase} Acc', epoch_acc, epoch + 1)

# Testing 
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for (X, y) in loaders['test']:

        X = sc(X).to(device)
        y = y.to(device)

        # Model outputs
        outputs = model(X)
        
        # Convert to sigmoid layer, threshold at 0.5 and convert to 0,1
        threshold = torch.tensor([0.5]).to(device)
        pred = torch.sigmoid(outputs)
        results = (pred > 0.5).float()

        n_samples += y.shape[0]
        n_correct += (results == y).sum().item()

    acc = 100 * (n_correct/n_samples)
    print(f'Accuracy = {acc}')


