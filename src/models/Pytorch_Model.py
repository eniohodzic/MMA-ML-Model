import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import *

sys.path.insert(0, 'C:/Users/enioh/Documents/Github/MMA-ML-Model')
from src.features.load_data import load_data

# Device setting to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset loading class
class MMADataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        super(MMADataset, self).__init__()
        self.X, self.y = load_data(as_numpy=True)
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

# Hyperparameters
input_size = 1083
hidden_size = 100
num_classes = 2
batch_size = 100
lr = 0.001
epochs = 100

# Dataset loading 
dataset = MMADataset(transform=transforms.ToTensor(), 
                     target_transform=transforms.ToTensor())

train, test = random_split(dataset=dataset, 
                           lengths=[0.8, 0.2])

train_loader = DataLoader(dataset=train, 
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=False)

# Network architecture
class MMANet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MMANet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

# Model init
model = MMANet(input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (X, y) in enumerate(train_loader):

        X = X.to(device)
        y = y.to(device)

        # Forward
        outputs = model(X)

        # Loss
        loss = criterion(outputs, y)

        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Current Epoch: {epoch+1}/{epochs}, Step {i+1}/{n_total_steps}, loss = {loss.item():.8f}')

# Testing 
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for (X, y) in test_loader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)

        _, pred = torch.max(outputs, 1)
        n_samples += y.shape[0]
        n_correct += (pred == y).sum().item

    acc = 100 * (n_correct/n_samples)
    print(f'Accuracy = {acc}')


