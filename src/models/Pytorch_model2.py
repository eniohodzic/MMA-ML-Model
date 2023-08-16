import datetime
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Compose

sys.path.insert(0, 'C:/Users/enioh/Documents/Github/MMA-ML-Model')
from src.features.load_data import load_data

# Tensorboard Writer
writer = SummaryWriter('C:/Users/enioh/Documents/Github/MMA-ML-Model/runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Device setting to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class MMADataset3D(Dataset):
    def __init__(self, transform=None, target_transform=None):
        super(MMADataset3D, self).__init__()
        
        # Dataloading
        self.X, self.y = load_data(as_3D=True)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index,:,:]
        y = self.y[index,:,:]
        if self.transform:
            X, y = self.transform([X, y])
        return X, y

# Transformer classes for setting up padding 
class CutFiller(object):
    def __call__(self, sample) -> Any:
        X, y = sample
        label_idx = np.argwhere(~np.isnan(y))[:,0]
        
        X = X[label_idx,:]
        y = y[label_idx,:]
        
        return X, y

class ToTensor(object):
    def __call__(self, sample):
        X, y = sample
        return torch.from_numpy(X), torch.from_numpy(y)

dataset = MMADataset3D(transform=Compose([CutFiller(), 
                                          ToTensor()]))

# Scaling function for standardizing columns
class StandardScaler:
    def __call__(self, X):
        X -= torch.tensor(self.mean).to(device)
        X /= torch.tensor(self.std + 1e-7).to(device)
        return X
    def fit(self, X):
        self.mean = np.nanmean(X, 0, keepdims=True)
        self.std = np.nanstd(X, 0, keepdims=True)

# Split datasets 
train, val, test = random_split(dataset=dataset, 
                           lengths=[0.8, 0.1, 0.1])

# Standardize dataset 
sc = StandardScaler()
mat = train[0][0]
for i in range(1, len(train)):
    mat = np.vstack((mat, train[i][0]))
sc.fit(mat)

# Hyperparameters
input_size = dataset[0][0].shape[1]
hidden1_size = 500
hidden2_size = 100
hidden3_size = 25
output_size = 1
batch_size = 5
lr = 1e-3
epochs = 400

# Colating function for dataloader to handle variable sequence lengths
class PadSequence():
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = [x[1] for x in sorted_batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return sequences_padded, lengths, labels_padded

# Dataloader 
train_loader = DataLoader(dataset=train, 
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=PadSequence())

val_loader = DataLoader(dataset=val, 
                          batch_size=batch_size,
                          shuffle=False,
                          collate_fn=PadSequence())

test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=PadSequence())

loaders = {'train': train_loader,
           'val': val_loader,
           'test': test_loader}

datasets = {'train': train,
            'val': val,
            'test': test}

# Model Architecture 
class MMA_RNN(nn.Module):
    def __init__(self, input_size, hidden1_size, output_size):
        super().__init__()

        self.rnn = nn.RNN(input_size, 
                          hidden1_size, 
                          num_layers=1)
        
        self.l1 = nn.Linear(hidden1_size,
                            output_size)

    def forward(self, batch):
        X, lengths, _ = batch
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

        out, hidden = self.rnn(X_packed)
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.l1(out)
        return out

# Model
model = MMA_RNN(input_size, hidden1_size, output_size).to(device)

# Loss and optimizer 
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
        
    for phase in ['train', 'val']:
        
        # Set model to proper mode
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for i, batch in enumerate(loaders[phase]):

            # Unpack batch
            X, l, y = [b.to(device) for b in batch]
            
            # Scale X, l requires to be on cpu
            X = torch.nan_to_num(sc(X))
            l = l.to('cpu')

            with torch.set_grad_enabled(phase == 'train'):
                
                # Forward 
                outputs = model([X,l,y])

                # Loss
                loss = criterion(outputs, y)

                # Make masking array based on l
                mask = np.full((len(l), l[0], 1), False)
                for i in range(0,len(l)):
                    mask[i,0:l[i],0] = True
                mask = torch.tensor(mask)
                
                loss[~mask] = 0
                loss = loss[mask].sum() / mask.sum()

                if phase == 'train':
                    # Backwards
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            threshold = torch.tensor([0.5]).to(device)
            pred = torch.sigmoid(outputs)
            results = (pred > 0.5).float()

            running_loss += loss.item() * mask.sum()
            running_corrects += (results == y)[mask].sum().item()
            running_total += mask.sum()

        epoch_loss = running_loss / running_total
        epoch_acc = running_corrects / running_total

        print(f'Current Phase: {phase}, Epoch: {epoch+1}, Current Loss: {epoch_loss:.6f}, Current Acc: {epoch_acc:.6f}')