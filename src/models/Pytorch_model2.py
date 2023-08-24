import copy
import datetime
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from mlflow.tracking import MlflowClient
from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset, random_split
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
    def __init__(self, transform=None, target_transform=None, train=True):
        super(MMADataset3D, self).__init__()
        
        # Dataloading
        if train: 
            self.X, self.y, _, _ = load_data(as_3D=True)
        else:
            _, _, self.X, self.y = load_data(as_3D=True)
        
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

# Scaling function for standardizing columns
class StandardScaler:
    def __call__(self, X):
        X -= torch.tensor(self.mean).to(device)
        X /= torch.tensor(self.std + 1e-7).to(device)
        return X
    def fit(self, X):
        self.mean = np.nanmean(X, 0, keepdims=True)
        self.std = np.nanstd(X, 0, keepdims=True)

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
    
# Model Architecture 
class MMA_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers)
        
        self.drop = nn.Dropout(p=dropout)
        
        self.l1 = nn.Linear(hidden_size,
                            output_size)

    def forward(self, batch):
        X, lengths, _ = batch
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)

        out, hidden = self.rnn(X_packed, (h0, c0))
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.drop(out)
        out = self.l1(out)
        return out

# Training function
def train_model(config, train, val):

    # Model Initialization 
    model = MMA_RNN(input_size=input_size, 
                        hidden_size=config['hidden_size'], 
                        num_layers=config['num_layers'], 
                        dropout=config['dropout'],
                        output_size=output_size).to(device)

    # Standardize dataset 
    sc = StandardScaler()
    mat = train[0][0]
    for i in range(1, len(train)):
        mat = np.vstack((mat, train[i][0]))
    sc.fit(mat)

    # Dataloaders 
    train_loader = DataLoader(dataset=train, 
                            batch_size=config['batch_size'],
                            shuffle=True,
                            collate_fn=PadSequence())

    val_loader = DataLoader(dataset=val, 
                            batch_size=config['batch_size'],
                            shuffle=False,
                            collate_fn=PadSequence())

    loaders = {'train': train_loader,
                'val': val_loader}

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    # Training Loop
    while True:
      
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

                    # Loss, Raw per individual sequence 
                    loss = criterion(outputs, y)

                    # Make masking array based on l
                    mask = np.full((len(l), l[0], 1), False)
                    for i in range(0,len(l)):
                        mask[i,0:l[i],0] = True
                    mask = torch.tensor(mask)
                    
                    # Mean Loss Computed
                    loss[~mask] = 0
                    loss = loss[mask].sum() / mask.sum()

                    # Backwards
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Predicting values 
                pred = torch.sigmoid(outputs)
                results = (pred > 0.5).float()

                running_loss += loss.item() * mask.sum()
                running_corrects += (results == y)[mask].sum().item()
                running_total += mask.sum()

            # Compute Epoch wide loss and accuracy
            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects / running_total
            # If phase is validation, add to trial length list 
            if phase == 'val':
                # Save Epoch loss and wts for checkpoint
                checkpoint_data = {'wts': model.state_dict()}
                checkpoint = Checkpoint.from_dict(checkpoint_data)
                session.report({'loss': epoch_loss.item(),
                                'acc': epoch_acc.item()}, checkpoint=checkpoint)

def test_model(model, test_loader, sc):

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        model.eval()

        for batch in test_loader:

            # Unpack batch
            X, l, y = [b.to(device) for b in batch]
            
            # Scale X, l requires to be on cpu
            X = torch.nan_to_num(sc(X))
            l = l.to('cpu')

            # Forward 
            outputs = model([X,l,y])

            # Loss, Raw per individual sequence 
            loss = criterion(outputs, y)

            # Make masking array based on l
            mask = np.full((len(l), l[0], 1), False)
            for i in range(0,len(l)):
                mask[i,0:l[i],0] = True
            mask = torch.tensor(mask)
            
            # Mean Loss Computed
            loss[~mask] = 0
            loss = loss[mask].sum() / mask.sum()
            
            # Convert to sigmoid layer, threshold at 0.5 and convert to 0,1
            pred = torch.sigmoid(outputs)
            results = (pred > 0.5).float()

            running_loss += loss.item() * mask.sum()
            running_total += mask.sum()
            running_corrects += (results == y)[mask].sum().item()

        loss = running_loss / running_total
        acc = running_corrects / running_total
        print(f'Accuracy = {acc}, Loss = {loss}')

if __name__=='__main__':

    # Mlflow setup
    client = MlflowClient()
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    exp_name = 'RNN-'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = client.create_experiment(name=exp_name)

    # Dataset loading
    train = MMADataset3D(train=True,
                        transform=Compose([CutFiller(), 
                                            ToTensor()]))
    
    test = MMADataset3D(train=False,
                        transform=Compose([CutFiller(), 
                                            ToTensor()]))

    # Split datasets 
    val, test = random_split(dataset=test, 
                            lengths=[0.9, 0.1])

    # Search space for hyperparameters and ML Flow 
    search_space = {'mlflow_experiment_id': experiment_id,
                    'lr': tune.loguniform(1e-3, 1e-2),
                    'hidden_size': tune.qrandint(600,800,1),
                    'batch_size': tune.choice([2,4,8]),
                    'num_layers': tune.choice([1]),
                    'dropout': tune.uniform(0.2,0.5)
                    }
    # search_space = {'mlflow_experiment_id': experiment_id,
    #                 'lr': 5e-2,
    #                 'hidden_size': 55,
    #                 'batch_size': 2,
    #                 'num_layers': tune.choice([1])
    #                 }

    # Hyperparameters
    input_size = train[0][0].shape[1]
    output_size = 1

    # Hyperparameter search 
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable=train_model, train=train, val=val),
            resources={'cpu': 4, 'gpu': 0.25}
        ),
        tune_config=tune.TuneConfig(
            num_samples=50,
            scheduler=ASHAScheduler(
                max_t=200,
                grace_period=3
            ),
            metric='loss',
            mode='min',
            search_alg=algo,
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": 200},
            callbacks=[MLflowLoggerCallback(
                experiment_name=exp_name)],
        ),
        param_space=search_space
    )

    results = tuner.fit()
    best_model = results.get_best_result(scope='all')
    best_config = best_model.config
    print('Best config is: ', best_model.config)
    
    best_checkpoint = best_model.get_best_checkpoint('loss', 'min')
    checkpoint_data = best_checkpoint.to_dict()

    # Init best config model with appropriate weights 
    config = best_model.config
    best_model = MMA_RNN(input_size=input_size, 
                            hidden_size=best_config['hidden_size'], 
                            num_layers=best_config['num_layers'], 
                            dropout=best_config['dropout'],
                            output_size=output_size).to(device)
    best_model.load_state_dict(checkpoint_data['wts'])

    # Standardize dataset 
    sc = StandardScaler()
    mat = train[0][0]
    for i in range(1, len(train)):
        mat = np.vstack((mat, train[i][0]))
    sc.fit(mat)

    test_loader = DataLoader(dataset=test,
                    batch_size=best_config['batch_size'],
                    shuffle=False,
                    collate_fn=PadSequence())

    test_model(best_model, test_loader, sc)

# Tensorboard
# writer.add_graph(model, iter(train_loader).__next__()[0][0].to(device))
# writer.close()

