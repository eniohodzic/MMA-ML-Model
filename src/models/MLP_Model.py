"""
Standard MLP model 
"""

import datetime
import sys
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

writer = SummaryWriter('C:/Users/enioh/Documents/Github/MMA-ML-Model/runs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Device setting to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# # Custom Dataset loading class
# class MMADataset(Dataset):
#     def __init__(self, transform=None, target_transform=None):
#         super(MMADataset, self).__init__()
        
#         # Dataloading
#         self.X, self.y = load_data(as_3D=True)
#         self.X = np.nan_to_num(self.X)
#         self.X, self.y = torch.from_numpy(self.X), torch.from_numpy(self.y)
        
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, index):
#         X = self.X[index,:]
#         y = self.y[index]
#         if self.transform:
#             X = self.transform(X)
#         if self.target_transform:
#             y = self.target_transform(y)
#         return X, y"""

class MMADataset3D(Dataset):
    def __init__(self, transform=None, target_transform=None, split='train'):
        super(MMADataset3D, self).__init__()
        
        # Dataloading
        self.X, self.y, _ = load_data(split=split)
        
        # Converting to 2D 
        self.X = np.reshape(self.X, (-1, self.X.shape[2]))
        self.y = np.reshape(self.y, (-1, self.y.shape[2]))

        # Remove rows with all nans 
        idxs = np.argwhere(np.isfinite(self.y))[:,0]
        self.X = self.X[idxs]
        self.y = self.y[idxs]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = self.X[index,:]
        y = self.y[index,:]
        if self.transform:
            X, y = self.transform([X, y])
        return X, y

class ToTensor(object):
    def __call__(self, sample):
        X, y = sample
        return torch.from_numpy(X), torch.from_numpy(y)

class StandardScaler:
    def __call__(self, X):
        X -= torch.tensor(self.mean).to(device)
        X /= torch.tensor(self.std + 1e-7).to(device)
        return X
    def fit(self, X):
        self.mean = np.nanmean(X, 0, keepdims=True)
        self.std = np.nanstd(X, 0, keepdims=True)

class MinMaxScaler:
    def __call__(self, X):
        X = (X - self.min) / (self.max - self.min)
        return X
    def fit(self, X):
        self.max = X.max(0, keepdim=True)[0]
        self.min = X.min(0, keepdim=True)[0]

# Network architecture
class MMANet(nn.Module):
    def __init__(self, input_size, hidden1_size, 
                 hidden2_size, 
                #  hidden3_size, 
                 alpha, dropout, output_size):
        super(MMANet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1_size)
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        # self.l3 = nn.Linear(hidden2_size, hidden3_size)
        self.l4 = nn.Linear(hidden2_size, output_size)
        self.lrelu = nn.LeakyReLU(alpha)
        self.drop1 = nn.Dropout(p=dropout)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.lrelu(out)
        out = self.l2(out)
        out = self.lrelu(out)
        # out = self.l3(out)
        # out = self.lrelu(out)
        out = self.drop1(out)
        out = self.l4(out)
        return out

def train_model(config, train, val):

    # Model init
    model = MMANet(input_size=input_size,
                   hidden1_size=config['hidden1_size'],
                   hidden2_size=config['hidden2_size'],
                #    hidden3_size=config['hidden3_size'],
                   alpha=config['alpha'],
                   dropout=config['dropout'],
                   output_size=output_size).to(device)
    
    # Standardize dataset 
    sc = StandardScaler()
    sc.fit(train[:][0])

    train_loader = DataLoader(dataset=train, 
                              batch_size=config['batch_size'],
                              shuffle=True)

    val_loader = DataLoader(dataset=val, 
                            batch_size=config['batch_size'],
                            shuffle=False)
    
    loaders = {'train': train_loader,
                'val': val_loader}

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

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

            for i, (X, y) in enumerate(loaders[phase]):

                X = X.to(device)
                X = torch.nan_to_num(sc(X))
                y = y.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    
                    # Forward 
                    outputs = model(X)

                    # Loss, Raw per individual sequence 
                    loss = criterion(outputs, y)

                    # Backwards
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                pred = torch.sigmoid(outputs)
                results = (pred > 0.5).float()
                running_total += X.size(0)
                running_loss += loss.item() * X.size(0)
                running_corrects += (results == y).sum().item()

            # Compute Epoch wide loss and accuracy
            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects / running_total
            # If phase is validation, add to trial length list 
            if phase == 'val':
                # Save Epoch loss and wts for checkpoint
                checkpoint_data = {'wts': model.state_dict()}
                checkpoint = Checkpoint.from_dict(checkpoint_data)
                session.report({'loss': epoch_loss,
                                'acc': epoch_acc}, checkpoint=checkpoint)
                
def test_model(model, test_loader, sc):
     
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        model.eval()

        for (X, y) in test_loader:

            X = X.to(device)
            X = torch.nan_to_num(sc(X))
            y = y.to(device)
            
            # Model outputs
            outputs = model(X)
            
            loss = criterion(outputs, y)

            # Convert to sigmoid layer, threshold at 0.5 and convert to 0,1
            pred = torch.sigmoid(outputs)
            results = (pred > 0.5).float()
            running_loss += loss.item() * y.shape[0]
            running_total += y.shape[0]
            running_corrects += (results == y).sum().item()

        loss = running_loss / running_total
        acc = running_corrects / running_total
        print(f'Accuracy = {acc}, Loss = {loss}')

if __name__ == '__main__':
    # Mlflow setup
    client = MlflowClient()
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    exp_name = 'MLP-'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = client.create_experiment(name=exp_name)

    # Dataset loading
    train = MMADataset3D(split='train',
                        transform=ToTensor())
    
    val = MMADataset3D(split='val',
                        transform=ToTensor())
    
    test = MMADataset3D(split='test',
                        transform=ToTensor())

    search_space = {'mlflow_experiment_id': experiment_id,
                    'lr': tune.loguniform(1e-5, 1e-3),
                    'momentum': tune.loguniform(1e-3, 1),
                    'batch_size': tune.choice([2,4,8,16,32]),
                    'hidden1_size': tune.qrandint(300,2000,5),
                    'hidden2_size': tune.qrandint(100,2000,5),
                    # 'hidden3_size': tune.qrandint(10,100,1),
                    'alpha': tune.uniform(0,0.5),
                    'dropout': tune.uniform(0,0.5)
                    }
    
    # Constants for model use
    input_size = train[0][0].shape[0]
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
            num_samples=150,
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
    best_model = MMANet(input_size=input_size,
                        hidden1_size=best_config['hidden1_size'],
                        hidden2_size=best_config['hidden2_size'],
                        # hidden3_size=best_config['hidden3_size'],
                        alpha=best_config['alpha'],
                        dropout=best_config['dropout'],
                        output_size=output_size).to(device)
    best_model.load_state_dict(checkpoint_data['wts'])

    sc = StandardScaler()
    sc.fit(train[:][0])
    
    test_loader = DataLoader(dataset=test,
                    batch_size=best_config['batch_size'],
                    shuffle=False)
    
    test_model(best_model, test_loader, sc)
    torch.save(best_model, 'C:/Users/enioh/Documents/Github/MMA-ML-Model/models/mlp.pt')
