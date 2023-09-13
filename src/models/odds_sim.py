import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'C:/Users/enioh/Documents/Github/MMA-ML-Model')
from src.features.load_data import load_data
from src.models.RNN_Model import MMA_RNN, StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting up data for odds comparison
X_train, y_train, _ = load_data(split='train')
X_test, y_test, odds_test = load_data(split='test')

# Model loading
model = torch.load('C:/Users/enioh/Documents/Github/MMA-ML-Model/models/rnn.pt')
model.eval()

# Need to get scaling factor for X_test first 
sc = StandardScaler()
mat = np.reshape(X_train, (-1, X_train.shape[2]))
sc.fit(mat)

X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

fighter_len = X_test.shape[0]
fight_len = X_test.shape[1]

# Odds simulation constants
bank = 1000

# counts
correct = 0
total = 0

rng = np.random.default_rng(4)
for fighter in rng.permutation(fighter_len):
    for fight in range(0, fight_len):

        with torch.no_grad():
            X = X_test[fighter, fight, :]
            if X.isnan().sum() / X.shape[0] > 1:
                continue
            
            X = torch.unsqueeze(X, 0)
            X = torch.nan_to_num(sc(X))
            X = torch.unsqueeze(X, 0)

            y = y_test[fighter, fight, 0]
            odd = odds_test[fighter, fight, 0]

            if torch.isnan(y):
                continue

            L = torch.LongTensor([1]).to('cpu')

            output = model([X,L,y])
            pred = torch.sigmoid(output).item()

            if pred > 0.5 and np.isfinite(odd):
                print(f'Model Predicts a win likelihood of {pred:.3f} | Odds are {odd:.3f}')
                print(f'Placing bet...')
                
                # Betting criterion 
                fraction = pred - ((1 - pred) / ((1 - odd) / odd))
                if fraction < 0:
                    print('No bet')
                    continue

                bet_amount = fraction * bank * 0.1
                print(f'Betting {bet_amount:.2f}')

                # Check win and adjust bank
                if y == 1:
                    bank += bet_amount * ((1 - odd) / odd)
                    print(f'Bet won, Current Bank: {bank:.2f}')
                    correct += 1
                else:
                    bank -= bet_amount
                    print(f'Bet lost, Current Bank: {bank:.2f}')
                
                total += 1
                time.sleep(0.1)

print(f'Final Bank Value: {bank:.2f}')
print(f'Total Acc: {correct/total:.2f}')
