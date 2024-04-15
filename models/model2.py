
from oracle import oracle_inference
from helpers import (format_sequence, format_datasets, load_data_from_db)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


""" Model """

class ModelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # Replace LSTM with basic RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)

        # Only hidden state is required for basic RNN
        out, _ = self.rnn(x, h0)
        out = self.linear(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    # TODO: do we want h1 and c1 to perserve state between batches?
    def forward(self, x):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def train_one_epoch(model, loss_function, optimizer, train_loader, log=False):
    model.train(True)
    loss = 0.0
    
    # i = batch index, (X, y) = batch
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # forward pass, compute loss
        y_pred = model(X)
        print(y_pred, y)
        print(y_pred.shape, y.shape)
        loss = loss_function(y_pred, y)
        loss += loss.item()

        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # print batch number and average loss every 10 batches
        if log and i % 10 == 0:
            print(f'batch loss{i+1}: {loss / 10}')
            loss = 0.0


def validate_one_epoch(model, loss_function, test_loader, log=False):
    model.train(False)
    loss = 0.0

    for _, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        # forward pass, compute loss
        with torch.no_grad():
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            loss += loss.item()

    if log:
        print(f'validate loss: {loss / len(test_loader)}')
        print('*' * 30)


""" Data """

trials, all_trials = load_data_from_db()

train_test_split = 0.7
train_size = int(len(all_trials) * train_test_split)
train_data, test_data = all_trials[:train_size], all_trials[train_size:]


""" Hyperparameters """

def get_hyperparameters():
    seq_len = 20            # randint(20, 25)
    batch_size = 10         # randint(10, 20)
    hidden_size = 5         # randint(15, 20)
    num_stacked_layers = 1  # randint(1, 5)
    learning_rate = 0.01    # 10 ** randint(-2, -1)
    num_epochs = 20         # randint(25, 30)

    return seq_len, batch_size, hidden_size, num_stacked_layers, learning_rate, num_epochs


""" Inference """

models = []
hyperparameters = []

for m in range(2):
    print(f"Training model {m + 1}")

    # define hyperparameters
    seq_len, batch_size, hidden_size, num_stacked_layers, \
    learning_rate, num_epochs = get_hyperparameters()
    
    # construct datasets and dataloaders
    X_train, y_train = format_datasets(train_data, seq_len)
    X_test, y_test = format_datasets(test_data, seq_len)

    train_dataset = ModelDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ModelDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # define model, loss function, optimizer
    model = LSTM(2, hidden_size, num_stacked_layers).to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model on flattened data from all trials
    for epoch in range(num_epochs):
        train_one_epoch(model, loss_function, optimizer, train_loader)
        validate_one_epoch(model, loss_function, test_loader)

    # save models and hyperparameters for trial-by-trial evaluation
    models.append(model)
    hyperparameters.append((seq_len, batch_size, hidden_size, 
                            num_stacked_layers, learning_rate, num_epochs))


# evaluate model accuracy for each trial
for m, model in enumerate(models):
    # print hyperparameters
    print(f"\n\nModel {m + 1}")
    print(f"seq_len: {hyperparameters[m][0]}")
    print(f"batch_size: {hyperparameters[m][1]}")
    print(f"hidden_size: {hyperparameters[m][2]}")
    print(f"num_stacked_layers: {hyperparameters[m][3]}")
    print(f"learning_rate: {hyperparameters[m][4]}")
    print(f"num_epochs: {hyperparameters[m][5]}\n")

    average_accuracy = 0
    for trial_number, trial in enumerate(trials):
        preds = []
        hits = 0

        # format trial data (Ls and Rs to 0s and 1s)
        sequence = format_sequence(trial)

        # run inference on each trial
        for i, key in enumerate(sequence): 
            try:
                input = sequence[i:i + seq_len]
                target = sequence[i + seq_len]
            except IndexError:
                break

            # format input for model
            input = torch.tensor(input).reshape(1, -1, 2).float().to(device)

            # make prediction
            with torch.no_grad(): pred = model(input).item()
            pred = 0 if pred < 0.5 else 1

            # evaluate prediction
            if pred == target: hits += 1
            preds.append(pred)

        # calculate model accuracy for individual trial
        accuracy = (hits / len(preds)) * 100
        print(f"trial {trial_number + 1} accuracy: {accuracy:.2f}%")
        average_accuracy += accuracy
    
    # calculate average accuracy of model over all trials
    average_accuracy /= len(trials)
    print(f"\nModel {m + 1} accuracy: {average_accuracy:.2f}%")


# Report accuracy of aaronson oracle
oracle_inference(trials, window_size=5)