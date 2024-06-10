# TODO: track train and inference times

import os, sys
from pprint import pprint
from datetime import datetime
from random import randint, uniform

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

for module in ['actions']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from load_data import load_data_vary

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# for debugging
# torch.set_printoptions(threshold=torch.inf)


""" Model """

def get_hyperparameters(random):

    if random == 'binary':
        learning_rate = 10 ** uniform(-2.5, -1.5)
        hyperparameters = {
            'input_size': 1,
            'batch_size': randint(15, 25) * 10,
            'hidden_size': randint(3, 9) * 10,
            'num_stacked_layers': randint(1, 3),
            'learning_rate': round(learning_rate, 4),
            # 'num_epochs': randint(2, 5),
            'num_epochs': 1,
        }

    elif random == 'delay':
        learning_rate = 10 ** uniform(-2.5, -1.5)
        hyperparameters = {
            'input_size': 2,
            'batch_size': randint(15, 25) * 10,
            'hidden_size': randint(3, 9) * 10,
            'num_stacked_layers': randint(1, 3),
            'learning_rate': round(learning_rate, 4),
            # 'num_epochs': randint(2, 5),
            'num_epochs': 1,
        }

    elif random == 'pupil':
        learning_rate = 10 ** uniform(-2.5, -1.5)
        hyperparameters = {
            'input_size': 2,
            'batch_size': randint(15, 25) * 10,
            'hidden_size': randint(3, 9) * 10,
            'num_stacked_layers': randint(1, 3),
            'learning_rate': round(learning_rate, 4),
            # 'num_epochs': randint(2, 5),
            'num_epochs': 1,
        }

    elif random == 'all':
        learning_rate = 10 ** uniform(-2.5, -1.5)
        hyperparameters = {
            'input_size': 3,
            'batch_size': randint(15, 25) * 10,
            'hidden_size': randint(3, 9) * 10,
            'num_stacked_layers': randint(1, 3),
            'learning_rate': round(learning_rate, 4),
            # 'num_epochs': randint(2, 5),
            'num_epochs': 1,
        }

    else:
        hyperparameters = {
            'input_size': 3,
            'batch_size': 100,
            'hidden_size': 10,
            'num_stacked_layers': 1,
            'learning_rate': 0.01,
            # 'num_epochs': randint(2, 5),
            'num_epochs': 1,
        }

    return hyperparameters


class LSTM_Vary(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)

        # add channel dimension if input size is 1        
        if self.input_size == 1:
            x = x.unsqueeze(2)

        # pack sequence to avoid processing padded values
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # pass packed sequence through LSTM
        packed_out, _ = self.lstm(packed_x, (h0, c0))

        # unpack sequence for processing by linear layer
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # batch indices to extract the last hidden state
        batch_indices = torch.arange(out.size(0)).to(device)

        # extract last hidden state of each sequence
        out = self.linear(out[batch_indices, lengths - 1])
        out = self.sigmoid(out)

        return out


def train_one_epoch(model, loss_function, optimizer, train_loader, logging=True):
    model.train(True)
    total_loss = 0.0
    
    # i = batch index, (X, y, lengths) = batch
    for i, (X, y, lengths) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        # forward pass, compute loss
        y_pred = model(X, lengths)
        loss = loss_function(y_pred, y)
        total_loss += loss.item()

        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # print batch number and average loss every 10 batches
        if logging and i % 10 == 0:
            print(f'batch {i} loss: {total_loss / (i+1)}')
            loss = 0.0

    return total_loss / len(train_loader)


def validate_one_epoch(model, loss_function, eval_loader, logging=True):
    model.train(False)
    total_loss = 0.0

    for _, (X, y, lengths) in enumerate(eval_loader):
        X = X.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        # forward pass, compute loss
        with torch.no_grad():
            y_pred = model(X, lengths)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()

    if logging:
        print(f'validation loss: {total_loss / len(eval_loader)}\n\n')

    return total_loss / len(eval_loader)


""" Train/Test """

def train_vary(train_data, eval_data, test_data, params=None, logging=True, random=False):

    """ Training """

    # name model
    model_name = datetime.now().strftime('%m%d-%H%M')
    if logging:
        print(model_name + '\n')
        
    # get hyperparameters
    if params is None:
        params = get_hyperparameters(random)

    if logging:
        pprint(params), print("\n")

    # load data
    train_loader, eval_loader, test_loader = load_data_vary(
        train_data, eval_data, test_data, batch_size=params['batch_size']
    )
    
    # initialize model, loss function, and optimizer
    model = LSTM_Vary(
        params['input_size'],
        params['hidden_size'], 
        params['num_stacked_layers']
    ).to(device)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # train model on flattened data from all trials
    for _ in range(params['num_epochs']):
        train_one_epoch(model, loss_function, optimizer, train_loader, logging)
        validate_one_epoch(model, loss_function, eval_loader, logging)

    # save model
    base_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    model_path = os.path.abspath(os.path.join(base_path, f"{model_name}.pt"))
    torch.save({"model": model, "model_name": model_name, "params": params}, model_path)

    """ Testing  """

    average_accuracy = 0
    
    # run inference on each test trial
    for trial, (segments, lengths) in enumerate(test_loader):
        correct = 0
        accuracies = []
        predictions = []

        # format each segment for inference
        for i in range(len(segments) - 1):
            X = segments[i].unsqueeze(0).to(device)

            if params['input_size'] == 1:
                y = segments[i + 1][i + 1].item()
            else:
                y = segments[i + 1][i + 1][0].item()

            length = torch.tensor([lengths[i]]).to(device)

            # make prediction
            with torch.no_grad():
                prediction = 0 if (model(X, length).item()) < 0.5 else 1
            if prediction == y: correct += 1
            
            predictions.append(prediction)
            accuracies.append(100 * correct / (i + 1))

        if logging:
            print(f"trial {trial + 1} accuracy: {accuracies[-1]:.2f}%")
        average_accuracy += accuracies[-1]

    # calculate average accuracy of model over all test trials
    average_accuracy /= len(test_loader)
    if logging:
        print(f"\nModel {model_name} accuracy: {average_accuracy:.2f}%\n")

    return model_name, params, average_accuracy


# if __name__ == "__main__":
