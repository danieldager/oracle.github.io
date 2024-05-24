# TODO: track train and inference times

import os, sys
import numpy as np
from pprint import pprint
from random import randint, uniform
from datetime import datetime

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

for module in ['actions']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from load_data import load_data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# for debugging
# torch.set_printoptions(threshold=torch.inf)


""" Model """

def get_hyperparameters(random):
    if random:
        learning_rate = 10 ** uniform(-2.5, -1.5)
        hyperparameters = {
            'batch_size': randint(50, 150),
            'hidden_size': randint(30, 80),
            'num_stacked_layers': randint(1, 3),
            'learning_rate': round(learning_rate, 4),
            'num_epochs': randint(1, 3)
        }
    else:
        hyperparameters = {
            'batch_size': 100,
            'hidden_size': 10,
            'num_stacked_layers': 1,
            'learning_rate': 0.01,
            'num_epochs': 1
        }

    return hyperparameters


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)

        # pack sequence to avoid processing padded values
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x, (h0, c0))

        # unpack sequence for processing by linear layer
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # batch indices to extract the last hidden state
        batch_indices = torch.arange(out.size(0)).to(device)

        # extract last hidden state of each sequence
        out = self.linear(out[batch_indices, lengths - 1])
        out = self.sigmoid(out)

        return out


def train_one_epoch(model, loss_function, optimizer, train_loader, log=True):
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
        if log and i % 10 == 0:
            print(f'batch {i} loss: {total_loss / (i+1)}')
            loss = 0.0

    return total_loss / len(train_loader)


def validate_one_epoch(model, loss_function, eval_loader, log=True):
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

    if log:
        print(f'validation loss: {total_loss / len(eval_loader)}\n\n')

    return total_loss / len(eval_loader)


""" Train/Test """

def train(n_models=1, directories=[], logging=True, random=True):
    all_models = []
    all_params = []
    model_names = []

    """ Training """

    for m in range(n_models):
        print(f"Training model {m + 1}\n")

        # get hyperparameters
        params = get_hyperparameters(random)
        pprint(params), print("\n")

        # load data
        train_loader, eval_loader, test_loader = load_data(directories, params['batch_size'])
        
        # inputs = decision, delay, pupil diameter
        input_size = 3
        model = LSTM(
            input_size, 
            params['hidden_size'], 
            params['num_stacked_layers']
        ).to(device)

        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # train model on flattened data from all trials
        for _ in range(params['num_epochs']):
            train_one_epoch(model, loss_function, optimizer, train_loader)
            validate_one_epoch(model, loss_function, eval_loader)

        # model info for testing
        all_models.append(model)
        all_params.append(params)
    
        # name model
        model_name = datetime.now().strftime('%m%d-%H%M')
        model_names.append(model_name)

        # save model
        base_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        model_path = os.path.abspath(os.path.join(base_path, f"{model_name}.pt"))
        torch.save({"model": model, "model_name": model_name, "params": params}, model_path)

    """ Testing  """

    for m, model in enumerate(all_models):

        # print model details
        print(f"\nModel {model_names[m]}\n")
        pprint(all_params[m]), print("\n")

        all_accuracies = []
        all_predictions = []

        # run inference on each test trial
        for trial, (segments, lengths) in enumerate(test_loader):
            correct = 0
            accuracies = []
            predictions = []

            # format each segment for inference
            for i in range(len(segments) - 1):
                X = segments[i].unsqueeze(0).to(device)
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
            all_accuracies.append(accuracies)

        # calculate average accuracy of model over all test trials
        average_accuracy = sum([trial[-1] for trial in all_accuracies]) / len(test_loader)
        if logging:
            print(f"\nModel {model_name} accuracy: {average_accuracy:.2f}%\n")

    return all_models if len(model_names) > 1 else all_models[0]


if __name__ == "__main__":
    models = train()
