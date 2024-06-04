import os, sys
from pprint import pprint
from datetime import datetime
from random import randint, uniform

import torch
from torch import nn

for module in ['actions']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from load_data3 import load_data_fixed

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
            'num_epochs': randint(1, 3),
            'segment_length': randint(3, 8)
        }
    else:
        hyperparameters = {
            'batch_size': 100,
            'hidden_size': 10,
            'num_stacked_layers': 1,
            'learning_rate': 0.01,
            'num_epochs': 1,
            'segment_length': 5
        }

    return hyperparameters


class LSTM_Fixed(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)

        # since segments are of fixed length, we can pass them directly through the model 
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = self.sigmoid(out)

        return out


def train_one_epoch(model, loss_function, optimizer, train_loader, log=True):
    model.train(True)
    total_loss = 0.0
    
    # i = batch index, (X, y, lengths) = batch
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # forward pass, compute loss
        y_pred = model(X)
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

    for _, (X, y) in enumerate(eval_loader):
        X = X.to(device)
        y = y.to(device)

        # forward pass, compute loss
        with torch.no_grad():
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()

    if log:
        print(f'validation loss: {total_loss / len(eval_loader)}\n\n')

    return total_loss / len(eval_loader)


""" Train/Test """

def train_fixed(train_data, eval_data, test_data, input_size=3, logging=True, random=False):

    """ Training """

    # name model
    model_name = datetime.now().strftime('%m%d-%H%M')
    print(model_name + '\n')

    # get hyperparameters
    params = get_hyperparameters(random)
    pprint(params), print("\n")

    # load data
    train_loader, eval_loader, test_loader = load_data_fixed(
        train_data, eval_data, test_data, params['batch_size'], params['segment_length']
    )
    
    # initialize model, loss function, and optimizer
    model = LSTM_Fixed(
        input_size,       # inputs => decision, delay, pupil diameter
        params['hidden_size'], 
        params['num_stacked_layers']
    ).to(device)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # train model on flattened data from all trials
    for _ in range(params['num_epochs']):
        train_one_epoch(model, loss_function, optimizer, train_loader)
        validate_one_epoch(model, loss_function, eval_loader)

    # save model
    base_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    model_path = os.path.abspath(os.path.join(base_path, f"{model_name}.pt"))
    torch.save({"model": model, "model_name": model_name, "params": params}, model_path)

    """ Testing  """

    average_accuracy = 0

    # run inference on each test trial
    for trial, segments in enumerate(test_loader):
        correct = 0
        accuracies = []
        predictions = []

        # run inference on each trial
        for i in range(len(segments) - 1):
            X = segments[i][:, :params['segment_length'], :].to(device)
            y = segments[i][:, params['segment_length'], 0].item()
            
            # make prediction
            with torch.no_grad():
                prediction = 0 if (model(X).item()) < 0.5 else 1
            if prediction == y: correct += 1
            
            # store prediction and accuracy at each step
            predictions.append(prediction)
            accuracies.append(100 * correct / (i + 1))

        # add final accuracy of the trial to average
        average_accuracy += accuracies[-1]
        
        if logging:
            print(f"trial {trial + 1} accuracy: {accuracies[-1]:.2f}%")

    # calculate average accuracy of model over all test trials
    average_accuracy /= len(test_loader)
    if logging:
        print(f"\nModel {model_name} accuracy: {average_accuracy:.2f}%\n")

    return model


# if __name__ == "__main__":
#     model = train5()
