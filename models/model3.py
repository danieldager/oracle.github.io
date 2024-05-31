import os, sys
import numpy as np
from pprint import pprint
from random import randint
from datetime import datetime

# from oracle2 import oracle_inference
# from ..actions.load_data import load_data

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Add the project root to sys.path
# root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# cwd = os.getcwd()
# for module in ['modules']:
#     module_path = os.path.abspath(os.path.join(cwd, '..', module))
#     sys.path.append(module_path)

for module in ['actions']:
    # path = os.path.join(os.getcwd(), '..', module)
    # sys.path.append(os.path.abspath(path))

    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))




# print(sys.path)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# base_path = os.path.join(script_dir, '..', 'trials')

# print(script_dir)
# print(base_path)

# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trials'))
# print(base_path)

from oracle2 import oracle_inference
from load_data2 import load_data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# for debugging
torch.set_printoptions(threshold=torch.inf)


""" Dataset """

# # segmenting so we can train on variable sequence lengths
# def segment_sequences(data):
#     segments = []
#     for sequence in data:
#         for i in range(2, len(sequence)):
#             segments.append(sequence[:i])
#     return segments

# # custom class to ensure that y = last decision
# class SegmentDataset(Dataset):
#     def __init__(self, segments):
#         self.segments = segments

#     def __len__(self):
#         return len(self.segments)

#     def __getitem__(self, i):
#         segment = self.segments[i]
#         X = segment[:-1]
#         y = segment[-1][0]
#         return X, y

# # custom collate function to pad sequences
# def collate_fn1(batch):
#     # sort shuffled batch in descending order by length
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     sequences, labels = zip(*batch)

#     for seq in sequences:
#         if len(seq) == 0:
#             print(seq)
#             print(seq.shape)

#     # calculate the lengths of each sequence
#     lengths = torch.tensor([len(s) for s in sequences]).to(device)

#     # pad sequences to the maximum length in batch
#     padded_sequences = pad_sequence(sequences, batch_first=True)

#     # convert labels to tensors and add a dimension
#     labels = torch.tensor(labels).float().unsqueeze(1).to(device)

#     return padded_sequences, labels, lengths


# def load_data(directory):
#     # grab file_paths from local directory
#     file_paths = []
#     for file_name in os.listdir(directory):
#         file_paths.append(os.path.join(directory, file_name))

#     # convert numpy arrays to torch tensors
#     sequences = []
#     for file_path in file_paths:
#         sequence = np.load(file_path)
#         sequence = torch.tensor(sequence, dtype=torch.float32)
#         sequences.append(sequence.T) # transpose to (n, 3) shape

#     # split test data
#     split_ratio = 0.1
#     total_size = len(sequences)
#     test_size = int(split_ratio * total_size)
#     test_data = sequences[-test_size:]
#     sequences = sequences[:-test_size]

#     # split train and eval data
#     split_ratio = 0.8
#     total_size = len(sequences)
#     train_size = int(split_ratio * total_size)
#     eval_size = total_size - train_size
#     train_data, eval_data = random_split(sequences, [train_size, eval_size])

#     # segment sequences
#     train_segments = SegmentDataset(segment_sequences(train_data))
#     eval_segments = SegmentDataset(segment_sequences(eval_data))

#     # create data loaders
#     train_loader = DataLoader(train_segments, batch_size=32, shuffle=True, collate_fn=collate_fn1)
#     eval_loader = DataLoader(eval_segments, batch_size=32, shuffle=False, collate_fn=collate_fn1)

#     return train_loader, eval_loader, test_data, sequences


""" Model """

def get_hyperparameters(random=False):
    if random:
        hyperparameters = {
            'batch_size': randint(50, 150),
            'hidden_size': randint(30, 80),
            'num_stacked_layers': randint(1, 3),
            'learning_rate': 10 ** randint(-2.5, -1.5),
            'num_epochs': randint(2, 5)
        }
    else:
        hyperparameters = {
            'batch_size': 100,         # randint(10, 20)
            'hidden_size': 50,         # randint(15, 20)
            'num_stacked_layers': 1,   # randint(1, 5)
            'learning_rate': 0.01,     # 10 ** randint(-2, -1)
            'num_epochs': 1            # randint(25, 30)
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
    
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_stacked_layers):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_stacked_layers = num_stacked_layers

#         # Replace LSTM with basic RNN
#         self.rnn = nn.RNN(input_size, hidden_size, num_stacked_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(device)

#         # Only hidden state is required for basic RNN
#         out, _ = self.rnn(x, h0)
#         out = self.linear(out[:, -1, :])
#         out = self.sigmoid(out)
#         return out    

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
        print(f'validation loss: {total_loss / len(eval_loader)}')
        print('*' * 40)

    return total_loss / len(eval_loader)


""" Train/Test """

def train(n_models, directories=["blind"], oracle=True, logging=True, random=False):
    all_models = []
    all_params = []
    model_names = []

    """ Training """

    for m in range(n_models):
        print(f"Training model {m + 1}")

        # get hyperparameters and load data
        params = get_hyperparameters(random)
        train_loader, eval_loader, test_loader = load_data(directories, params['batch_size'])
        
        # define model, loss function, optimizer
        input_size = 3  # decision, delay, pupil diameter
        model = LSTM(
            input_size, 
            params['hidden_size'], 
            params['num_stacked_layers']
        ).to(device)

        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # train model on flattened data from all trials
        for epoch in range(params['num_epochs']):
            train_one_epoch(model, loss_function, optimizer, train_loader)
            validate_one_epoch(model, loss_function, eval_loader)

        # model info for testing
        all_models.append(model)
        all_params.append(params)
    
        # name model
        model_name = datetime.now().strftime('%m%d-%H%M')
        model_names.append(model_name)

        # save model
        model_path = f"saved_models/{model_name}.pt"
        torch.save({"model": model, "model_name": model_name, "params": params}, model_path)


    # class InferenceDataset(Dataset):
    #     def __init__(self, sequences):
    #         self.sequences = sequences

    #     def __len__(self):
    #         return len(self.sequences)

    #     def __getitem__(self, idx):
    #         sequence = self.sequences[idx]
    #         segments = [sequence[:i] for i in range(1, len(sequence) + 1)]
    #         return segments
        
    # def collate_fn2(batch):
    #     sequences = batch[0]
    #     lengths = [len(seq) for seq in sequences]
    #     padded_sequences = pad_sequence(sequences, batch_first=True)
    #     return padded_sequences, lengths

    # inference_loader = DataLoader(InferenceDataset(test_data), batch_size=1, shuffle=False, collate_fn=collate_fn2)


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


         # Report accuracy of aaronson oracle
        if oracle:
            oracle_inference(test_loader, window_size=5)


if __name__ == "__main__":
    train(1, oracle=False)



    # Report accuracy of aaronson oracle
    # oracle_inference(sequences, window_size=5)