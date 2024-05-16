import os
import numpy as np

from oracle import oracle_inference

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# for debugging
torch.set_printoptions(threshold=torch.inf)


""" Dataset """

# grab file_paths from local directory
directory = 'trials'
file_paths = []
for file_name in os.listdir(directory):
    file_paths.append(os.path.join(directory, file_name))

# convert numpy arrays to torch tensors
sequences = []
for file_path in file_paths:
    sequence = np.load(file_path)
    if len(sequence) == 0:
        print("here")
        print(sequence)
        print(sequence.shape)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequences.append(sequence.T) # transpose to (n, 3) shape

# split sequences into training and testing sets
train_size = int(0.8 * len(sequences))
test_size = len(sequences) - train_size
train_data, test_data = random_split(sequences, [train_size, test_size])

# segmenting so we can train on variable sequence lengths
def segment_sequences(data):
    segments = []
    for sequence in data:
        for i in range(2, len(sequence)):
            segments.append(sequence[:i])
    return segments

train_segments = segment_sequences(train_data)
test_segments = segment_sequences(test_data)

# custom class to ensure that y = last decision
class SegmentDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        segment = self.segments[i]
        X = segment[:-1]
        y = segment[-1][0]
        return X, y

train_segments = SegmentDataset(train_segments)
test_segments = SegmentDataset(test_segments)

# custom collate function to pad sequences
def collate_fn(batch):
    # sort shuffled batch in descending order by length
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)

    for seq in sequences:
        if len(seq) == 0:
            print(seq)
            print(seq.shape)

    # calculate the lengths of each sequence
    lengths = torch.tensor([len(s) for s in sequences]).to(device)

    # pad sequences to the maximum length in batch
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # convert labels to tensors and add a dimension
    labels = torch.tensor(labels).float().unsqueeze(1).to(device)

    return padded_sequences, labels, lengths

train_loader = DataLoader(train_segments, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_segments, batch_size=32, shuffle=False, collate_fn=collate_fn)


""" Model """

def get_hyperparameters():
    batch_size = 32         # randint(10, 20)
    hidden_size = 50        # randint(15, 20)
    num_stacked_layers = 1  # randint(1, 5)
    learning_rate = 0.01    # 10 ** randint(-2, -1)
    num_epochs = 3         # randint(25, 30)

    return batch_size, hidden_size, num_stacked_layers, learning_rate, num_epochs

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
            print(f'batch {i+1} loss: {total_loss / (i+1)}')
            loss = 0.0

    return total_loss / len(train_loader)


def validate_one_epoch(model, loss_function, test_loader, log=True):
    model.train(False)
    total_loss = 0.0

    for _, (X, y, lengths) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        # forward pass, compute loss
        with torch.no_grad():
            y_pred = model(X, lengths)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()

    if log:
        print(f'validation loss: {total_loss / len(test_loader)}')
        print('*' * 40)

    return total_loss / len(test_loader)


""" Inference """

models = []
hyperparameters = []

for m in range(1):
    print(f"Training model {m + 1}")

    # define hyperparameters
    batch_size, hidden_size, num_stacked_layers, \
    learning_rate, num_epochs = get_hyperparameters()
    
    # define model, loss function, optimizer
    input_size = 3  # decision, delay, pupil diameter
    model = LSTM(input_size, hidden_size, num_stacked_layers).to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model on flattened data from all trials
    for epoch in range(num_epochs):
        train_one_epoch(model, loss_function, optimizer, train_loader)
        validate_one_epoch(model, loss_function, test_loader)

    # save models and hyperparameters for trial-by-trial evaluation
    models.append(model)
    hyperparameters.append((batch_size, hidden_size, 
                            num_stacked_layers, learning_rate, num_epochs))


class InferenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        segments = [sequence[:i] for i in range(1, len(sequence) + 1)]
        return segments
    
def collate_inf(batch):
    sequences = batch[0]
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths

inference_loader = DataLoader(InferenceDataset(sequences), batch_size=1, shuffle=False, collate_fn=collate_inf)


# evaluate model accuracy for each trial
for m, model in enumerate(models):

    # print hyperparameters
    print(f"\n\nModel {m + 1}")
    print(f"batch_size: {hyperparameters[m][0]}")
    print(f"hidden_size: {hyperparameters[m][1]}")
    print(f"num_stacked_layers: {hyperparameters[m][2]}")
    print(f"learning_rate: {hyperparameters[m][3]}")
    print(f"num_epochs: {hyperparameters[m][4]}\n")

    avg_accuracy = 0
    for trial, (segments, lengths) in enumerate(inference_loader):
        preds = []
        hits = 0

        # run inference on each trial
        for i in range(len(segments) - 1):
            X = segments[i].unsqueeze(0).to(device)
            y = segments[i + 1][0][0].item()
            length = torch.tensor([lengths[i]]).to(device)

            # make prediction
            with torch.no_grad(): 
                pred = model(X, length).item()
            pred = 0 if pred < 0.5 else 1

            # evaluate prediction
            if pred == y: hits += 1
            preds.append(pred)

        # calculate model accuracy for individual trial
        accuracy = (hits / len(preds)) * 100
        print(f"trial {trial + 1} accuracy: {accuracy:.2f}%")
        avg_accuracy += accuracy
    
    # calculate average accuracy of model over all trials
    avg_accuracy /= len(sequences)
    print(f"\nModel {m + 1} accuracy: {avg_accuracy:.2f}%")


# Report accuracy of aaronson oracle
# oracle_inference(sequences, window_size=5)