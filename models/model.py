""" TODO """

# Get the basic (just keystrokes) model running
# Look at trial-by-trial and global predictive power with original oracle

# Get with_delay model running
# Look at trial-by-trial and global predictive power keystrokes model

# Vary the parameters of the models, compare

# Calculate the algorithmic randomness of 5-length strings for trials
# Compare this with the predictability of the trial, what do we see? 
# Also compute all other randomness measures

# Start working with the pupil headset (add pupil diameter as feature)
# Get the pupil model running, compare with other models

# try different model architectures (Transformer)


import torch
import keyboard
from random import randint, randrange

from torch import nn
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import os
from dotenv import load_dotenv
from pymongo import MongoClient


""" Notes """

# How can we batch the data so that the model treats one session of prediction 
# as coming one individual? Meaning, can we get the model to personalize itself
# to the current user? The idea is not that the model learn to predict the key 
# following a 5-key sequence, but rather that it build identify the behavioral
# type of the user and predict the next key based on that. (Hidden Markov Model?)


""" Database """

# connect to database
load_dotenv()
client = MongoClient(os.environ['MONGO_URL'])
database = client['Oracle']
collection = database['trials']


# test uuid
# uuid = "2ed055c9-e820-4cf7-a3ff-bc66ae35257e"
# doc = collection.find_one({"uuid": uuid})

# extract data and make a flattened array of all keystrokes
# note: here we are losing the individuality of each session 
data = collection.find()
data = [doc['data'] for doc in data]

trials = data

# index = randrange(len(data) - 20)
# trials2 = data[index: index + 20]

data = [dp for doc in data for dp in doc]

# extract features, convert L and R to 1 and 2
keystrokes = []
with_delay = []
table = {"L": 0, "R": 1}

for dp in data:
    keystrokes.append(table[dp['key']])
    with_delay.append((table[dp['key']], dp['delay']))

# split data into train and test
train_size = int(len(keystrokes) * 0.7)
train, test = keystrokes[:train_size], keystrokes[train_size:]

# TODO: implement model with with_delay sequences (extra feature)


""" Helper Functions """
        
def format_trial(trial):
        keys = []
        delays = []

        for i, datapoint in enumerate(trial):
            keys.append(0 if datapoint['key'] == 'L' else 1)
            delays.append((keys[i], datapoint['delay']))
        
        return keys, delays


def get_hyperparameters():
    # seq_len = randint(20, 25)
    # batch_size = randint(10, 20)
    # hidden_size = randint(15, 20)
    # num_stacked_layers = randint(1, 5)
    # learning_rate = 10 ** randint(-2, -1)
    # num_epochs = randint(25, 30)
    seq_len = 30
    batch_size = 20
    hidden_size = 30
    num_stacked_layers = randint(3, 6)
    learning_rate = 0.01
    num_epochs = 30

    return seq_len, batch_size, hidden_size, num_stacked_layers, learning_rate, num_epochs


""" Hyperparameters """

seq_len = 20 # number of previous keystrokes to consider
batch_size = 10 # number of sequences to consider at once
hidden_size = 5 # number of hidden units in lstm
num_stacked_layers = 1 # number of stacked lstm layers
learning_rate = 0.001 # learning rate for optimizer
num_epochs = 20 # number of times to iterate over the entire dataset

""" Datasets """

# create feature and target sequences
def create_sequences(data, seq_len):
    features = []
    targets = []

    # feature is an array of seq_len elements
    # target is an array of the very next element
    for i in range(len(data) - seq_len - 1):
        f = data[i : i+seq_len]
        t = data[i+1+seq_len]
        features.append(f)
        targets.append(t)

    # reshape to conform to lstm input
    features = torch.tensor(features).reshape(-1, seq_len, 1).float()
    targets = torch.tensor(targets).reshape(-1, 1).float()
    return features, targets

X_train, y_train = create_sequences(train, seq_len)
X_test, y_test = create_sequences(test, seq_len)


""" Model """

# create datasets and dataloaders
class ModelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_dataset = ModelDataset(X_train, y_train)
test_dataset = ModelDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
    

# # create model, loss function, and optimizer
# rnn = RNN(1, hidden_size, num_stacked_layers).to(device)
# lstm = LSTM(1, hidden_size, num_stacked_layers).to(device)

# loss_function = nn.BCELoss()
# r_opt = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
# l_opt = torch.optim.Adam(lstm.parameters(), lr=learning_rate)


def train_one_epoch(model, loss_function, optimizer, train_loader):
    model.train(True)
    loss = 0.0

    # i = batch index, (X, y) = batch
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # forward pass, compute loss
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        loss += loss.item()

        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # print batch number and average loss every 10 batches
        # if i % 500 == 0:
            # print(f'batch {i+1}: {r_loss / 500}')
            # loss = 0.0
            # print(f'batch {i+1}: {l_loss / 500}')
            # l_loss = 0.0

def validate_one_epoch(model, loss_function, test_loader):
    model.train(False)
    loss = 0.0

    for _, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            # forward pass, compute loss
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            loss += loss.item()

    # print(f'loss: {loss / len(test_loader)}')
    # print('*' * 30)

# for epoch in range(num_epochs):
#     train_one_epoch()
#     validate_one_epoch()


""" Inference """

# train 10 random models
models = []
hyperparameters = []

for m in range(10):
    print(f"Training model {m + 1}")
    seq_len, batch_size, hidden_size, num_stacked_layers, \
    learning_rate, num_epochs = get_hyperparameters()
    
    X_train, y_train = create_sequences(train, seq_len)
    X_test, y_test = create_sequences(test, seq_len)

    train_dataset = ModelDataset(X_train, y_train)
    test_dataset = ModelDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM(1, hidden_size, num_stacked_layers).to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, loss_function, optimizer, train_loader)
        validate_one_epoch(model, loss_function, test_loader)

    models.append(model)
    hyperparameters.append((seq_len, batch_size, hidden_size, 
                            num_stacked_layers, learning_rate, num_epochs))

# test models on all trials, print accuracy and hyperparameters
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

        keystroke_array, reaction_times = format_trial(trial)

        for i, key in enumerate(keystroke_array): 
            try:
                ngram = keystroke_array[i:i + seq_len]
                target = keystroke_array[i + seq_len]
            except IndexError:
                break

            # make prediction, append to list
            ngram_tsr = torch.tensor(ngram).reshape(1, -1, 1).float().to(device)
            with torch.no_grad(): pred = model(ngram_tsr).item()
            pred = 0 if pred < 0.5 else 1
            if pred == target: hits += 1
            preds.append(pred)

        r_accuracy = (hits / len(preds)) * 100
        print(f"trial {trial_number + 1} accuracy: {r_accuracy:.2f}%")
        average_accuracy += r_accuracy
    
    average_accuracy /= len(trials)
    print(f"Model {m + 1} accuracy: {average_accuracy:.2f}%")

# now report trial and total accuracy for aaronson oracle
average_accuracy = 0
print("\n\nOracle inference\n")
for trial_number, trial in enumerate(trials):
    oracle = {}
    window = 5
    preds = []
    hits = 0

    keystroke_array, reaction_times = format_trial(trial)

    # prediction loop
    for i, key in enumerate(keystroke_array): 
        try:
            ngram = keystroke_array[i:i + window]
            target = keystroke_array[i + window]
        except IndexError:
            break

        ngram_str = "".join(map(str, ngram))
        try:
            # use string of ngram as key for oracle dictionary
            pred = oracle[ngram_str]
            # assess which key has been seen most following this ngram
            if pred["0"] == pred["1"]: pred = randrange(1)
            else: pred = 0 if pred["0"] > pred["1"] else 1
            
        except KeyError:
            # if ngram not already in oracle, predict randomly
            pred = randrange(1)

        # assess prediction 
        if pred == target: hits += 1
        # append prediction to list
        preds.append(pred)
        
        # update oracle with new key
        oracle[ngram_str] = {"0": 0, "1": 0}
        oracle[ngram_str][str(target)] += 1

    accuracy = (hits / len(preds)) * 100
    print(f"trial {trial_number + 1} accuracy: {accuracy:.2f}%")
    average_accuracy += accuracy

average_accuracy /= len(trials)
print(f"Oracle accuracy: {average_accuracy:.2f}%")






# for trial_number, trial in enumerate(trials):
#     keys = []
#     delays = []
#     table = {"L": 0, "R": 1}

#     for dp in trial:
#         keys.append(table[dp['key']])
#         delays.append((table[dp['key']], dp['delay']))

#     oracle = {}
#     window = 5
#     o_preds = []
#     o_hits = 0

#     r_preds = []
#     r_hits = 0

#     l_preds = []
#     l_hits = 0

#     print(f"\ntrial {trial_number + 1}")

#     # prediction loop
#     for i, key in enumerate(keys): 

#         try:
#             ngram = keys[i:i + window]
#             target = keys[i + window]
#         except IndexError:
#             break

#         # aaronson oracle
#         ngram_str = "".join(map(str, ngram))
#         try:
#             # use string of ngram as key for oracle dictionary
#             o_pred = oracle[ngram_str]
#             # assess which key has been seen most following this ngram
#             if o_pred["0"] == o_pred["1"]: o_pred = randrange(1)
#             else: o_pred = 0 if o_pred["0"] > o_pred["1"] else 1
            
#         except KeyError:
#             # if ngram not already in oracle, predict randomly
#             o_pred = randrange(1)

#         # assess prediction 
#         if o_pred == target: o_hits += 1
#         # append prediction to list
#         o_preds.append(o_pred)
        
#         # update oracle with new key
#         oracle[ngram_str] = {"0": 0, "1": 0}
#         oracle[ngram_str][str(target)] += 1


#     for i, key in enumerate(keys): 

#         try:
#             ngram = keys[i:i + seq_len]
#             target = keys[i + seq_len]
#         except IndexError:
#             break

#         # rnn prediction
#         ngram_tsr = torch.tensor(ngram).reshape(1, -1, 1).float().to(device)
#         with torch.no_grad(): r_pred = rnn(ngram_tsr).item()
#         r_pred = 0 if r_pred < 0.5 else 1
#         if r_pred == target: r_hits += 1
#         r_preds.append(r_pred)

#         # # lstm prediction
#         # with torch.no_grad(): l_pred = lstm(ngram_tsr).item()
#         # l_pred = 0 if l_pred < 0.5 else 1
#         # if l_pred == target: l_hits += 1
#         # l_preds.append(l_pred)


#     o_accuracy = (o_hits / len(o_preds)) * 100
#     print(f"Oracle accuracy: {o_accuracy:.2f}%")

#     r_accuracy = (r_hits / len(r_preds)) * 100
#     print(f"RNN accuracy: {r_accuracy:.2f}%")

    # l_accuracy = (l_hits / len(l_preds)) * 100
    # print(f"LSTM accuracy: {l_accuracy:.2f}%")



# def predict(input):
#     input = 0 if input == 'left' else 1
#     input = torch.FloatTensor(input).unsqueeze(0).unsqueeze(-1).to(device)
#     with torch.no_grad(): prediction = model(input).item()
#     prediction = (prediction > 0.5).float()
#     print("prediction: ", prediction)
#     prediction = 'right' if prediction == 1 else 'left'  # Convert to 'left' or 'right'

#     return prediction

# """ Oracle """

# print("Ready!")
# correct = 0
# keystrokes = []
# prediction = None

# def predict(input):
#     input = 0 if input == 'left' else 1
#     input = torch.FloatTensor(input).unsqueeze(0).unsqueeze(-1).to(device)
#     with torch.no_grad(): prediction = model(input).item()
#     prediction = (prediction > 0.5).float()
#     print("prediction: ", prediction)
#     prediction = 'right' if prediction == 1 else 'left'  # Convert to 'left' or 'right'

#     return prediction

# def on_keypress(e):
#     global correct, keystrokes, prediction

#     if e.event_type == 'down' and e.name in ['left', 'right']:
#         print("keystrokes: ", keystrokes)
#         if len(keystrokes) > seq_len:
#             if e.name == prediction: correct += 1
#             accuracy = (correct / (len(keystrokes) - seq_len)) * 100
#             print(f"Predicted: {prediction}, Actual: {e.name}, Accuracy: {accuracy:.2f}%")

#         if e.name == 'left':
#             keystrokes.append('left')
#         elif e.name == 'right':
#             keystrokes.append('right')

#         if (len(keystrokes) >= seq_len):
#             prediction = predict(keystrokes[-seq_len:])

# keyboard.hook(on_keypress, suppress=True)
# keyboard.wait('esc')  # Use 'esc' key to stop the listener
