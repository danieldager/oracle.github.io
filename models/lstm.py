import torch
import keyboard
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



""" Hyperparameters """

seq_len = 5 # number of previous keystrokes to consider
batch_size = 16 # number of sequences to consider at once
hidden_size = 4 # number of hidden units in lstm
num_stacked_layers = 1 # number of stacked lstm layers
learning_rate = 0.01 # learning rate for optimizer
num_epochs = 10 # number of times to iterate over the entire dataset



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
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_dataset = LSTMDataset(X_train, y_train)
test_dataset = LSTMDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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

# create model, loss function, and optimizer
model = LSTM(1, hidden_size, num_stacked_layers).to(device)
# loss_function = nn.MSELoss()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch():
    model.train(True)
    running_loss = 0.0

    # i = batch index, (X, y) = batch
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # forward pass, compute loss
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()

        # backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print batch number and average loss every 10 batches
        # if i % 10 == 0:
        #     print(f'batch {i+1}: {running_loss / 10}')
        #     running_loss = 0.0

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for _, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            # forward pass, compute loss
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            running_loss += loss.item()
        
    print(f'validation loss: {running_loss / len(test_loader)}')
    print('*' * 30)


for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()



""" Oracle """

print("Ready!")
correct = 0
keystrokes = []
prediction = None

def predict(input):
    input = 0 if input == 'left' else 1
    input = torch.FloatTensor(input).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad(): prediction = model(input).item()
    print("prediction: ", prediction)
    prediction = 'right' if round(prediction) == 1 else 'left'  # Convert to 'left' or 'right'

    return prediction

def on_keypress(e):
    global correct, keystrokes, prediction

    if e.event_type == 'down' and e.name in ['left', 'right']:
        print("keystrokes: ", keystrokes)
        if len(keystrokes) > seq_len:
            if e.name == prediction: correct += 1
            accuracy = (correct / (len(keystrokes) - seq_len)) * 100
            print(f"Predicted: {prediction}, Actual: {e.name}, Accuracy: {accuracy:.2f}%")

        if e.name == 'left':
            keystrokes.append('left')
        elif e.name == 'right':
            keystrokes.append('right')

        if (len(keystrokes) >= seq_len):
            prediction = predict(keystrokes[-seq_len:])

keyboard.hook(on_keypress, suppress=True)
keyboard.wait('esc')  # Use 'esc' key to stop the listener