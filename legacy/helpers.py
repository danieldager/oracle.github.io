import os, torch
from dotenv import load_dotenv
from pymongo import MongoClient


""" Database """

def load_data_from_db():
    load_dotenv()
    client = MongoClient(os.environ['MONGO_URL'])
    database = client['Oracle']
    collection = database['trials']

    # get all trials
    data = collection.find()
    trials = [trial['data'] for trial in data]

    # flatten and format data
    all_trials = [datum for trial in trials for datum in trial]
    all_trials = format_sequence(all_trials)

    return trials, all_trials


""" Formatting """

def format_sequence(sequence):
        return [[0 if e['key'] == 'L' else 1, e['delay']] for e in sequence]

# def format_datasets(data, seq_len):
#     features = []
#     targets = []

#     # feature is an array of seq_len elements
#     # target is an array with only the next element
#     for i in range(len(data) - seq_len - 1):
#         f = data[i : i+seq_len]
#         t = data[i+1+seq_len]
#         features.append(f)
#         targets.append(t)

#     # reshape to conform to lstm input
#     features = torch.tensor(features).reshape(-1, seq_len, 1).float()
#     targets = torch.tensor(targets).reshape(-1, 1).float()
    
#     return features, targets

def format_datasets(sequence, seq_len):
    features = []
    targets = []

    # Create sequences using keypress and delay data
    for i in range(len(sequence) - seq_len - 1):
        features.append(sequence[i:i + seq_len])
        targets.append(sequence[i + seq_len + 1])

    # Reshape for RNN/LSTM input
    features = torch.tensor(features, dtype=torch.float32).reshape(-1, seq_len, 2)
    targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    return features, targets
