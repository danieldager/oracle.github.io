# TODO: test data needs to be a split of trial types
# TODO: load a structure which also includes the trial names
# TODO: be able to load specific trials from a list

# TODO: run feedback trials

import os
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

""" 
Notes 

Vary => models for which the inputs are segments of varying lengths
Fixed => models for which the inputs are segments of a fixed length
"""

split_ratio = 0.1

""" Dataset Classes """

class SegmentDatasetVary(Dataset):
    def __init__(self, data):
        self.segments = self.segment_sequences(data)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        segment = self.segments[i]
        X = segment[:-1]

        # check if segment[-1] is a zero index tensor
        if segment[-1].shape == torch.Size([]):
            y = segment[-1]
        else:
            y = segment[-1][0]
        
        return X, y

    def segment_sequences(self, data):
        segments = []
        for sequence in data:
            for i in range(2, len(sequence) + 1):
                segments.append(sequence[:i])
        return segments

# custom collate function to pad sequences
def collate_vary(batch):
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


class SegmentDatasetFixed(Dataset):
    def __init__(self, data, segment_length):
        self.segments = self.segment_sequences(data, segment_length)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        segment = self.segments[i]
        X = segment[:-1]

        # check if segment[-1] is a zero index tensor
        if segment[-1].shape == torch.Size([]):
            y = segment[-1]
        else:
            y = segment[-1][0]

        return X, y

    def segment_sequences(self, data, segment_length):
        segments = []
        for sequence in data:
            for i in range(segment_length, len(sequence) + 1):
                segments.append(sequence[i-segment_length:i])
        return segments

# Custom collate function without padding
def collate_fixed(batch):
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences).to(device)
    labels = torch.tensor(labels).float().unsqueeze(1).to(device)
    return sequences, labels


class TestDatasetVary(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        segments = [sequence[:i] for i in range(1, len(sequence) + 1)]
        return segments
        
def collate_test(batch):
    sequences = batch[0]
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths


class TestDatasetFixed(Dataset):
    def __init__(self, sequences, segment_length):
        self.sequences = sequences
        self.seg_len = segment_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        end = len(sequence) - self.seg_len - 2
        segments = [sequence[i:i + self.seg_len + 1] for i in range(0, end)]
        return segments


""" Load Data """

def get_sequences(folders, path):
    # get names of all files in list of folders
    files = []
    for folder in folders:
        folder_path = os.path.join(path, folder)
        files += [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    # extract numpy arrays
    sequences = []
    for file in files:
        sequence = np.load(file)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequences.append(sequence.T)

    return sequences


def load_data(train_folders=None, test_folders=None, inputs=None):

    # use script path to build absolute path to trial data
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, '..', 'trials')
    path = os.path.abspath(path)

    # get names of all folders (subject + condition) in trials folder 
    folder_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # if arguments are provided, get paths from specified folders
    if train_folders or test_folders:
        
        # get paths to train folders
        if train_folders:
            train_paths = [os.path.join(path, folder) for folder in folder_names if folder in train_folders]

        # if no train folders are specified, get all non-test folders
        else:
            train_paths = [os.path.join(path, folder) for folder in folder_names if folder not in test_folders]

        # get sequences from train folders and shuffle
        train_data = get_sequences(train_paths, path)

        # get paths to test folders
        if test_folders:
            test_paths = [os.path.join(path, folder) for folder in folder_names if folder in test_folders]

            # get sequences from test folders
            test_data = get_sequences(test_paths, path)
        
        # if no test folders are specified, split train data
        else:
            test_size = int(split_ratio * len(sequences))
            train_size = len(sequences) - test_size
            train_data, test_data = random_split(sequences, [train_size, test_size])
            
    # if no arguments are specified, split all trial data
    else:
        sequences = get_sequences(folder_names, path)

        # if inputs are specified, extract only those columns
        if inputs and len(inputs) != 3:
            if 'delay' in inputs:
                # extract the first two columns
                sequence = sequence[:, :2]
            elif 'pupil' in inputs:
                # extract the first and third columns
                sequence = sequence[:, [0, 2]]
            else:
                # extract only the first column
                sequence = sequence[:, 0]

        # split test data
        test_size = int(split_ratio * len(sequences))
        train_size = len(sequences) - test_size
        train_data, test_data = random_split(sequences, [train_size, test_size])

    # split train and eval data
    eval_size = int(split_ratio * len(train_data))
    train_size = len(train_data) - eval_size
    train_data, eval_data = random_split(train_data, [train_size, eval_size])

    return train_data, eval_data, test_data


def prune(data, input):
    pruned_data = []
    
    for sequence in data:
        if input == 'delay':
            pruned_data.append(sequence[:, :2])
        
        elif input == 'pupil':
            pruned_data.append(sequence[:, [0, 2]])
        
        elif input == 'decision':
            pruned_data.append(sequence[:, 0])

        else:
            print("Valid inputs are 'decision', 'delay', or 'pupil'")
            return None

    return pruned_data


def load_data_vary(train_data, eval_data, test_data, batch_size):
    train_dataset = SegmentDatasetVary(train_data)
    eval_dataset = SegmentDatasetVary(eval_data)
    test_dataset = TestDatasetVary(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_vary)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_vary)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)

    return train_loader, eval_loader, test_loader

def test_data_vary(test_data):
    test_dataset = TestDatasetVary(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)
    return test_loader


def load_data_fixed(train_data, eval_data, test_data, batch_size, segment_length):
    train_dataset = SegmentDatasetFixed(train_data, segment_length)
    eval_dataset = SegmentDatasetFixed(eval_data, segment_length)
    test_dataset = TestDatasetFixed(test_data, segment_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fixed)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fixed)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, eval_loader, test_loader

def test_data_fixed(test_data, segment_length):
    test_dataset = TestDatasetFixed(test_data, segment_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader


if __name__ == '__main__':
    d = load_data()
