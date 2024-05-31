# TODO: generalize base_path setting (oracle3)
# TODO: test data needs to be a split of trial types
# TODO: load a structure which also includes the trial names
# TODO: be able to load specific trials from a list

import os, sys
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


""" Dataset Classes """

class SegmentDataset(Dataset):
    def __init__(self, data):
        self.segments = self.segment_sequences(data)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        segment = self.segments[i]
        X = segment[:-1]
        y = segment[-1][0]
        return X, y

    def segment_sequences(self, data):
        segments = []
        for sequence in data:
            for i in range(2, len(sequence) + 1):
                segments.append(sequence[:i])
        return segments

# custom collate function to pad sequences
def collate_fn1(batch):
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


class TestDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        segments = [sequence[:i] for i in range(1, len(sequence) + 1)]
        return segments
        
def collate_fn2(batch):
    sequences = batch[0]
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths


class SegmentDataset2(Dataset):
    def __init__(self, data, segment_length):
        self.segments = self.segment_sequences(data, segment_length)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        segment = self.segments[i]
        X = segment[:-1]
        y = segment[-1][0]
        return X, y

    def segment_sequences(self, data, segment_length):
        segments = []
        for sequence in data:
            for i in range(segment_length, len(sequence) + 1):
                segments.append(sequence[i-segment_length:i])
        return segments

# Custom collate function without padding
def collate_fn3(batch):
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences).to(device)
    labels = torch.tensor(labels).float().unsqueeze(1).to(device)
    return sequences, labels


class TestDataset2(Dataset):
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

def load_data(directories, segment_length=None, batch_size=100):

    # Get the current directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', 'trials')
    base_path = os.path.abspath(base_path)
    
    # if no directories are specified, use all subdirectories under base_path
    if not directories:
        directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # if directories is not a list, convert to list
    if not isinstance(directories, list):
        directories = [directories]
    
    # TODO: divide test data into trial types
    file_paths = []
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        for file_name in os.listdir(full_path):
            file_paths.append(os.path.join(full_path, file_name))

    # convert numpy arrays to torch tensors
    sequences = []
    for file_path in file_paths:
        sequence = np.load(file_path)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequences.append(sequence.T) # transpose to (n, 3) shape


    # TODO: god this is so ugly 
    # split test data
    split_ratio = 0.1
    test_size = int(split_ratio * len(sequences))
    test_data = sequences[-test_size:]
    sequences = sequences[:-test_size]

    # split train and eval data
    split_ratio = 0.8
    train_size = int(split_ratio * len(sequences))
    eval_size = len(sequences) - train_size
    train_data, eval_data = random_split(sequences, [train_size, eval_size])


    if not segment_length:
        # segment sequences
        train_segments = SegmentDataset(train_data)
        eval_segments = SegmentDataset(eval_data)
        test_segments = TestDataset(test_data)

        fnA = collate_fn1
        fnB = collate_fn2

    else:
        # segment sequences
        train_segments = SegmentDataset2(train_data, segment_length)
        eval_segments = SegmentDataset2(eval_data, segment_length)
        test_segments = TestDataset2(test_data, segment_length)

        fnA = collate_fn3
        fnB = None

    # create data loaders
    train_loader = DataLoader(train_segments, batch_size=batch_size, shuffle=True, collate_fn=fnA)
    eval_loader = DataLoader(eval_segments, batch_size=batch_size, shuffle=False, collate_fn=fnA)
    test_loader = DataLoader(test_segments, batch_size=1, shuffle=False, collate_fn=fnB)

    return train_loader, eval_loader, test_loader


if __name__ == '__main__':
    train_loader, eval_loader, test_loader = load_data()
