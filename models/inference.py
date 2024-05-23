import os, sys
from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from oracle2 import oracle_inference
from model3 import LSTM, load_data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


""" Test Dataset """

class TestDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        segments = [sequence[:i] for i in range(1, len(sequence) + 1)]
        return segments
        
def collate_fn(batch):
    sequences = batch[0]
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths


""" Run Inference """

def run_inference(test_loader, model_names=['0521-2327'], logging=True):

    # Get the current directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', 'saved_models')
    base_path = os.path.abspath(base_path)

    # load models and hyperparameters
    for model_name in model_names:
        checkpoint = torch.load(os.path.join(base_path, f"{model_name}.pt"))
        model = checkpoint['model']
        params = checkpoint['params']
        model.to(device)

        # print model details
        if logging:
            print(f"Model {model_name}\n")
            pprint(params), print("\n")

        all_accuracies = []
        all_predictions = []

        for trial, (segments, lengths) in enumerate(test_loader):
            correct = 0
            accuracies = []
            predictions = []

            # run inference on each trial
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

        # calculate average accuracy of model over all trials
        average_accuracy = sum([trial[-1] for trial in all_accuracies]) / len(test_loader)
        if logging:
            print(f"\nModel {model_name} accuracy: {average_accuracy:.2f}%\n")

    return average_accuracy, all_accuracies, all_predictions

    # Report accuracy of aaronson oracle
    # oracle_inference(sequences, window_size=5)

if __name__ == "__main__":
    run_inference()