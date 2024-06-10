import math 
import torch
import os, sys
from pprint import pprint
from matplotlib import pyplot as plt

from load_data import test_data_fixed, test_data_vary

for module in ['models']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from oracle import oracle
from lstm_vary import LSTM_Vary
from lstm_fixed import LSTM_Fixed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def plot(o_trials, m_trials, window_size, model_name, trial_numbers=None, cutoff=False):
    if trial_numbers is None:
        if len(o_trials) >= 12:
            trial_numbers = list(range(1, 13))
        else:
            trial_numbers = list(range(1, len(o_trials) + 1))

    # if trial numbers is not a list of trial numbers, make it one
    if not isinstance(trial_numbers, list):
        trial_numbers = [trial_numbers]
    
    o_results = oracle(window_size, o_trials, cutoff, logging=False)
    m_results = inference(model_name, m_trials, cutoff, logging=False)

    cols = 4
    num_trials = len(trial_numbers) 
    rows = math.ceil(num_trials / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()

    for idx, trial in enumerate(trial_numbers):
        o_accuracies = o_results[window_size][trial]['accuracies']
        m_accuracies = m_results[model_name][trial]['accuracies']

        ax = axes[idx]
        ax.plot(o_accuracies, label='Oracle', lw=2.5)
        ax.plot(m_accuracies, label='Model', lw=2.5)
        ax.set_title(f'Trial {trial}')
        ax.set_ylabel('Accuracy (%)')
        if cutoff:
            ax.set_xlim(20, len(o_accuracies))
        ax.set_ylim(0, 100)
        if cutoff:
            ax.set_ylim(30, 100)
        ax.legend()
        ax.grid(True)

    # Hide any empty subplots
    for idx in range(num_trials, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


    return None


""" Inference """

def inference(model_names, test_data, cutoff=False, logging=True):
    if not isinstance(model_names, list):
        model_names = [model_names]

    results = {}

    # Get the current directory of this script
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, '..', 'saved_models')
    path = os.path.abspath(path)

    # load models and hyperparameters
    for model_name in model_names:
        checkpoint = torch.load(os.path.join(path, f"{model_name}.pt"))
        model = checkpoint['model']
        params = checkpoint['params']
        model.to(device)

        # print model details
        if logging:
            print(f"Model {model_name}\n")
            pprint(params), print("\n")

        # average accuracy for a model across all test trials
        results[model_name] = { "accuracy": 0 }

        # get model parameters
        input_size = params.get('input_size', None)
        segment_length = params.get('segment_length', 0)

        # inference for LSTM_Fixed
        if segment_length:
            test_loader = test_data_fixed(test_data, segment_length)
        
            for trial, segments in enumerate(test_loader):
                correct = 0
                accuracies = []
                predictions = []

                # run inference on each trial
                for i in range(len(segments) - 1):
                    
                    if input_size == 1:
                        X = segments[i][:, :segment_length].to(device)
                        y = segments[i][:, segment_length].item()
                    else:
                        X = segments[i][:, :segment_length, :].to(device)
                        y = segments[i][:, segment_length, 0].item()

                    # make prediction
                    with torch.no_grad():
                        prediction = 0 if (model(X).item()) < 0.5 else 1
                    
                    # store prediction and accuracy at each step
                    if cutoff:
                        cut = 20 - segment_length
                        if i >= cut:
                            predictions.append(prediction)
                            if prediction == y: correct += 1
                            accuracies.append(100 * correct / (i + 1 - cut))
                    else:
                        predictions.append(prediction)
                        if prediction == y: correct += 1
                        accuracies.append(100 * correct / (i + 1))

                # add final accuracy of the trial to average
                results[model_name]['accuracy'] += accuracies[-1]

                # store trial predictions and accuracies for each trial
                results[model_name][trial + 1] = {
                    "accuracies": accuracies,
                    "predictions": predictions
                }
                
                if logging:
                    print(f"trial {trial + 1} accuracy: {accuracies[-1]:.2f}%")
     
        # inference for LSTM_Full
        else:
            test_loader = test_data_vary(test_data)
        
            for trial, (segments, lengths) in enumerate(test_loader):
                correct = 0
                accuracies = []
                predictions = []

                # run inference on each trial
                for i in range(len(segments) - 1):
                    X = segments[i].unsqueeze(0).to(device)

                    if input_size == 1:
                        y = segments[i + 1][i + 1].item()
                    else:
                        y = segments[i + 1][i + 1][0].item()

                    length = torch.tensor([lengths[i]]).to(device)

                    # make prediction
                    with torch.no_grad():
                        prediction = 0 if (model(X, length).item()) < 0.5 else 1
                    
                    # store prediction and accuracy at each step
                    if not cutoff or i >= 20:
                        predictions.append(prediction)
                        if prediction == y: correct += 1
                        if cutoff:
                            accuracies.append(100 * correct / (i - 20 + 1))
                        else:
                            accuracies.append(100 * correct / (i + 1))

                # add final accuracy of the trial to average
                results[model_name]['accuracy'] += accuracies[-1]

                # store trial predictions and accuracies for each trial
                results[model_name][trial + 1] = {
                    "accuracies": accuracies,
                    "predictions": predictions
                }
                
                if logging:
                    print(f"trial {trial + 1} accuracy: {accuracies[-1]:.2f}%")

        # calculate average accuracy of model over all trials
        results[model_name]['accuracy'] /= len(test_loader)
        
        if logging:
            print(f"\nAvg Accuracy: {results[model_name]['accuracy']:.2f}%\n")
 
    return results


# if __name__ == '__main__':
#     model_names = ['0604-0037', '0604-0936']

#     results = inference(model_names, test_data)
