import torch
import os, sys
from pprint import pprint

from load_data2 import load_data

for module in ['models']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from model4 import LSTM4
from model5 import LSTM5

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

""" Inference """

def inference(model_names, directories=None, logging=True):
    if not isinstance(model_names, list):
        model_names = [model_names]

    results = {}

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

        # if params contains segment length
        seg_len = params.get('segment_length', None)

        # load test data
        _, _, test_loader = load_data(directories, seg_len)
        
        # average accuracy for a model across all test trials
        results[model_name] = { "accuracy": 0 }

        if not seg_len:
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
                    
                    # store prediction and accuracy at each step
                    predictions.append(prediction)
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

        else:
            for trial, segments in enumerate(test_loader):
                correct = 0
                accuracies = []
                predictions = []

                # run inference on each trial
                for i in range(len(segments) - 1):
                    X = segments[i][:, :seg_len, :].to(device)
                    y = segments[i][:, seg_len, 0].item()

                    # make prediction
                    with torch.no_grad():
                        prediction = 0 if (model(X).item()) < 0.5 else 1
                    if prediction == y: correct += 1
                    
                    # store prediction and accuracy at each step
                    predictions.append(prediction)
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


if __name__ == "__main__":
    results = inference()