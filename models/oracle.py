# TODO: divide test data into trial types

import os, sys
import numpy as np
from random import randrange

for module in ['actions']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from load_data3 import split_data


def oracle(test_data, window_sizes=5, logging=True):
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    # convert test_data to numpy array and extract first column
    test_data = [sequence.numpy()[:, 0] for sequence in test_data]


    """ Inference """

    results = {}

    for window_size in window_sizes:
        average_accuracy = 0
        results[window_size] = {}

        if logging:
            print(f"\nOracle (window {window_size})\n")
        
        for trial_number, sequence in enumerate(test_data):
            oracle = {}
            correct = 0
            
            results[window_size][trial_number+1] = {
                "accuracies": [],
                "predictions": []
            }

            # prediction loop
            for i in range(len(sequence)): 
                try:
                    ngram = sequence[i:i + window_size + 1]
                    target = sequence[i + window_size + 1]
                except IndexError:
                    break

                ngram_str = "".join(map(str, ngram))
                
                try:
                    # use string of ngram as key for oracle dictionary
                    prediction = oracle[ngram_str]
                    
                    # assess which key has been seen most following this ngram
                    if prediction["0"] > prediction["1"]:
                        prediction = 0
                    elif prediction["0"] < prediction["1"]:
                        prediction = 1
                    
                    # if equal, predict randomly
                    else: prediction = randrange(2)

                # if ngram not already in oracle, predict randomly
                except KeyError: prediction = randrange(2)

                # assess prediction, update accuracy
                if prediction == target: correct += 1
                accuracy = 100 * correct / (i + 1)
                results[window_size][trial_number+1]["accuracies"].append(accuracy)
                results[window_size][trial_number+1]["predictions"].append(prediction)
                
                # update oracle with new key
                oracle[ngram_str] = {"0": 0, "1": 0}
                oracle[ngram_str][str(int(target))] += 1

            average_accuracy += accuracy
            
            if logging:
                print(f"trial {trial_number + 1} accuracy: {accuracy:.2f}%")

        # print average accuracy of oracle
        average_accuracy /= len(test_data)
        results[window_size]["accuracy"] = average_accuracy
        
        if logging:
            print(f"\nAvg Accuracy: {average_accuracy:.2f}%\n")

    return results


if __name__ == "__main__":
    _, _, test_data = split_data()
    r = oracle(test_data, [3, 6, 9])