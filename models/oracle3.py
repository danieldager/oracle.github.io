# TODO: divide test data into trial types

import os, sys
import numpy as np
from random import randrange

def oracle(window_sizes=[5], trial_types=None, logging=True):
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]


    """ Load Data """

    # use current directory to set trials path
    cwd = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(cwd, '..', 'trials'))

    # if no trial types, get names of all subdirectories in trials
    if not trial_types:
        trial_types = os.listdir(base_path)

    # get directory names for each trial type
    trial_types = [os.path.join(base_path, t) for t in trial_types]

    # load numpy arrays, extract decisions
    sequences = []
    for t in trial_types:
        for trial in os.listdir(t):
            sequence = np.load(os.path.join(t, trial))
            sequences.append(sequence[0, :])

    # test data split
    split_ratio = 0.1
    total_size = len(sequences)
    test_size = int(split_ratio * total_size)
    test_data = sequences[-test_size:]
        

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
    r = oracle([3,4,5,6,7,8,9,10])