# TODO: divide test data into trial types

import os, sys
import numpy as np
from random import randrange

def oracle(window_sizes=[5], trial_types=['blind']):

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

    for window_size in window_sizes:

        print(f"\nOracle (window {window_size})\n")
        average_accuracy = 0
        
        for trial_number, sequence in enumerate(test_data):
            oracle = {}
            preds = []
            hits = 0

            # prediction loop
            for i in range(len(sequence)): 
                try:
                    ngram = sequence[i:i + window_size]
                    target = sequence[i + window_size]
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
                oracle[ngram_str][str(int(target))] += 1

            accuracy = (hits / len(preds)) * 100
            print(f"trial {trial_number + 1} accuracy: {accuracy:.2f}%")
            average_accuracy += accuracy

        average_accuracy /= len(test_data)
        print(f"\nOracle accuracy: {average_accuracy:.2f}%\n")


if __name__ == "__main__":
    oracle([3,4,5,6,7,8,9,10])