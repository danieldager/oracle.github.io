import os, sys
from random import randrange

for module in ['actions']:
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '..', module)
    sys.path.append(os.path.abspath(path))

from load_data import load_data


def oracle(window_sizes, test_data, cutoff=False, logging=True):
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    # convert test_data to numpy array and extract first column
    test_data = [sequence.numpy()[:, 0] for sequence in test_data]

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
            for i in range(len(sequence) - window_size - 2): 
                ngram = sequence[i:i + window_size]
                ngram_str = "".join(map(str, ngram))
                target = sequence[i + window_size]
                
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
                if cutoff:
                    cut = 20 - window_size + 1
                    if i >= cut:
                        if prediction == target: correct += 1
                        accuracy = 100 * correct / (i + 1 - cut)
                        
                        results[window_size][trial_number+1]["accuracies"].append(accuracy)
                        results[window_size][trial_number+1]["predictions"].append(prediction)

                elif i >= window_size:
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

            # if accuracy < 40:
            #     print(sequence)

        # print average accuracy of oracle
        average_accuracy /= len(test_data)
        results[window_size]["accuracy"] = average_accuracy
        
        if logging:
            print(f"\nAvg Accuracy: {average_accuracy:.2f}%\n")

    return results


if __name__ == "__main__":
    _, _, test_data = load_data()
    r = oracle(test_data, [3, 6, 9])