
from random import randrange

def oracle_inference(trials, window_size=5):
    print("\n\nOracle inference\n")
    average_accuracy = 0
    
    for trial_number, trial in enumerate(trials):
        oracle = {}
        preds = []
        hits = 0

        sequence = [0 if e['key'] == 'L' else 1 for e in trial]

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
            oracle[ngram_str][str(target)] += 1

        accuracy = (hits / len(preds)) * 100
        print(f"trial {trial_number + 1} accuracy: {accuracy:.2f}%")
        average_accuracy += accuracy

    average_accuracy /= len(trials)
    print(f"\nOracle accuracy: {average_accuracy:.2f}%")
