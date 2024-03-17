import os
import numpy as np
from pprint import pformat
from dotenv import load_dotenv
from pymongo import MongoClient

from scipy.special import kl_div
from scipy.stats import chi2_contingency, norm

''' Tests '''

# define significance level
alpha = 0.05

def chi_sqrd(sequence):
    # calculate frequencies
    observed_frequencies = [sequence.count(0), sequence.count(1)]
    expected_frequencies = [len(sequence) / 2, len(sequence) / 2]

    p_value = chi2_contingency([observed_frequencies, expected_frequencies]).pvalue

    return p_value

def ww_runs(sequence):
    n1 = sequence.count(0)
    n2 = sequence.count(1)

    # calculate observed # of runs
    runs = 1 + sum(sequence[i] != sequence[i+1] for i in range(len(sequence)-1))
    
    # calculate expected # of runs and variance
    runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
    runs_var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))

    # calculate z-score and p-value
    z = (runs - runs_exp) / np.sqrt(runs_var)
    p_value = 2 * norm.cdf(-abs(z))

    return p_value

def markov_test(sequence):
    # ideal transition matrice
    imatrix = np.array([[0.5, 0.5], [0.5, 0.5]])

    # observed transition matrice
    transitions = np.zeros((2, 2))

    for i in range(len(sequence)-1):
        transitions[sequence[i]][sequence[i+1]] += 1

    tmatrix = transitions / len(sequence)

    # Flattening the matrices, handle zero probabilities
    epsilon = 1e-10
    imatrix = imatrix.flatten() + epsilon
    tmatrix = tmatrix.flatten() + epsilon

    # Calculating KL Divergence
    kl_divergence = np.sum(kl_div(imatrix, tmatrix))

    return kl_divergence

def autocorrelation(sequence, max_lag=10):
    sequence = np.array([int(x) for x in sequence])
    result = [np.corrcoef(sequence[:-i], sequence[i:])[0, 1] for i in range(1, max_lag+1)]
    return result


''' Main '''

load_dotenv()
client = MongoClient(os.environ['MONGO_URL'])
database = client['Oracle']
collection = database['trials']

tested = []

trials = collection.find()
for trial in trials:

    # extract keypresses, convert L and R to 0 and 1
    data = trial['data']
    table = {'L': 0, 'R': 1}
    sequence = [table[d['key']] for d in data]

    print("".join(str(s) for s in sequence))
    break

    # calculate entropy and chi-squared
    chi2_pvalue = chi_sqrd(sequence)

    # calculate wald-wolfowitz runs test
    runs_pvalue = ww_runs(sequence)

    # calculate kl divergence from ideal markov chain
    markov_kldg = markov_test(sequence)

    trial = {
        'uuid': trial['uuid'],
        'size': len(data),
        'chi2_pvalue': round(chi2_pvalue, 5),
        'runs_pvalue': round(runs_pvalue, 5),
        'markov_kldg': round(markov_kldg, 5),
    }

    tested.append(trial)

# pretty print tested
print(pformat(tested))



# input_file = 'path/to/your/textfile.txt'  # Replace with your text file path
# output_file = 'path/to/your/outputfile.bin'  # Replace with your desired output file path

# text_to_binary(input_file, output_file)