import numpy as np

from scipy.special import rel_entr
from scipy.stats import chisquare, norm


''' Tests '''

# significance level
alpha = 0.05


def chi_sqrd(sequence):
    # calculate frequencies of 0s and 1s
    observed_frequencies = [np.sum(sequence == 0), np.sum(sequence == 1)]
    expected_frequencies = [len(sequence) / 2, len(sequence) / 2]
    
    # perform Chi-squared test
    _, p_value = chisquare(observed_frequencies, f_exp=expected_frequencies)
    
    return p_value


def ww_runs(sequence):
    # calculate n1 and n2 (count of 1s and 0s respectively)
    n1 = np.sum(sequence)
    n2 = len(sequence) - n1
    
    # calculate observed number of runs
    runs = 1 + np.sum(sequence[:-1] != sequence[1:])
    
    # calculate expected number of runs and variance
    runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
    runs_var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    
    # calculate z-score and p-value
    z = (runs - runs_exp) / np.sqrt(runs_var)
    p_value = 2 * norm.cdf(-abs(z))
    
    return p_value


def markov_test(sequence):
    # ideal transition matrix
    imatrix = np.array([[0.5, 0.5], [0.5, 0.5]])

    # observed transition matrix
    transitions = np.zeros((2, 2))

    for i in range(len(sequence)-1):
        transitions[int(sequence[i])][int(sequence[i+1])] += 1

    # normalize the transition matrix to get probabilities
    row_sums = transitions.sum(axis=1)
    tmatrix = transitions / row_sums[:, np.newaxis]

    # flattening the matrices, handle zero probabilities
    epsilon = 1e-10
    imatrix = imatrix.flatten() + epsilon
    tmatrix = tmatrix.flatten() + epsilon

    # calculating KL Divergence
    kl_divergence = np.sum(rel_entr(imatrix, tmatrix))

    return kl_divergence


def autocorrelation(sequence, max_lag=15):
    sequence = np.array([int(x) for x in sequence])
    
    results = []    
    for lag in range(1, max_lag + 1):
        # calculate correlation for the current lag
        corr = np.corrcoef(sequence[:-lag], sequence[lag:])[0, 1]
        results.append(corr)
    
    average = np.mean(results)
    
    return average

def test_randomness(sequence):
    # convert sequence to numpy array, extract 1st column
    sequence = np.array(sequence)[:, 0]

    # calculate entropy and chi-squared
    chi2_pvalue = chi_sqrd(sequence)

    # calculate wald-wolfowitz runs test
    runs_pvalue = ww_runs(sequence)

    # calculate kl divergence from ideal markov chain
    markov_kldg = markov_test(sequence)

    ac_value = autocorrelation(sequence)

    results = {
        # 'chi2_pvalue': round(chi2_pvalue, 2),
        # 'runs_pvalue': round(runs_pvalue, 2),
        # 'markov_kldg': round(markov_kldg, 2),
        # 'autocorrelation': round(ac_value, 2)

        'chi2_pvalue': chi2_pvalue,
        'runs_pvalue': runs_pvalue,
        'markov_kldg': markov_kldg,
        'autocorrelation': ac_value,
    }

    return results
