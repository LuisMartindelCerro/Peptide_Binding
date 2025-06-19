import numpy as np
import random
import copy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utilities_smm import encode, cumulative_error, predict, cal_mse, gradient_descent, gradient_descent_batch

data_dir = "/Users/Administrator/Documents/unixdir/Peptide_Binding/Data/"

training_file = data_dir + "A0101/f000"
training = np.loadtxt(training_file, dtype=str)

evaluation_file = data_dir + "A0101/c000"
evaluation = np.loadtxt(evaluation_file, dtype=str)

alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

sparse_file = data_dir + "Matrices/sparse"
_sparse = np.loadtxt(sparse_file, dtype=float)
sparse = {}

for i, letter_1 in enumerate(alphabet):
    sparse[letter_1] = {}
    for j, letter_2 in enumerate(alphabet):
        sparse[letter_1][letter_2] = _sparse[i, j]

np.random.seed(1)

peptides = training[:, 0]
peptides = encode(peptides, sparse, alphabet)
N = len(peptides)

y = np.array(training[:, 1], dtype=float)

evaluation_peptides = evaluation[:, 0]
evaluation_peptides = encode(evaluation_peptides, sparse, alphabet)

evaluation_targets = np.array(evaluation[:, 1], dtype=float)

input_dim = len(peptides[0])
output_dim = 1
w_bound = 0.1
weights = np.random.uniform(-w_bound, w_bound, size=input_dim)

epochs = 100

gerror_plot = []
mse_plot = []
train_mse_plot = []
eval_mse_plot = []
train_pcc_plot = []
eval_pcc_plot = []


# 'C:\Users\Administrator\Documents\unixdir\Peptide_Binding\Data\A0101'
# peptides
# y
# weights
# lamb_N
# epsilon
