import numpy as np
import random
import copy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utilities_smm import encode, cumulative_error, predict, cal_mse, gradient_descent, vector_to_matrix, to_psi_blast

parser = ArgumentParser(prog="SMM_GradientDescent", description="Train a simple SMM_GD model")

parser.add_argument("-l", dest="LAMB", type=float, default=0.01, help="Lambda (default: 0.01)")
parser.add_argument("-t", dest="TRAINING_FILE", type=str, help="File with training data")
parser.add_argument("-e", dest="EVALUATION_FILE", type=str, help="File with evaluation data")
parser.add_argument("-epi", dest="EPSILON", type=float, default=0.05, help="Epsilon (default: 0.05)")
parser.add_argument("-s", dest="SEED", type=int, default=1, help="Seed for random numbers (default: 1)")
parser.add_argument("-i", dest="EPOCHS", type=int, default=100, help="Number of epochs to train (default: 100)")

# Parse arguments
args = parser.parse_args()

lambdas = args.LAMB
epsilon = args.EPSILON

data_dir = "/Users/mblanco/Desktop/DTU/AlgorithmsInBioinf/code/Project/Peptide_Binding/Data/"

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

epochs = args.EPOCHS

gerror_plot = []
mse_plot = []
train_mse_plot = []
eval_mse_plot = []
train_pcc_plot = []
eval_pcc_plot = []

for e in range(0, epochs):
    for lamb in lambdas:
        lamb_N = lamb / N
        for eps in epsilon:
            for i in range(0, N):
                ix = np.random.randint(0, N)
                peptide = peptides[ix]
                y_target = y[ix]
                y_pred = np.dot(peptide, weights)
                gradient_descent(y_pred, y_target, peptide, weights, lamb_N, eps)

                gerr, mse = cumulative_error(peptides, y, lamb, weights)
                gerror_plot.append(gerr)
                mse_plot.append(mse)

                train_pred = predict(peptides, weights)
                train_mse = cal_mse(y, train_pred)
                train_mse_plot.append(train_mse)
                train_pcc = pearsonr(y, train_pred)
                train_pcc_plot.append(train_pcc[0])

                eval_pred = predict(evaluation_peptides, weights)
                eval_mse = cal_mse(evaluation_targets, eval_pred)
                eval_mse_plot.append(eval_mse)
                eval_pcc = pearsonr(evaluation_targets, eval_pred)
                eval_pcc_plot.append(eval_pcc[0])

                print("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, eval_pcc[0], eval_mse)

fig = plt.figure(figsize=(10, 10), dpi=80)

x = np.arange(0, len(gerror_plot))

plt.subplot(2, 2, 1)
plt.plot(x, gerror_plot)
plt.ylabel("Global Error", fontsize=10)
plt.xlabel("Iterations", fontsize=10)

plt.subplot(2, 2, 2)
plt.plot(x, mse_plot)
plt.ylabel("MSE", fontsize=10)
plt.xlabel("Iterations", fontsize=10)

x = np.arange(0, len(train_mse_plot))

plt.subplot(2, 2, 3)
plt.plot(x, train_mse_plot, label="Training Set")
plt.plot(x, eval_mse_plot, label="Evaluation Set")
plt.ylabel("Mean Squared Error", fontsize=10)
plt.xlabel("Iterations", fontsize=10)
plt.legend(loc='upper right')

plt.subplot(2, 2, 4)
plt.plot(x, train_pcc_plot, label="Training Set")
plt.plot(x, eval_pcc_plot, label="Evaluation Set")
plt.ylabel("Pearson Correlation", fontsize=10)
plt.xlabel("Iterations", fontsize=10)
plt.legend(loc='upper left')

matrix = vector_to_matrix(weights, alphabet)
to_psi_blast(matrix)

evaluation_peptides = evaluation[:, 0]
evaluation_peptides = np.array(encode(evaluation_peptides, sparse, alphabet))

evaluation_targets = np.array(evaluation[:, 1], dtype=float)

y_pred = []
for i in range(0, len(evaluation_peptides)):
    y_pred.append(np.dot(evaluation_peptides[i].T, weights))

y_pred = np.array(y_pred)

pcc = pearsonr(evaluation_targets, np.array(y_pred))
print("PCC: ", pcc[0])

plt.scatter(y_pred, evaluation_targets)