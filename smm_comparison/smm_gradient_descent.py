#!/usr/bin/env python
# coding: utf-8

# # SMM with Gradient Descent

# ## Python Imports

# In[1]:


import math
from pathlib import Path
import numpy as np
import random
from scipy.stats import pearsonr

from argparse import ArgumentParser

# Command-line parser setup
parser = ArgumentParser(prog="SMM_GradientDescent", description="Train a simple SMM_GD model")

parser.add_argument("-l", dest="LAMB", type=float, default=0.01, help="Lambda (default: 0.01)")
parser.add_argument("-t", dest="TRAINING_FILE", type=str, help="File with training data")
parser.add_argument("-epi", dest="EPSILON", type=float, default=0.05, help="Epsilon (default: 0.05)")
parser.add_argument("-s", dest="SEED", type=int, default=1, help="Seed for random numbers (default: 1)")
parser.add_argument("-i", dest="EPOCHS", type=int, default=100, help="Number of epochs to train (default: 100)")

# Parse arguments
args = parser.parse_args()

# Extract arguments into variables
lamb = args.LAMB
training_file = args.TRAINING_FILE
epsilon = args.EPSILON
seed = args.SEED
epochs = args.EPOCHS

# Print arguments (optional, for debug/logging)
#print(f"Lambda: {lamb}")
#print(f"Training file: {training_file}")
#print(f"Epsilon: {epsilon}")
#print(f"Seed: {seed}")
#print(f"Epochs: {epochs}")


# Load the functions of smm_gradient_descent just once

def encode(peptides, encoding_scheme, alphabet):
    
    encoded_peptides = []

    for peptide in peptides:

        encoded_peptide = []

        for peptide_letter in peptide:

            for alphabet_letter in alphabet:

                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])

        encoded_peptides.append(encoded_peptide)
        
    return np.array(encoded_peptides)

def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):
    
    # do is dE/dO
    do = y_pred - y_target
        
    for i in range(0, len(weights)):
        
        de_dw_i = do * peptide[i] + (2 * lamb_N * weights[i] * peptide[i])

        weights[i] -= epsilon * de_dw_i
        
def vector_to_matrix(vector, alphabet):
    
    rows = int(len(vector)/len(alphabet))
    
    matrix = [0] * rows
    
    offset = 0
    
    for i in range(0, rows):
        
        matrix[i] = {}
        
        for j in range(0, 20):
            
            matrix[i][alphabet[j]] = vector[j+offset] 
        
        offset += len(alphabet)

    return matrix

def to_psi_blast(matrix):

    # print to user
    
    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header)) 

    letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    for i, row in enumerate(matrix):

        scores = []

        scores.append(str(i+1) + " A")

        for letter in letter_order:

            score = row[letter]

            scores.append(round(score, 4))

        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores)) 

# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY

# In[2]:


data_dir = "/home/luis_ubuntu/unixdir/Peptide_Binding/Data/"


# ### Training Data

# In[3]:


#training_file = data_dir + "SMM/A0201_training"
#training_file = data_dir + "SMM/A2403_training"

training = np.loadtxt(training_file, dtype=str)


# ### Alphabet

# In[5]:


alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)


sparse_file = data_dir + "Matrices/sparse"
_sparse = np.loadtxt(sparse_file, dtype=float)
sparse = {}

for i, letter_1 in enumerate(alphabet):
    
    sparse[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        sparse[letter_1][letter_2] = _sparse[i, j]


# Random seed 
np.random.seed( seed )

# peptides
peptides = training[:, 0]
peptides = encode(peptides, sparse, alphabet)
N = len(peptides)

# target values
y = np.array(training[:, 1], dtype=float)

# weights
input_dim  = len(peptides[0])
output_dim = 1
w_bound = 0.1
weights = np.random.uniform(-w_bound, w_bound, size=input_dim)

# training epochs
#epochs = 100

# regularization lambda
# lamb = 1
# lamb = 10
#lamb = 0.01

# regularization lambda per target value
lamb_N = lamb/N

# learning rate
#epsilon = 0.01

# for each training epoch
for e in range(0, epochs):

    # for each peptide
    for i in range(0, N):

        # random index
        ix = np.random.randint(0, N)
        
        # get peptide       
        peptide = peptides[ix]

        # get target prediction value
        y_target = y[ix]
       
        # get initial prediction
        y_pred = np.dot(peptide, weights)

        # gradient descent 
        gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)

matrix = vector_to_matrix(weights, alphabet)
to_psi_blast(matrix)






