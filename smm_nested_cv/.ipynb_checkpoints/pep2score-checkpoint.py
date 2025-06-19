#!/usr/bin/env python
# coding: utf-8

# # Description
# Scoring peptides to a weight matrix

# ## Python Imports

# In[1]:

import numpy as np
from pprint import pprint

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser(description="Peptide Scoring to a weight Matrix")
parser.add_argument("-f", action="store", dest="PEPTIDES_FILE", required=True, type=str, help="File with peptides")
parser.add_argument("-mat", action="store", dest="MAT_FILE", required=True, type=str, help="File with the PSSM matrix")
args = parser.parse_args()

peptides_file = args.PEPTIDES_FILE
mat_file = args.MAT_FILE

# Load the functions of pep2score just once

def initialize_matrix(peptide_length, alphabet):

    init_matrix = [0]*peptide_length

    for i in range(0, peptide_length):

        row = {}

        for letter in alphabet: 
            row[letter] = 0.0

        #fancy way:  row = dict( zip( alphabet, [0.0]*len(alphabet) ) )

        init_matrix[i] = row
        
    return init_matrix

def from_psi_blast(file_name):

    f = open(file_name, "r")
    
    nline = 0
    for line in f:
    
        sline = str.split( line )
        
        if nline == 0:
        # recover alphabet
            alphabet = [str]*len(sline)
            for i in range(0, len(sline)):
                alphabet[i] = sline[i]
                
            matrix = initialize_matrix(peptide_length, alphabet)
        
        else:
            i = int(sline[0])
            
            for j in range(2,len(sline)):
                matrix[i-1][alphabet[j-2]] = float(sline[j])
                
        nline+= 1
            
    return matrix

def score_peptide(peptide, matrix):
    acum = 0
    for i in range(0, len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum

# Example: Print values to confirm
print(f"#Peptides file: {peptides_file}")
print(f"#PSSM file: {mat_file}")


# ## DEFINE THE PATH TO YOUR COURSE DATA DIRECTORY

# In[2]:


data_dir = "/home/luis_ubuntu/unixdir/Peptide_Binding/Data/"


# Read evaluation data
#evaluation_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_2/A0201.eval"
# evaluation_file = data_dir + "PSSM/A0201.eval"
evaluation_file = peptides_file
print(evaluation_file)
evaluation = np.loadtxt(evaluation_file, dtype=str).reshape(-2,3)
evaluation_peptides = evaluation[:, 0]
evaluation_targets = evaluation[:, 1].astype(float)

peptide_length = len(evaluation_peptides[0])
# Define which PSSM file to use (file save from pep2mat)
# pssm_file = data_dir + "PSSM/w_matrix_test"
pssm_file = mat_file

w_matrix = from_psi_blast(pssm_file)
evaluation_predictions = []
for i in range(len(evaluation_peptides)):
    score = score_peptide(evaluation_peptides[i], w_matrix)
    evaluation_predictions.append(score)
    print (evaluation_peptides[i], score, evaluation_targets[i])
    
pcc = pearsonr(evaluation_targets, evaluation_predictions)
print("#PCC: ", pcc[0])





