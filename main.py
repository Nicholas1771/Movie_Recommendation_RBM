# Import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import RBM

# Import the dataset
movies = pd.read_csv('data/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('data/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Prepare training set and test set
training_set = pd.read_csv('data/ml-100k/u1.base', engine='python', header=None, delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('data/ml-100k/u1.test', engine='python', header=None, delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Get the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Create the matrix for the RBM
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings_list = np.zeros(nb_movies)
        ratings_list[id_movies - 1] = id_ratings
        new_data.append(list(ratings_list))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Convert data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings to 1s and 0s
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the RBM object
# Number of visible nodes
nv = len(training_set[0])
# Number of hidden nodes
nh = 100
batch_size = 100
rbm = RBM(nv=nv, nh=nh)



