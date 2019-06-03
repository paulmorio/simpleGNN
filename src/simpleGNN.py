"""
A simple linear implementation of the Graph Convolutional Neural Network applied in the 
context of node classification. It's aim is to show a fast implementation using only
python and numpy for pedagogic purposes.

For this example we will use the CORA citation network dataset and attempt to label
the nodes given a small sample of labeled ones.

Author: Paul Scherer (paulmorio)
Date: May 2019
MIT License
"""

import random
import numpy as np
import networkx as nx
import json
from collections import defaultdict


##
## Data Loading and Preprocessing
##
print("## Preprocessing Data ##")
# Cora NX Graph
cora = nx.read_edgelist(path="/home/morio/workspace/simpleGNN/data/cora_edges.csv", delimiter=",", nodetype=int) # we will keep the node type string as its used in the other data files

# Sparse node features dictionary
with open('/home/morio/workspace/simpleGNN/data/cora_features.json') as json_file:
	features_dict = defaultdict(list)
	json_str = json_file.read()
	features_dict_str = json.loads(json_str)
	for key in features_dict_str.keys():
		key_int = int(key)
		features_dict[key_int] = features_dict_str[key]
	del(features_dict_str)

# Node index to target dictionary
with open('/home/morio/workspace/simpleGNN/data/cora_target.csv') as fh:
	targets = defaultdict()
	lines = fh.readlines()
	for line in lines:
		id_target_pair = line.strip().split(",")
		idx, target = id_target_pair
		targets[int(idx)] = int(target)

# Adjacency matrix
num_nodes = cora.number_of_nodes()
identity = np.identity(num_nodes)
A = nx.to_numpy_matrix(G=cora, nodelist=range(num_nodes))
A_tilde = A+identity # Adjacency matrix with self loop

# Degree matrix
row_sum = np.sum(A, axis=0)
D = np.zeros(A.shape)
for i in range(num_nodes):
	D[i,i] = row_sum[0,i]
D_tilde = D+identity

# Node Feature matrix
features_list = [features_dict[x] for x in features_dict.keys()]
d = max([item for sublist in features_list for item in sublist]) # number of features
X = np.zeros((num_nodes, d+1))


for row in range(num_nodes):
	for col in features_dict[row]:
		X[row,col] = 1.0 # One Hot feature vector for each node

# 'Normalised' Degree Matrix
neg_rec_sqrt_D_row = np.negative(np.reciprocal(np.sqrt(D.diagonal())))
neg_rec_sqrt_D = np.zeros(D.shape)
for i in range(num_nodes):
	neg_rec_sqrt_D[i,i] = neg_rec_sqrt_D_row[i]
S = np.dot(np.dot(neg_rec_sqrt_D, A_tilde), neg_rec_sqrt_D)

##
## SGC Model
##
# Weight matrix (on which we learn) is a |V|xC matrix of weights
num_classes = max([targets[x] for x in targets.keys()])+1 # +1 for 0
W = np.random.randn(X.shape[1], num_classes)

# a k-deep SGC model (ie the new data)
k = 3
X_sgc = np.dot(np.linalg.matrix_power(S,k), X)


##
## Utility functions
##
def feedforward(a, W):
	a = sigmoid(np.dot(w,a))
	return a

def vectorized_result(j, num_classes):
	"""
	Return a num_classes dimensional unit vector with a 1.0 
	in the j'th position and zeroes elsewhere. One Hot Encoding
	"""
	e = np.zeros((num_classes,1))
	e[j] = 1.0
	return e

def sigmoid(z):
	"""
	Returns the sigmoid of z
	"""
	return (1.0/(1.0+np.exp(-z)))

def sigmoid_prime(z):
	"""
	Returns the value of derivative of sigmoid at z
	"""
	return sigmoid(z) * (1-sigmoid(z))
