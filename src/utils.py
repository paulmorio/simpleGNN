"""
Utilities file for loading and preprocessing of data used in SimpleGNN classes.

Author: Paul Scherer
Date: May 2019
"""

import numpy as np
import networkx as nx
import json
from collections import defaultdict

def get_graph_and_matrices_edge_list(edge_list_csv, features_json, targets_csv):
	"""
	Given the edgelist features and node/id targets this function returns:
	- graphnx: NX Graph with nodes as ints
	- features_dict: Dictionary matching int node-id with sparse list of locations for binary encoding
	- targets: Dictionary matchine int node-id with int target id
	- A: numpy adjacency matrix
	- A_tilde: adjacency matrix with self loops
	- D: degree matrix
	- D_tilde: degree matrix of A_tilde
	- X: node features matrix
	- S: normalised matrix (applies feature propagation)
	"""
	graphnx = nx.read_edgelist(path=edge_list_csv, delimiter=",", nodetype=int) # we will keep the node type string as its used in the other data files

	# Sparse node features dictionary
	with open(features_json) as json_file:
		features_dict = defaultdict(list)
		json_str = json_file.read()
		features_dict_str = json.loads(json_str)
		for key in features_dict_str.keys():
			key_int = int(key)
			features_dict[key_int] = features_dict_str[key]
		del(features_dict_str)

	# Node index to target dictionary
	with open(targets_csv) as fh:
		targets = defaultdict()
		lines = fh.readlines()
		for line in lines:
			id_target_pair = line.strip().split(",")
			idx, target = id_target_pair
			targets[int(idx)] = int(target)

	# Adjacency matrix
	num_nodes = graphnx.number_of_nodes()
	identity = np.identity(num_nodes)
	A = nx.to_numpy_matrix(G=graphnx, nodelist=range(num_nodes))
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

	return((graphnx, features_dict, targets, A, A_tilde, D, D_tilde, X, S))