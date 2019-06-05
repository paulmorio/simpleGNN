"""
SimpleGNN classes for instantiating pytorch implementations of SGC and GCN

Author: Paul Scherer
Date: May 2019
"""

import numpy as np
from utils import get_graph_and_matrices_edge_list

# PyTorch tensors and neural network packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch optimizers
import torch.optim as optim


class SimpleGNN(object):
	"""docstring for simpleGNN"""
	def __init__(self, edge_list_csv, features_json, targets_csv):
		self.edge_list_csv = edge_list_csv
		self.features_json = features_json
		self.targets_csv = targets_csv

		self.graph_nx, self.features_dict, self.targets, self.A, self.A_tilde, self.D, self.D_tilde, self.X, self.S = get_graph_and_matrices_edge_list(edge_list_csv, features_json, targets_csv)


class SimpleSGC(SimpleGNN):
	"""SGC Class with k layers"""
	def __init__(self, edge_list_csv, features_json, targets_csv, k):
		super(SimpleSGC, self).__init__(edge_list_csv, features_json, targets_csv)
		self.k = k

		self.X_sgc = np.dot(np.linalg.matrix_power(S,k), X)


# Test Code
if __name__ == '__main__':
	gnn = SimpleGNN("../data/cora_edges.csv", 
		'../data/cora_features.json', 
		'../data/cora_target.csv')

	sgc = SimpleSGC("../data/cora_edges.csv", 
		'../data/cora_features.json', 
		'../data/cora_target.csv', k=3)



