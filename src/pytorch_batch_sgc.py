"""
Simple SGC PyTorch implementation with mini-batch training
with the cora dataset as an example
Author: Paul Scherer (paulmorio)
Date: May 2019
"""

import random
import numpy as np
import networkx as nx
import json
from collections import defaultdict

# PyTorch tensors and neural network packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch optimizers
import torch.optim as optim

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

print("## Preprocessing Complete ##")

##
## Utility functions
##
def vectorized_result(j, num_classes):
	"""
	Return a num_classes dimensional unit vector with a 1.0 
	in the j'th position and zeroes elsewhere. One Hot Encoding
	"""
	e = np.zeros((num_classes,1))
	e[j] = 1.0
	return e

##
## SGC Model
##
# a k-deep SGC model (ie the new data)
k = 3
X_sgc = np.dot(np.linalg.matrix_power(S,k), X)

# Redefining training and test data
training_data = []
for row in range(1708):
	x = X_sgc[row].T
	y = np.ravel(vectorized_result(targets[row], 7))
	training_data.append((x,y))

test_data = []
for row in range(1708,2708):
	x = X_sgc[row].T
	y = targets[row]
	test_data.append((x,y))

# Have to change our data set into Pytorch tensors to work.
torch_training_data = []
torch_test_data = []
for i in range(len(training_data)):
	x, y = training_data[i]
	torch_training_data.append((torch.from_numpy(x).float(), torch.from_numpy(y).float()))

for i in range(len(test_data)):
	x, y = test_data[i]
	torch_test_data.append((torch.from_numpy(x).float(), torch.as_tensor(y)))



##
## Focus point of this module, examine the sizes in the interactive python interpreter
def create_mini_batches(training_data, batch_size):
	"""
	Small helper function to create mini-batched training data set
	for mini-batch stochastic gradient descent
	"""
	batches = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size)]
	torch_compliant_batches = [] # torch likes batches to be in the shape tensor[num, (dimsIN)]
	for mini_batch in batches:
		mini_batch_obs = []
		mini_batch_labels = []
		for obs in mini_batch:
			x,y = obs
			mini_batch_obs.append(x.flatten())
			mini_batch_labels.append(y.flatten())
			# mini_batch_obs.append(x)
			# mini_batch_labels.append(y)
		stacked_obs = torch.stack(mini_batch_obs)
		stacked_labels = torch.stack(mini_batch_labels)
		torch_compliant_batches.append( (stacked_obs, stacked_labels) )

	return torch_compliant_batches

# Architecture Definition
class SimpleSGC(nn.Module):
	def __init__(self):
		super(SimpleSGC, self).__init__()
		self.fc1 = nn.Linear(1433, 64)
		self.fc2 = nn.Linear(64, 7)

	def forward(self,x):
		x = F.sigmoid(self.fc1(x))
		x = self.fc2(x)
		return x

# Instantiate the network and define the loss and optimizer
net = SimpleSGC()
net = net.float()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
batch_size = 100

##################################
###### Training the Network ######
##################################
for epoch in range(500):
	running_loss = 0.0
	random.shuffle(torch_training_data)
	batched_training_data = create_mini_batches(torch_training_data, batch_size)
	for x_batch, y_batch in batched_training_data:
		# forward pass
		outputs = net(x_batch)

		# backward pass
		loss = criterion(outputs, y_batch)
		optimizer.zero_grad()
		loss.backward()

		# Update weights
		optimizer.step()

		running_loss += loss.item()

	if epoch % 2 == 0:
		print('[%d] Loss: %.3f' % (epoch, running_loss))

print ("\n #### Finished Training ### \n")

########################
###### Evaluation ######
########################
correct = 0
total = 0
with torch.no_grad():
	for x,y in torch_test_data:
		x = (x.flatten().unsqueeze(0))
		y_pred = net(x)
		max_value, predicted_index = torch.max(y_pred.data, 1)

		total+=1
		if predicted_index == y:
			correct += 1
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
