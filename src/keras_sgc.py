"""
Simple SGC Keras implementation
Author: Paul Scherer (paulmorio)
Date: May 2019
"""

import random
import numpy as np
import networkx as nx
import json
from collections import defaultdict

from keras.layers import Input, Dense
from keras.models import Model

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


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
## SGC Model
##
# a k-deep SGC model (ie the new data)
k = 3
X_sgc = np.dot(np.linalg.matrix_power(S,k), X)


###############################
## Hyperparameters for Keras ##
###############################
batch_size = 100
output_dim = 7
epochs = 500

# Redefining training and test data
training_data = []
for row in range(1708):
	x = np.ravel(X_sgc[row].T)
	y = np.ravel(vectorized_result(targets[row], 7))
	training_data.append((x,y))

test_data = []
for row in range(1708,2708):
	x = np.ravel(X_sgc[row].T)
	y = targets[row]
	test_data.append((x,y))


keras_compliant_train_x = []
keras_compliant_train_y = []
for x, y in training_data:
	keras_compliant_train_x.append(x.flatten())
	keras_compliant_train_y.append(np.argmax(y.flatten()))
x_train = np.array(keras_compliant_train_x)
y_train = np.array(keras_compliant_train_y)

keras_compliant_test_x = []
keras_compliant_test_y = []
for x, y in test_data:
	keras_compliant_test_x.append(x.flatten())
	keras_compliant_test_y.append(y)
x_test = np.array(keras_compliant_test_x)
y_test = np.array(keras_compliant_test_y)


# Change labels into binary categorical vectors
y_train = keras.utils.to_categorical(y_train, output_dim)
y_test = keras.utils.to_categorical(y_test, output_dim)


########################
## Network Definition ##
########################
net = Sequential()
net.add(Dense(64, activation="sigmoid", input_shape=(1433,)))
net.add(Dense(7, activation="softmax"))
net.summary()

net.compile(loss="categorical_crossentropy",
			optimizer=SGD(),
			metrics=["accuracy"])

#######################
## Train the Network ##
#######################
history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data = (x_test, y_test))
scores = net.evaluate(x_test, y_test, verbose=0)
print ("Test Loss: %s, Test Accuracy: %s"%(scores[0], scores[1]))