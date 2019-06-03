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
## Data Preprocessing
##
cora = nx.read_edgelist(path="/home/morio/workspace/simpleGNN/data/cora_edges.csv", delimiter=",") # we will keep the node type string as its used in the other data files
with open('/home/morio/workspace/simpleGNN/data/cora_features.json') as json_file:
	json_str = json_file.read()
	cora_feat_dict = json.loads(json_str)
with open('/home/morio/workspace/simpleGNN/data/cora_target.csv') as fh:
	targets = defaultdict()
	lines = fh.readlines()
	for line in lines:
		id_target_pair = line.strip().split(",")
		idx, target = id_target_pair
		targets[idx] = target

# cora_target = 

