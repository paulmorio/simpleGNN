"""
A simple linear implementation of the Graph Convolutional Neural Network applied in the 
context of node classification. It's aim is to show a fast implementation using only
python and numpy for pedagogic purposes.

Author: Paul Scherer (paulmorio)
Date: May 2019
MIT License
"""

import random
import numpy as np
import networkx as nx
import json


##
## Data Preprocessing
##
cora= nx.read_edgelist(path="/home/morio/workspace/simpleGNN/data/cora_edges.csv", delimiter=",", nodetype=int)
json_data = json.loads("/home/morio/workspace/simpleGNN/data/cora_features.json")
# cora_target = 

