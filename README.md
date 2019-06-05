# simpleGNN

This repo consists of python implementations of the general GNN, GCN, SGC models for expositional purposes. If adapts the principles of the original Graph Neural Network specification written by Scarselli et. al. *The Graph Neural Network Model*, as well as Kipf and Welling's *Graph Convolutional Networks*. 

## Explanation and Overview of Codebase 

I personally found that the original code implementations of GNN and more importantly the popular GCN may be difficult to read for those not well versed with the model and Python/Tensorflow when it is split into different modules for the model, layers, etc. For educational purposes (and my own benefit to understand it better) I have written minimal implementations starting from a single script implementation coupling data processing, model, and evaluation (to show the classification process end-to-end) to a more modular class based GNN model which can be used with different datasets. The implementations were created purely in Python and Numpy, moving on to implementations with PyTorch and Keras (running on Tensorflow). 

As a result, there is an ordering of the code base from simple single module, simple model to the more complicated full models written OOP style across different modules for less coupling. Interestingly the simplest linear model of a graph convolutional neural network published by Wu et. al. (SGC) was published 3 years after the GCN. For educational purposes we focus on these first building our intuition towards the GCN. We will actually find that the SGC is incredibly good and on par with the GCN in many tasks but badly on others prompting a much needed review into principles of geometric learning and the problem at hand (this is an open research topic!).

- simpleSGC_script.py
- simpleGCN_script.py
- pytorch_sgc.py
- pytorch_gcn.py
- keras_sgc.py
- keras_gcn.py

- simpleGNN_pytorch.py: Contains class definitions for the above. 


## Theory
TODO
### GCN

### Linear GCN
