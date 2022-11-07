import dgl
import torch
import numpy as np
import dgl.nn as dglnn
if __name__ == '__main__':
    a = np.array([10,20,40,70,110])
    print(np.ediff1d(a))
    
    dataset = dgl.data.RedditDataset()
    g = dataset[0]
    print(g)