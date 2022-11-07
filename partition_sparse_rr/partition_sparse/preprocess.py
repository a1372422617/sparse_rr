#from functools import cache
from partition_sparse.data.rr import sparse_rr
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import networkx as nx
import scipy.sparse as sp
import numpy as np
import argparse
import os.path
from dgl.partition import metis_partition_assignment

def main(data,k,round,pre_flag,reorder_flag):
    if data == 'cite':
        dataset = dgl.data.CiteseerGraphDataset()
    elif data =='cora':
        dataset = dgl.data.CoraFullDataset()
    elif data=='ppi':
        dataset = dgl.data.PPIDataset()
    elif data =='coauthorP':
        dataset = dgl.data.CoauthorPhysicsDataset()
    elif data =='coauthorC':
        dataset = dgl.data.CoauthorCSDataset()
    elif data =='pubmed':
        dataset = dgl.data.PubmedGraphDataset()
    elif data =='amazon':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
    elif data =='reddit':
        dataset = dgl.data.RedditDataset()
    elif data =='wiki':
        dataset = dgl.data.WikiCSDataset()
    elif data =='flickr':
        dataset = dgl.data.FlickrDataset()
    elif data =='yelp':
        dataset = dgl.data.YelpDataset()
    else:
        raise ValueError('dataset name must be cite,cora...')
    
    print('Dataset loading completed')

    g = dataset[0]
    #split train,valid,test
    nids = np.arange(g.num_nodes())
    np.random.shuffle(nids)
    train_len = int(g.num_nodes()*0.6)
    val_len = int(g.num_nodes()*0.2)
    test_len = g.num_nodes() - train_len - val_len

    #not preprocess
    if not pre_flag:
        g = dgl.add_self_loop(g)
        # train mask
        train_mask = np.zeros(g.num_nodes(), dtype=np.int)
        train_mask[nids[0:train_len]] = 1
        g.ndata['train_mask'] = torch.from_numpy(train_mask).to(torch.bool)

        # val mask
        val_mask = np.zeros(g.num_nodes(), dtype=np.int)
        val_mask[nids[train_len:train_len + val_len]] = 1
        g.ndata['val_mask'] = torch.from_numpy(val_mask).to(torch.bool)

        # test mask
        test_mask = np.zeros(g.num_nodes(), dtype=np.int)
        test_mask[nids[train_len + val_len:g.num_nodes()]] = 1
        g.ndata['test_mask'] = torch.from_numpy(test_mask).to(torch.bool)
        if reorder_flag:
            g = dgl.reorder_graph(g, node_permute_algo='rcmk')
            return g,0
        else:
            return g,0
    adj_gs = g.adj(scipy_fmt='csr')
    feats = g.ndata['feat'].numpy()
    labels = g.ndata['label']

    out_dir = '../data/dataset'
    adj_file_name = data+'_adj_'+str(round)+'.npz'
    feat_file_name = data+'_feat_'+str(round)+'.npy'
    adj_file = os.path.join(out_dir,adj_file_name)
    feat_file = os.path.join(out_dir,feat_file_name)
    
    
    g_new = g
    if os.path.isfile(adj_file) and os.path.isfile(feat_file):
        adj_new = sp.load_npz(os.path.join(out_dir,adj_file_name))
        feats_new = np.load(os.path.join(out_dir,feat_file_name))
        g_new = dgl.from_scipy(adj_new)
        g_new.ndata['feat'] = torch.from_numpy(feats_new).to(torch.float32)
    else:
        time1 = time.time()
        start_round = 0
        for base in range(1,round):
            adj_base = os.path.join(out_dir,data+'_adj_'+str(round-base)+'.npz')
            feat_base = os.path.join(out_dir,data+'_feat_'+str(round-base)+'.npy')
            if os.path.isfile(adj_base) and os.path.isfile(feat_base):
                adj_new = sp.load_npz(adj_base)
                feats_new = np.load(feat_base)
                g_new = dgl.from_scipy(adj_new)
                g_new.ndata['feat'] = torch.from_numpy(feats_new).to(torch.float32)
                start_round = round - base
                break
        print(start_round,":lalalalaa")
        
        for r in range(start_round,round):
            print("第",r+1,"轮冗余消除开始")
            time_round1 = time.time()
            node_part = metis_partition_assignment(g_new,k)
            part_num = torch.zeros(k)
            for i in range(g.num_nodes()):
                part_num[node_part[i]] +=1
            part_offset = torch.cumsum(part_num,dim=0)
            part_offset = torch.cat([torch.zeros(1),part_offset],dim=0).to(torch.int32)
            node_part = torch.argsort(node_part)
            for i in range(k):
                node_id = node_part[part_offset[i]:part_offset[i+1]]
                g_sub = g_new.subgraph(node_id)
                adj_sub = g_sub.adj(scipy_fmt='csr')
                adj_new,feats_new = sparse_rr.main(adj_sub,g_new.adj(scipy_fmt='csr'),g_new.adj(scipy_fmt='csr').T,node_id.numpy(),g_new.ndata['feat'].numpy(),1)
                g_new = dgl.from_scipy(adj_new)
                g_new.ndata['feat'] = torch.from_numpy(feats_new).to(torch.float32)
                print(g_new)
            time_round2= time.time()
            print("第",r+1,"轮冗余消除时间:",time_round2-time_round1)
            
            #save data
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            sp.save_npz(os.path.join(out_dir,data+'_adj_'+str(r+1)+'.npz'),g_new.adj(scipy_fmt='csr'))
            np.save(os.path.join(out_dir,data+'_feat_'+str(r+1)+'.npy'),g_new.ndata['feat'].numpy()) 
              
        time2 = time.time()
        print("冗余消除时间:",time2-time1)
        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)
        #sp.save_npz(os.path.join(out_dir,adj_file_name),g_new.adj(scipy_fmt='csr'))
        #np.save(os.path.join(out_dir,feat_file_name),g_new.ndata['feat'].numpy())               

    n_labels = int(labels.max().item() + 1)
    labels_new = torch.ones(g_new.num_nodes())*(n_labels)
    labels_new[:g.num_nodes()] = labels
    g_new.ndata['label'] = labels_new.to(torch.int64)
    


    # newnode_mask = torch.zeros(g_new.num_nodes()-g.num_nodes()).to(torch.bool)
    # g_new.ndata['train_mask'] = torch.cat([g.ndata['train_mask'],newnode_mask],dim = 0)
    # g_new.ndata['val_mask'] = torch.cat([g.ndata['val_mask'],newnode_mask],dim = 0)
    # g_new.ndata['test_mask'] = torch.cat([g.ndata['test_mask'],newnode_mask],dim = 0)


    # train mask
    train_mask = np.zeros(g_new.num_nodes(), dtype=np.int)
    train_mask[nids[0:train_len]] = 1
    g_new.ndata['train_mask'] = torch.from_numpy(train_mask).to(torch.bool)

    # val mask
    val_mask = np.zeros(g_new.num_nodes(), dtype=np.int)
    val_mask[nids[train_len:train_len + val_len]] = 1
    g_new.ndata['val_mask'] = torch.from_numpy(val_mask).to(torch.bool)

    # test mask
    test_mask = np.zeros(g_new.num_nodes(), dtype=np.int)
    test_mask[nids[train_len + val_len:g.num_nodes()]] = 1
    g_new.ndata['test_mask'] = torch.from_numpy(test_mask).to(torch.bool)

    if reorder_flag:
        g_new = dgl.reorder_graph(g_new, node_permute_algo='rcmk')
    g_new = dgl.add_self_loop(g_new)
    #cache the redundant node feature
    return g_new,g_new.num_nodes()-g.num_nodes()



