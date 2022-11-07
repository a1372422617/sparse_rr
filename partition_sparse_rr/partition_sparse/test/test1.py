import dgl
import torch
import numpy as np
import dgl.nn as dglnn

import torch.nn as nn
import torch.nn.functional as F
from dgl.partition import metis_partition_assignment
from partition_sparse.data.rr import sparse_rr
from partition_sparse import preprocess
import time
import argparse
# dataset = dgl.data.CoraFullDataset()
# #dataset = dgl.data.RedditDataset()
# g = dataset[0]
# node_part = metis_partition_assignment(g,2)
# part_num = torch.zeros(2)
# for i in range(g.num_nodes()):
#   part_num[node_part[i]] +=1
# part_offset = torch.cumsum(part_num,dim=0)
# part_offset = torch.cat([torch.zeros(1),part_offset],dim=0).to(torch.int32)
# node_part = torch.argsort(node_part)
# print(node_part)
# print(part_offset)

# time1 = time.time()
# g_new = g
# for i in range(1):
#   node_id = node_part[part_offset[i]:part_offset[i+1]]
#   g_sub = g.subgraph(node_id)
#   adj_sub = g_sub.adj(scipy_fmt='csr')
#   if i ==0:
#     print(g_sub.successors(0))
#     print(adj_sub)
#   adj_new,feats_new = sparse_rr.main(adj_sub,g_new.adj(scipy_fmt='csr'),g_new.adj(scipy_fmt='csr').T,node_id.numpy(),g_new.ndata['feat'].numpy(),1)
#   g_new = dgl.from_scipy(adj_new)
#   g_new.ndata['feat'] = torch.from_numpy(feats_new).to(torch.float32)
# time2 = time.time()
# print(time2-time1)
def parse_args():
    parser = argparse.ArgumentParser(description='preprocess_test')
    parser.add_argument('--dataset', type=str, required=True, help='name of dgl dataset')
    parser.add_argument('--k', type=int, required=True, help='number of graph patition if --preprocess')
    parser.add_argument('--round', type=int, required=True, help='total number of rounds to perform reduction if --preprocess')
    parser.add_argument('--num_layers', type=int,default = 3,help='total number of GNN layers')
    parser.add_argument('--fanout', type=str, default='10,15,20')
    parser.add_argument("--preprocess", action='store_true',default=False,help='Preprocessing or not')
    parser.add_argument("--reorder", action='store_true',default=False,help='Reorder or not')
    parser.add_argument("--cache", action='store_true',default=False,help='Cache or not')
    args = parser.parse_args()
    return args
class StochasticGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_features, hidden_features))
        for i in range(1, num_layers - 1):
            self.layers.append(dglnn.GraphConv(hidden_features, hidden_features))
        self.layers.append(dglnn.GraphConv(hidden_features, out_features))
    def forward(self,blocks,x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = F.relu(layer(block, h))
        return h
        
        
        
        
class StochasticSage(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_features, hidden_features,'mean'))
        for i in range(1, num_layers - 1):
            self.layers.append(dglnn.SAGEConv(hidden_features, hidden_features,'mean'))
        self.layers.append(dglnn.SAGEConv(hidden_features, out_features,'mean'))
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            h = F.relu(layer(block, (h, h_dst)))
        return h
if __name__ == '__main__':
    args = parse_args()
    g , redundant_num= preprocess.main(args.dataset,args.k,args.round,args.preprocess,args.reorder)
    
    device = 'cuda:0'
    print('训练开始..................')
    print('节点数:',g.num_nodes())
    
    index = torch.arange(g.num_nodes())
    train_nids = index[g.ndata['train_mask']]
    val_nids = index[g.ndata['val_mask']]
    test_nids = index[g.ndata['test_mask']]

    train_nids = train_nids.to(device)
    print('训练集大小:',train_nids.shape[0])

    n_features = g.ndata['feat'].shape[1]
    n_labels = int(g.ndata['label'].max().item() + 1)
    g.ndata['label']=g.ndata['label'].to(torch.int64)
    
    node_feat = g.ndata.pop('feat')
    print(node_feat)
    node_feat = dgl.contrib.UnifiedTensor(node_feat,device = device)
    
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fanout.split(',')])
    train_dataloader = dgl.dataloading.DataLoader(
    g,train_nids.to('cpu'),sampler,
    batch_size = 1024,
    shuffle =True,
    drop_last = False,
    num_workers = 16
    )
    val_dataloader = dgl.dataloading.DataLoader(
    g,val_nids.to('cpu'),sampler,
    batch_size = 1024,
    shuffle =True,
    drop_last = False,
    num_workers = 16
    )
    test_dataloader = dgl.dataloading.DataLoader(
    g,test_nids.to('cpu'),sampler,
    batch_size = 1024,
    shuffle =True,
    drop_last = False,
    num_workers = 16
    )
    
    # dataloader = dgl.dataloading.DataLoader(
    # g,train_nids,sampler,
    # device = torch.device(device),
    # batch_size = 4096,
    # shuffle =True,
    # drop_last = False,
    # num_workers = 0,
    # use_uva=True
    # )
    model = StochasticGCN(in_features=n_features, hidden_features=100, out_features=n_labels,num_layers = args.num_layers)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    start=time.time()
    for epoch in range(20):
        model.train()
        epochstartime = time.time()
        trans_time = 0
        agg_time = 0
        back_time = 0
        for input_nodes, output_nodes, blocks in train_dataloader:

            start_trans = time.time()
            blocks = [b.to(torch.device(device)) for b in blocks]

            input_features = node_feat[input_nodes].to(device)
            #input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']
            end_trans = time.time()
            trans_time = trans_time+(end_trans-start_trans)

            start_agg = time.time()
            output_predictions = model(blocks, input_features)
            end_agg = time.time()
            agg_time =agg_time+(end_agg-start_agg)

            start_back = time.time()
            loss = F.cross_entropy(output_predictions,output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            end_back = time.time()
            back_time +=(end_back - start_back)

        print("loss:",loss)
        epochendtime = time.time()


        print("数据传输时间:",trans_time)
        print("聚合时间:",agg_time)
        print("反向传播时间:",back_time)
        print("epoch时间:%.2f秒"%(epochendtime-epochstartime))
    end = time.time()
    print("训练时间:%.2f秒"%(end-start))

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in test_dataloader:
            print(output_nodes.shape)
            print(blocks[-1].dstdata['label'].shape)
            blocks = [b.to(torch.device(device)) for b in blocks]
            input_features = node_feat[input_nodes].to(device)
            labels.append(blocks[-1].dstdata['label'].to('cpu').numpy())
            predictions.append(model(blocks, input_features).argmax(1).to('cpu').numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        result = (predictions==labels).sum().item()/labels.shape[0]
        print(result)
    
    
  


