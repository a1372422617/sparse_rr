import scipy.sparse as sp
import numpy as np
import torch
import csrgemm
import time

def construct_ga(adj_gs):

    num_v_sub = adj_gs.shape[0]

    rowptr_A = adj_gs.T.indptr.astype('int64').tolist()
    indices_A = adj_gs.T.indices.astype('int64').tolist()
    data_A = adj_gs.T.data.astype('float').tolist()

    rowptr_B = adj_gs.indptr.astype('int64').tolist()
    indices_B = adj_gs.indices.astype('int64').tolist()
    data_B = adj_gs.data.astype('float').tolist()
    
    adj_gs_T = adj_gs.T
    
    redundancy_matrix = adj_gs_T.dot(adj_gs)
    
    redundancy_coo = redundancy_matrix.tocoo()

    #redundancy_matrix = csrgemm.csrgemm(rowptr_A,indices_A,data_A,rowptr_B,indices_B,data_B)
    # rows = np.array(redundancy_matrix.rows)
    # indices = np.array(redundancy_matrix.indices)
    # value = np.array(redundancy_matrix.value,dtype = np.int64)
    # redundancy_csr = sp.csr_matrix((value,indices,rows))
    # redundancy_coo = redundancy_csr.tocoo()
    # print(redundancy_coo.row.shape)
    # print(redundancy_coo.col.shape)
    # print(redundancy_coo.data.shape)
    weight_edges = np.stack((redundancy_coo.row,redundancy_coo.col,redundancy_coo.data)).T
    print(weight_edges.shape)
    return weight_edges,num_v_sub,redundancy_matrix

def find_node_max_redundancy(indices,value,vis,num,node):
    max = 0
    maxid = -1
    for i in range(num):
        if vis[indices[i]] == 1:
            #print("node",indices[i],"has been visit")
            continue
        if indices[i] == node :
            continue
        if value[i] > max and value[i]>2:
            max = value[i]
            maxid = indices[i]     
    return (node,maxid)

def find_max_redundancy(redundancy_matrix,node_id,num_v_sub):
    
    indptr = redundancy_matrix.indptr
    indices = redundancy_matrix.indices
    redundancy_num = redundancy_matrix.data
    vis = np.zeros(num_v_sub)
    M = []
    for i in range(num_v_sub):
        if vis[i]==1 :
            continue
        if indptr[i+1]-indptr[i] == 0 :
            continue
        indices_i = indices[indptr[i]:indptr[i+1]]
        value_i = redundancy_num[indptr[i]:indptr[i+1]]
        res = find_node_max_redundancy(indices_i,value_i,vis,indptr[i+1]-indptr[i],i)
        if res[1] != -1:
            vis[res[0]] = 1
            vis[res[1]] = 1
            M.append((node_id[res[0]],node_id[res[1]]))
            #print(res[0],res[1])
        if len(M) == int(num_v_sub/2):
            break
    return M
        
    
    
def obtain_precompute_edges(weight_edges,node_id,num_v_sub):
    """
    operate on ga
    """

    time1 = time.time()
    M = []
    data = weight_edges.T[2]
    H = weight_edges[data>2]
    time2 = time.time()
    print("阶段1:",time2-time1)
    
    H_sorted = sorted(H,key = lambda x:(x[2]),reverse=True)
    #print(H_sorted)
    time3 = time.time()
    print("阶段2:",time3-time2)
    
    S = np.ones(num_v_sub)
    _W = 0
    for row,col,data in H_sorted:
        if not(S[row] and S[col]):
            continue
        if row == col:
            continue
        _W += data -1
        S[row] = 0
        S[col] = 0
        M.append((node_id[row],node_id[col]))
        if len(M) == int(num_v_sub/2):
            break
    time4 = time.time()
    print("阶段3:",time4-time3)
    #print(M)
    return M,_W

def obtain_compact_mat(adj_gs,adj_t,M,feat):
    """
    obtain updated gs from M
    """
    #ret_feat = np.zeros(feat.size+len(M))
    
    time_t1 = time.time()

    ret_feat = np.zeros((feat.shape[0]+len(M),feat.shape[1]))
    
    ret_feat[:feat.shape[0]] = feat
    

    idx = 0
    deg = np.ediff1d(adj_gs.indptr)
    num_v = deg.size
    #transpose adj first
    e_list = [[] for v in range(adj_gs.shape[0])]
    for v in range(adj_gs.shape[0]):
        n_list = adj_gs.indices[adj_gs.indptr[v]:adj_gs.indptr[v+1]]
        for n in n_list:
            e_list[n].append(v)
    e_list_full = []
    gs_t_indptr = np.zeros(adj_gs.shape[0]+1).astype(np.int32)       # indptr for adj_gs.T
    for i,el in enumerate(e_list):
        e_list_full.extend(sorted(el))
        gs_t_indptr[i+1] = gs_t_indptr[i] + len(el)
    gs_t_indices = np.array(e_list_full).astype(np.int32)            # indices for adj_gs.T
    
    
    
    # prepare I_edges here, after identifying the large-weight edges
    # gs_t_indptr= adj_t.indptr
    # gs_t_indices= adj_t.indices
    # assert(gs_t_indices.shape[0] == gs_t_indices1.shape[0])
    time_t2 = time.time()
    print("transpose time:",time_t2-time_t1)
    I_edges = dict()
    for (aggr1,aggr2) in M:
        # intersection of aggr1's neighbor and aggr2's neighbor
        _neigh1 = gs_t_indices[gs_t_indptr[aggr1]:gs_t_indptr[aggr1+1]]
        _neigh2 = gs_t_indices[gs_t_indptr[aggr2]:gs_t_indptr[aggr2+1]]
        I_edges[(aggr1,aggr2)] = np.intersect1d(_neigh1,_neigh2,assume_unique=True)
    for (aggr1,aggr2) in M:
        
        v_root = I_edges[(aggr1,aggr2)]
        # print("root:",v_root)
        # print("aggr:",(aggr1,aggr2))

        ret_feat[num_v+idx] = ret_feat[aggr1]+ret_feat[aggr2]
        #ret_feat[num_v+idx] = (ret_feat[aggr1]+ret_feat[aggr2])/2

        for v in v_root:
            neigh = adj_gs.indices[adj_gs.indptr[v]:adj_gs.indptr[v+1]]
            # if v>20000:
            #     print(neigh)
            i1 = np.where(neigh==aggr1)[0][0]
            i2 = np.where(neigh==aggr2)[0][0]       # searchsorted not applicable here since we insert -1
            adj_gs.indices[adj_gs.indptr[v]+i1] = num_v+idx
            adj_gs.indices[adj_gs.indptr[v]+i2] = -1
            deg[v] -= 1
        idx += 1
    _indptr_new = np.cumsum(deg)
    indptr_new = np.zeros(num_v+idx+1)
    indptr_new[1:num_v+1] = _indptr_new
    indptr_new[num_v+1:] = _indptr_new[-1]
    indices_new = adj_gs.indices[np.where(adj_gs.indices>-1)]
    assert indices_new.size == indptr_new[-1]
    data_new = np.ones(indices_new.size)
    ret_adj = sp.csr_matrix((data_new,indices_new,indptr_new),shape=(num_v+len(M),num_v+len(M)))


    return ret_adj, ret_feat
    

    

f_tot_ops = lambda adj: adj.size-np.where(np.ediff1d(adj.indptr)>0)[0].size
f_tot_read = lambda adj: adj.size#-np.where(np.ediff1d(adj.indptr)==1)[0].size
max_deg = lambda adj: np.ediff1d(adj.indptr).max()
mean_deg = lambda adj: np.ediff1d(adj.indptr).mean()
sigma_deg2 = lambda adj: (np.ediff1d(adj.indptr)**2).sum()/adj.shape[0]

def main(adj_sub,adj,adj_t,node_id,feat,num_round):
    adj_gs = adj_sub
    num_v_orig = adj_gs.shape[0]
    tot_ops_orig = f_tot_ops(adj_gs)
    tot_read_orig = f_tot_read(adj_gs)
    feat_num = feat.shape[1]
    cnt_precompute = 0
    cnt_preread = 0
    for r in range(num_round):
        ops_prev = f_tot_ops(adj_gs)
        time1 = time.time()
        weight_edges,num_v_sub,mat= construct_ga(adj_gs)
        time2 = time.time()
        print("first time:",time2-time1)
        #M,_W = obtain_precompute_edges(weight_edges,node_id,num_v_sub)
        M = find_max_redundancy(mat,node_id,num_v_sub)
        print("长度:",len(M))
        print("节点数/2:", num_v_sub/2)
        time3 = time.time()
        print("second time:",time3-time2)
        cnt_precompute += len(M)
        cnt_preread += 2*len(M)
        adj_gs,feat = obtain_compact_mat(adj,adj_t,M,feat)
        time4 = time.time()
        print("third time:",time4-time3)
        # ops_new = f_tot_ops(adj_gs) + cnt_precompute
        # read_new = f_tot_read(adj_gs) + cnt_preread
        # print("previous ops: ", ops_prev)
        # print("new ops: ", ops_new)
        # print("match size: ",len(M))
        # print("reduction comp compared to original: {:.2f} (precompute {:.3f} of original total ops, temp buffer {:.3f}% of |V|)"\
        #     .format(tot_ops_orig/ops_new,cnt_precompute/tot_ops_orig,cnt_precompute/num_v_orig*100))
        # print("reduction comm compared to original: {:.2f}".format(tot_read_orig/read_new))
    print("RESULT CORRECT!")
    return  adj_gs,feat
