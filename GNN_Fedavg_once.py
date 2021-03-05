import torch
import math
import copy
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_geometric.data import Data
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
#load GNN
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclasses):
#         super(GCN, self).__init__()
#
#         self.gc1 = GCNConv(in_channels=nfeat,out_channels=nhid)
#         self.gc2 = GCNConv(in_channels=nhid,out_channels=nhid)
#         self.lin = nn.Linear(nhid, nclasses)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.gc1(x,edge_index))
#         x = F.relu(self.gc2(x,edge_index))
#         x = self.lin(x)
#         return x
# model = torch.load('./model.pt')
device = 'cpu'
class GraphConvolution(Module):
    def __init__(self, K, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        out = []
        adj_ = adj
        for i in range(self.K):
            if i == 0:
                support = torch.mm(input, self.weight[i])
                out.append(support)
            else:
                tmp = torch.mm(adj_, input)
                support = torch.mm(tmp, self.weight[i])
                out.append(support)
                adj_ = torch.mm(adj_, adj)
        output = sum(out)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, K, nclasses):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(K=K, in_features=nfeat, out_features=nhid)
        self.gc2 = GraphConvolution(K=K, in_features=nhid, out_features=nhid)
        self.lin = nn.Linear(nhid, nclasses)

    def forward(self, x, adj):
        x = self.gc1(x,adj)
        x = F.relu(x)
        x = self.gc2(x,adj)
        x = F.relu(x)
        x = self.lin(x)
        return x
#get the normalized adjacency
def getNormalizedAdj(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()
    evals_large, evecs_large = eigsh(adj, 1, which='LM')
    adj = torch.Tensor(adj / evals_large)
    adj = adj.to(device)
    return adj

model = torch.load('./model.pt')

def generate_Metropolis_W(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = torch.Tensor(sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray())
    # adj = torch.Tensor(sp.coo_matrix((raw_data, (row, col)), shape=(L, L)).toarray())
    degree = torch.sum(adj, 1)

    W = torch.zeros((data.x.shape[0], data.x.shape[0]))
    for i in range(data.edge_index.shape[1]):
        W[data.edge_index[0][i]][data.edge_index[1][i]] = 1 / (
                    2 * max(degree[data.edge_index[0][i]], degree[data.edge_index[1][i]]))
    for i in range(data.x.shape[0]):
        sum_W = 0
        for j in range(data.x.shape[0]):
            if adj[i][j] != 0:
                sum_W += W[i][j]
        W[i][i] = 1 - sum_W

    return W

def norm_f(x):
    m,n = x.shape
    x = x.reshape(m*n)
    total_sum = 0
    for i in range(m*n):
        total_sum += x[i]*x[i]
    total_sum = torch.sqrt(total_sum)
    return total_sum

torch.manual_seed(100)
# problem:
L = 12
mi = 1
node_feature = 3

edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
print(edge_index)
x = 300*torch.ones((L,node_feature))
data = Data(x=x, edge_index=edge_index)

# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,mi))
U = torch.randn((L,mi,node_feature))
# x = torch.ones((L,node_feature))
# print(U)
v = torch.empty((L,mi))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 400

W = generate_Metropolis_W(data)
recon_x = torch.zeros((L,node_feature),requires_grad=True)
loss_list = []
gap = []
for k in range(Iteration):
    alpha = 6e-2
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(formula, formula)
        f.backward()

    loss = norm_f(recon_x - x)/norm_f(torch.zeros((L,node_feature))-x)
    loss_list.append(loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        with torch.no_grad():
            # print(recon_x.grad[i])
            recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * recon_x.grad[i]
    # for i in range(L):
    #     with torch.no_grad():
    #         recon_x[i] = sum(recon_x)/L

    with torch.no_grad():
        # print((torch.max(recon_x,0)))
        index_1 = sum(recon_x)/L * torch.ones((L,node_feature))
        recon_x = index_1
        # recon_x = index_1 + 1e-5*torch.rand((12,3))
        recon_x = recon_x.requires_grad_()


    with torch.no_grad():
        # index_1 = model(recon_x, data.edge_index).squeeze()
        adj = getNormalizedAdj(data)
        index_2 = torch.zeros((node_feature,L))
        for i in range(node_feature):
            index_2[i]= model(recon_x.T[i].unsqueeze(1), adj).squeeze()
        recon_x = index_2.T
        recon_x = recon_x.requires_grad_()
    # gap.append(torch.norm(index_1-recon_x).item())
    # print("loss between GNN and normal one", torch.norm(index_1-index_2[0]))
    # for i in range(L):
    #     with torch.no_grad():
    #         recon_x[i] = index_1[i].data
print(len(loss_list))
x_label = [i for i in range(len(loss_list))]
plt.plot(x_label,loss_list)
# plt.plot(x_label,gap)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.yscale('log')
plt.show()
print(loss_list[len(loss_list)-1])
