import torch
import math
import copy
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_geometric.data import Data
#load GNN
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclasses):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(in_channels=nfeat,out_channels=nhid)
        self.gc2 = GCNConv(in_channels=nhid,out_channels=nhid)
        self.lin = nn.Linear(nhid, nclasses)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x,edge_index))
        x = F.relu(self.gc2(x,edge_index))
        x = self.lin(x)
        return x
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

torch.manual_seed(10)
# problem:
L = 12
node_feature = 3

edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
print(edge_index)
x = 300*torch.ones((L,node_feature))
data = Data(x=x, edge_index=edge_index)

# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,node_feature))
U = torch.randn((L,node_feature,node_feature))
# x = torch.ones((L,node_feature))
# print(U)
v = torch.empty((L,node_feature))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 40

W = generate_Metropolis_W(data)
recon_x = torch.zeros((L,node_feature),requires_grad=True)
loss_list = []
for k in range(Iteration):
    alpha = 6e-2
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        f.backward()

    loss = norm_f(recon_x - x)/norm_f(torch.zeros((L,node_feature))-x)
    loss_list.append(loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        with torch.no_grad():
            # print(recon_x.grad[i])
            recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * recon_x.grad[i]
    with torch.no_grad():
        # use GNN to replace the weight aggregation here
       index_2 = sum(recon_x)/L
    with torch.no_grad():
        index_1 = model(torch.cat((recon_x, torch.ones(L, 1)), dim=1), data.edge_index).squeeze()
    print("loss between GNN and normal one", torch.norm(index_1-index_2[0]))
    for i in range(L):
        with torch.no_grad():
            recon_x[i] = index_1[i].data
print(len(loss_list))
x_label = [i for i in range(len(loss_list))]
plt.plot(x_label,loss_list)
plt.yscale('log')
plt.show()
print(loss_list[len(loss_list)-1])
