import torch
import numpy
from torch_geometric.data import Data
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
import argparse
import matplotlib.pyplot as plt
import math
import copy
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
from torch_geometric.data import DataLoader
torch.manual_seed(10)
parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=12,
                    help='number of the node')
parser.add_argument('--num_weight', type=int, default=1,
                    help='number of the weight')
parser.add_argument('--K', type=int, default=3,
                    help='the size of the filter order')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the size of a batch')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    # evals_large, evecs_large = eigsh(adj, 1, which='LM')
    adj_ = copy.deepcopy(torch.Tensor(adj))
    adj = torch.Tensor(adj / evals_large)
    adj = adj.to(device)
    return adj,adj_

# model = GCN(nfeat=args.num_weight, nhid=64, K=args.K+1, nclasses=args.num_weight)
os.chdir('./try_K_%d_weight_%dMar2nd'%(args.K,args.num_weight))
model = torch.load("./model_best.pt")
model.to(device)

os.makedirs('../try_K_%d_weight_%dMar4th'%(args.K,args.num_weight), exist_ok=True)
os.chdir('../try_K_%d_weight_%dMar4th'%(args.K,args.num_weight))

edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
print(edge_index)
fo = open("output.txt","w+")
data_list = []
val_data_list = []
test_data_list = []
for k in range(args.data_size):
    x = torch.randn(args.num_node, args.num_weight)
    y = []
    for i in range(args.num_weight):
        y.append(sum(torch.transpose(x, 0, 1)[i]))
    y = torch.Tensor(y) * torch.ones((args.num_node, args.num_weight))/ args.num_node
    # y = (sum(x) * torch.ones(args.num_node))
    data = Data(x=x,y=y,edge_index=edge_index)
    data_list.append(data)
for j in range(2):
    for k in range(500):
        x = torch.randn(args.num_node, args.num_weight)
        y = []
        for i in range(args.num_weight):
            y.append(sum(torch.transpose(x, 0, 1)[i]))
        y = torch.Tensor(y) * torch.ones((args.num_node, args.num_weight))/args.num_node
        # y = (sum(x) * torch.ones(args.num_node)) / args.num_node
        data = Data(x=x, y=y, edge_index=edge_index)
        if j == 0:
            val_data_list.append(data)
        else:
            test_data_list.append(data)

train_data = data_list
val_data = val_data_list
test_data = test_data_list

print(len(train_data),',',len(val_data_list),',',len(test_data_list))
torch.save(data_list,'./data_list.pt')
torch.save(val_data_list,'./val_data_list.pt')
torch.save(test_data_list,'./test_data_list.pt')


train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# define the structure of GNN

# model = torch.load('./model.pt')
# train the GNN
k = 0
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
crit = torch.nn.MSELoss()

ll_train_list = []
ll_val_list = []
ll_val_lowest = 1e6
while(k<1200):
    k += 1
    loss_train_list = []
    loss_val_list = []
    for data in train_dataloader:
        data.to(device)
        optimizer.zero_grad()

        adj,adj_ = getNormalizedAdj(data)
        # divi = torch.transpose(torch.ones((args.num_weight, args.num_node)) * sum(adj_), 0, 1)
        output = model(data.x, adj_).squeeze()
        loss_train = crit(output, data.y.squeeze())

        loss_train_list.append(loss_train.item())
        loss_train.backward()
        optimizer.step()
    ll_train = sum(loss_train_list) / len(train_data)
    ll_train_list.append(ll_train)

    fo.write('Epoch: {:04d},'.format(k+1200))
    # fo.write('Epoch: {:04d},'.format(k))
    fo.write('loss_train: {:.11f},'.format(ll_train))

    for j in range(len(val_data)):
        data = val_data[j]
        data.to(device)
        adj,adj_ = getNormalizedAdj(data)
        # divi = torch.transpose(torch.ones((args.num_weight, args.num_node)) * sum(adj_), 0, 1)
        output = model(data.x, adj_).squeeze()
        loss_val = crit(output, data.y.squeeze())

        loss_val_list.append(loss_val.item())

    ll_val = sum(loss_val_list) / len(val_data)
    ll_val_list.append(ll_val)
    fo.write('loss_val: {:.11f}\n'.format(ll_val))
    print("epoch ",k,": ",ll_train,',',ll_val)
    if ll_val < ll_val_lowest:
        ll_val_lowest = ll_val
        torch.save(model, './model_best.pt')
        # print(loss_train.item()/args.batch_size)
# x_label = [i for i in range(len(ll_train_list))]
# plt.plot(x_label,ll_train_list,label='training loss')
# plt.plot(x_label,ll_val_list,label='validation loss')
# plt.legend()
# plt.yscale('log')
# plt.show()
# test the GNN
loss_test_list = []
for j in range(len(test_data)):
    data = test_data[j]
    adj,adj_ = getNormalizedAdj(data)
    output = model(data.x, adj_).squeeze()
    loss_test = crit(output, data.y.squeeze())
    loss_test_list.append(loss_test.item())

fo.write('loss_test: {:.11f}\n'.format(sum(loss_test_list) / len(test_data)))

print("Test loss: ",sum(loss_test_list) / len(test_data))

torch.save(model, './model.pt')
torch.save(ll_train_list,'./ll_train_list.pt')
torch.save(ll_val_list,'./ll_val_list.pt')
torch.save(loss_test_list,'./loss_test_list.pt')
