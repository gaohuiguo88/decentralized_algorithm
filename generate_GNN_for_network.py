# import torch
# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module
# import math
# import scipy.sparse as sp
# from scipy.sparse.linalg import eigs, eigsh
# import torch_geometric
# import time
# import os,sys
# import argparse
# import numpy as np
# import warnings
# # device = 'cpu'
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--data_size', type=int, default=2500,
#                     help='Size of the training data')
# parser.add_argument('--num_node', type=int, default=100,
#                     help='number of the node')
# parser.add_argument('--K', type=int, default=25,
#                     help='the size of the filter order')
# parser.add_argument('--batch_size', type=int, default=1,
#                     help='the size of a batch')
# parser.add_argument('--distribution', type=str, default='normal',
#                     help='the distribution of the initial value of nodes')
# parser.add_argument('--obj', type=str, default='average',
#                     help='the object of the consensus problem')
#
# args = parser.parse_args()
#
#
# os.makedirs('./try_%s_Feb19th'%args.obj, exist_ok=True)
# os.chdir('./try_%s_Feb19th'%args.obj)
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# #define the layer
# class GraphConvolution(Module):
#     def __init__(self, K, in_features, out_features, bias=False):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.K = K
#         self.weight = Parameter(torch.FloatTensor(K, in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         out = []
#         adj_ = adj
#         for i in range(self.K):
#             if i == 0:
#                 support = torch.mm(input, self.weight[i])
#                 out.append(support)
#             else:
#                 tmp = torch.mm(adj_, input)
#                 support = torch.mm(tmp, self.weight[i])
#                 out.append(support)
#                 adj_ = torch.mm(adj_, adj)
#         output = sum(out)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
#
# #define the model
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, K, nclasses):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(K=K, in_features=nfeat, out_features=nhid)
#         self.gc2 = GraphConvolution(K=K, in_features=nhid, out_features=nhid)
#         self.lin = nn.Linear(nhid, nclasses)
#
#     def forward(self, x, adj):
#         x = self.gc1(x,adj)
#         x = F.relu(x)
#         x = self.gc2(x,adj)
#         x = F.relu(x)
#         x = self.lin(x)
#         return x
# #get the normalized adjacency
# def getNormalizedAdj(data):
#     row = data.edge_index[0].cpu()
#     col = data.edge_index[1].cpu()
#     raw_data = torch.ones(data.edge_index.shape[1])
#     adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()
#     evals_large, evecs_large = eigsh(adj, 1, which='LM')
#     adj = torch.Tensor(adj / evals_large)
#     adj = adj.to(device)
#     return adj
# #dataset generation
# from torch_geometric.data import Data
#
# p = 0.8
# q = 0.1
# C = 5
# block_size = (args.num_node / C * torch.ones(C, 1)).squeeze().long()
#
#
# edge_prob = q * torch.ones(C, C)
# for i in range(C):
#     edge_prob[i, i] = p
#
# # data_list = []
# # val_data_list = []
# # test_data_list = []
# # for i in range(args.data_size):
# #     edge_index = torch_geometric.utils.stochastic_blockmodel_graph(block_size, edge_prob,directed=False)
# #     if args.distribution == 'normal':
# #         x = torch.cat((torch.randn(args.num_node, 1), torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #     elif args.distribution == 'binomial':
# #         x = torch.cat((torch.Tensor(np.random.binomial(50,0.5,[args.num_node,1])), torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #     elif args.distribution == 'exp':
# #         x = torch.cat((torch.Tensor(np.random.exponential(scale=1.0, size=[args.num_node,1])), torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #     elif args.distribution == 'lognormal':
# #         x = torch.cat((torch.Tensor(np.random.lognormal(mean=0.0, sigma=1.5, size=[args.num_node, 1])), torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #     else:
# #         warnings.warn('You have selected an unexisted distribution\n')
# #         sys.exit()
# #
# #     if args.obj == 'average':
# #         y = (sum(torch.transpose(x,0,1)[0] * torch.transpose(x,0,1)[1]) / sum(torch.transpose(x,0,1)[1]) * torch.ones(args.num_node))
# #     elif args.obj == 'quantile':
# #         y = (np.median(torch.transpose(x,0,1)[0],axis=0).item() * torch.ones(args.num_node))
# #     elif args.obj == 'max':
# #         y = (max(torch.transpose(x,0,1)[0]) * torch.ones(args.num_node))
# #     else:
# #         warnings.warn('You have selected an unexisted objective\n')
# #         sys.exit()
# #
# #     data = Data(x=x, y=y, edge_index=edge_index)
# #
# #     data_list.append(data)
# # for j in range(2):
# #     for i in range(2500):
# #         edge_index = torch_geometric.utils.stochastic_blockmodel_graph(block_size, edge_prob,directed=False)
# #         if args.distribution == 'normal':
# #             x = torch.cat((torch.randn(args.num_node, 1), torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #         elif args.distribution == 'binomial':
# #             x = torch.cat((torch.Tensor(np.random.binomial(50, 0.5, [args.num_node, 1])),
# #                            torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #         elif args.distribution == 'exp':
# #             x = torch.cat((torch.Tensor(np.random.exponential(scale=1.0, size=[args.num_node, 1])),
# #                            torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #         elif args.distribution == 'lognormal':
# #             x = torch.cat((torch.Tensor(np.random.lognormal(mean=0.0, sigma=1.5, size=[args.num_node, 1])),
# #                            torch.randint(1, 100, (args.num_node, 1))), dim=1)
# #         else:
# #             warnings.warn('You have selected an unexisted distribution\n')
# #             sys.exit()
# #
# #         if args.obj == 'average':
# #             y = (sum(torch.transpose(x, 0, 1)[0] * torch.transpose(x, 0, 1)[1]) / sum(torch.transpose(x, 0, 1)[1]) * torch.ones(args.num_node))
# #         elif args.obj == 'quantile':
# #             y = (np.median(torch.transpose(x, 0, 1)[0], axis=0).item() * torch.ones(args.num_node))
# #         elif args.obj == 'max':
# #             y = (max(torch.transpose(x, 0, 1)[0]) * torch.ones(args.num_node))
# #         else:
# #             warnings.warn('You have selected an unexisted objective\n')
# #             sys.exit()
# #
# #         data = Data(x=x, y=y, edge_index=edge_index)
# #         if j == 0 :
# #             val_data_list.append(data)
# #         else:
# #             test_data_list.append(data)
# #
# # train_data = data_list[:int(0.8 * args.data_size)]
# # val_data = val_data_list
# # test_data = test_data_list
#
#
# # torch.save(data_list,'./data_list.pt')
# # torch.save(val_data_list,'./val_data_list.pt')
# # torch.save(test_data_list,'./test_data_list.pt')
#
# # data_list = torch.load('./data_list.pt')
# # val_data_list = torch.load('./val_data_list.pt')
# # test_data_list = torch.load('./test_data_list.pt')
# print(data)
# print(len(test_data),len(val_data),len(train_data))
#
# from torch_geometric.data import DataLoader
# train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
#
# model = GCN(nfeat=2, nhid=32, K=args.K+1, nclasses=1)
#
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
# crit = torch.nn.MSELoss()
#
# fo = open("output.txt","w+")
# model.train()
# ll_train = 1e6
# ll_val = 1e6
# ll_val_lowest = 1e7
# k = 0
# cur_time = time.time()
# # while(ll_val>1e-4):
# while(k<1200):
#     k += 1
#     loss_train_list = []
#     for data in train_dataloader:
#         data.to(device)
#         optimizer.zero_grad()
#         # adj = getTrainNormalizedAdj(data)
#         adj = getNormalizedAdj(data)
#         output = model(data.x, adj).squeeze()
#         loss_train = crit(output, data.y.squeeze())
#
#         loss_train_list.append(loss_train.item())
#         loss_train.backward()
#         optimizer.step()
#     ll_train = sum(loss_train_list) / len(train_data)
#     print(ll_train)
#     fo.write('Epoch: {:04d},'.format(k))
#     fo.write('loss_train: {:.11f},'.format(ll_train))
#
#     loss_val_list = []
#     for j in range(len(val_data)):
#         data = val_data[j]
#         data.to(device)
#         adj = getNormalizedAdj(data)
#         output = model(data.x, adj).squeeze()
#         loss_val = crit(output, data.y.squeeze())
#
#         loss_val_list.append(loss_val.item())
#
#     ll_val = sum(loss_val_list) / len(val_data)
#     print(ll_val)
#     if ll_val < ll_val_lowest:
#         ll_val_lowest = ll_val
#         torch.save(model, './model_best.pt')
#     fo.write('loss_val: {:.11f}\n'.format(ll_val))
#
#     if k % 1000 == 0:
#         torch.save(model, './model_epoch_%d.pt'%(k))
#
# torch.save(model, './model.pt')
#
# end_time = time.time()
# # model.eval()
# # loss_test_list = []
# # for j in range(len(test_data)):
# #     data = test_data[j]
# #     data.to(device)
# #     adj = getNormalizedAdj(data)
# #     output = model(data.x, adj).squeeze()
# #     loss_test = crit(output, data.y.squeeze())
# #     print(output,"\n",data.y.squeeze())
# #     print(loss_test.item())
# #     loss_test_list.append(loss_test.item())
# #     fo.write('data: {:04d},'.format(j))
# #     fo.write('loss_test: {:.11f}\n'.format(loss_test.item()))
# # print("oooooooooooooooooooooooooooooooooooooooooooooooo")
# # print('average loss: %f'%(sum(loss_test_list)/len(test_data)))
# # print(k)
# # print(end_time-cur_time,"s")
# # fo.write('average loss: %f,'%(sum(loss_test_list)/len(test_data)))
# # fo.write('the number of epoch:%d,'%(k))
# # fo.write('total time:%f'%(end_time-cur_time))
# # fo.write('s\n')


# # define the shape of network and the data
# import torch
# from torch_geometric.data import Data
# import argparse
# torch.manual_seed(10)
# parser = argparse.ArgumentParser()
# parser.add_argument('--num_node', type=int, default=12,
#                     help='number of the node')
# parser.add_argument('--num_weight', type=int, default=3,
#                     help='number of the weight')
# args = parser.parse_args()
# edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
#                            [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
# edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
# print(edge_index)
#
# # x = torch.cat((torch.randn(args.num_node, args.num_weight), torch.randint(1, 100, (args.num_node,1))), dim=1)
# x = torch.cat((torch.randn(args.num_node, args.num_weight), torch.ones(args.num_node,1)), dim=1)
# y = []
# multiplier = torch.transpose(x,0,1)[args.num_weight]
# divider = sum(multiplier)
# for i in range(args.num_weight):
#     y.append(sum(torch.transpose(x,0,1)[i] * multiplier) / divider)
# y = torch.Tensor(y) * torch.ones((args.num_node, args.num_weight))
#
# data = Data(x=x,y=y,edge_index=edge_index)
# # define the structure of GNN
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# import torch.optim as optim
#
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
#
# model = GCN(nfeat=args.num_weight+1, nhid=32,nclasses=args.num_weight)
#
# # train the GNN
# k = 0
# device = 'cpu'
# optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
# crit = torch.nn.MSELoss()
#
# while(k<1200):
#     k += 1
#     loss_train_list = []
#     data.to(device)
#     optimizer.zero_grad()
#
#     output = model(data.x, edge_index).squeeze()
#     loss_train = crit(output, data.y.squeeze())
#
#     loss_train_list.append(loss_train.item())
#     loss_train.backward()
#     optimizer.step()
#     print(loss_train.data)
#
# # test the GNN
# output = model(data.x, edge_index).squeeze()
# loss_test = crit(output, data.y.squeeze())
# print(loss_test.data)
# torch.save(model, './model.pt')


import torch
from torch_geometric.data import Data
import argparse
torch.manual_seed(10)
parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=12,
                    help='number of the node')
parser.add_argument('--num_weight', type=int, default=3,
                    help='number of the weight')
parser.add_argument('--batch_size', type=int, default=25,
                    help='the size of a batch')
args = parser.parse_args()
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
print(edge_index)

data_list = []
for k in range(args.data_size):
    x = torch.cat((torch.randn(args.num_node, args.num_weight), torch.ones(args.num_node,1)), dim=1)
    y = []
    multiplier = torch.transpose(x,0,1)[args.num_weight]
    divider = sum(multiplier)
    for i in range(args.num_weight):
        y.append(sum(torch.transpose(x,0,1)[i] * multiplier) / divider)
    y = torch.Tensor(y) * torch.ones((args.num_node, args.num_weight))

    data = Data(x=x,y=y,edge_index=edge_index)

    data_list.append(data)

train_data = data_list[:int(0.8 * args.data_size)]
val_data = data_list[int(0.8 * args.data_size):int(0.9* args.data_size)]
test_data = data_list[int(0.9 * args.data_size):]

from torch_geometric.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# define the structure of GNN
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


model = GCN(nfeat=args.num_weight+1, nhid=32,nclasses=args.num_weight)

# train the GNN
k = 0
device = 'cpu'
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
crit = torch.nn.MSELoss()
loss_train_list = []

while(k<1200):
    k += 1
    for data in train_dataloader:
        data.to(device)
        optimizer.zero_grad()

        output = model(data.x, edge_index).squeeze()
        loss_train = crit(output, data.y.squeeze())

        loss_train_list.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        loss_train_list.append(loss_train.data)
        print(loss_train.data)

# test the GNN
output = model(data.x, edge_index).squeeze()
loss_test = crit(output, data.y.squeeze())
print(loss_test.data)
torch.save(model, './model.pt')
