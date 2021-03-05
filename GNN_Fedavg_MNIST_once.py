import torch
from torchvision.datasets import mnist
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
import math
from torch_geometric.nn import GCNConv
import copy
import matplotlib.pyplot as plt
import random
device = 'cpu'
torch.manual_seed(0)
L = 12
node_feature = 3
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

model = torch.load('./model.pt',map_location=torch.device('cpu'))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = self.softmax(self.fc(x))
        return x
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
# print(edge_index)
data = Data(x=torch.randn((L,1)),edge_index=edge_index)
target_model = Net()
net_agent = []
optimizer_agent = []
for i in range(L):
    net_agent.append(Net())
    optimizer_agent.append(torch.optim.SGD(net_agent[i].parameters(),lr=1e-6))

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

train_set = list(mnist.MNIST('data/mnist', train=True, transform=transform, download=True))
valid_set = train_set[int(0.8 * len(train_set)):]
train_set = train_set[:int(0.8 * len(train_set))]

sorted_train_set = sorted(train_set, key=lambda t: t[1])
test_set = mnist.MNIST('data/mnist', train=False, transform=transform, download=True)
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False)

# generate sorted training dataset
number = torch.zeros(10)
for i in range(len(sorted_train_set)):
    number[sorted_train_set[i][1]] += 1

# train_set_size = sorted(torch.Tensor([0] + random.sample(range(1, len(train_set)), L - 1) + [len(train_set) - 1]))
train_set_size = [0, 4456, 6349, 17228, 21839, 25415, 28386, 31255, 35466, 36174, 42703, 45126, 47999]
print(train_set_size)

train_set_agent = []
train_dataloader_agent = []
for i in range(L):
    train_set_agent.append(train_set[int(train_set_size[i]): int(train_set_size[i + 1])])
    train_dataloader_agent.append(DataLoader(train_set_agent[i], batch_size=64, shuffle=True))

# train the GNN
crit = torch.nn.MSELoss()
acc_ll = []

for epoch in range(10):
    loss_list = []
    for i in range(L):
        for train_data in train_dataloader_agent[i]:
            optimizer_agent[i].zero_grad()
            target = torch.nn.functional.one_hot((train_data[1]).to(torch.int64), num_classes=10)
            target = target.type(torch.FloatTensor)
            out = net_agent[i](train_data[0])
            loss = crit(out, target)
            loss_list.append(loss.data)
            loss.backward()
            optimizer_agent[i].step()
        print("the loss of agent ", i, " : ", sum(loss_list) / (len(loss_list)))

    model_parameter_list = []
    for i in range(L):
        k = 0
        for param in net_agent[i].parameters():
            if k == 0:
                model_parameter_index = copy.deepcopy(param.data)
            else:
                model_parameter_index = torch.cat((model_parameter_index,param.data.unsqueeze(1)),1)
            k += 1
            # print("#",param.data,"#")
        model_parameter_list.append(model_parameter_index.reshape(7850))
    model_parameter_tensor = torch.Tensor([t.numpy() for t in model_parameter_list] )

    adj = getNormalizedAdj(data)
    gnn_value = torch.zeros((7850,12))
    for ii in range(7850):
        model_input = (model_parameter_tensor.T[ii].T).unsqueeze(1)
        gnn_value[i] = model(model_input, adj).squeeze(1)

    gnn_value = gnn_value.T

    for i in range(L):
        k = 0
        for param in net_agent[i].parameters():
            if k == 0:
                param.data.copy_(gnn_value[i][:7840].data.reshape(10,784))
            else:
                param.data.copy_(gnn_value[i][7840:].data)
            k += 1
            # print("#", param.data, "#")

    accuracy_list = []
    for i in range(L):
        for valid_data in valid_dataloader:
            out = net_agent[i](valid_data[0])
            accuracy = sum(torch.max(out, dim=1).indices == valid_data[1])
            accuracy_list.append(accuracy / 64)
    acc_ll.append(sum(accuracy_list)/len(accuracy_list))

        # model()
    # for i in range(L):
        # if i == 0:
    #         for target_param, param in zip(target_model.parameters(), net_agent[i].parameters()):
    #             target_param.data.copy_(param.data)
    #     else:
    #         for target_param, param in zip(target_model.parameters(), net_agent[i].parameters()):
    #             target_param.data.copy_(target_param.data + param.data)
    # for i in range(L):
    #     for target_param, param in zip(target_model.parameters(), net_agent[i].parameters()):
    #         print(target_param.data)
    #         param.data.copy_(target_param.data/L)
x_label = [i for i in range(len(acc_ll))]
plt.plot(x_label,acc_ll)
plt.xlabel('epochs')
plt.ylabel('average accuracy')
plt.title('GNN_avg Algorithm operated on MNIST dataset')
plt.show()

# test the GNN
accuracy_list = []
for i in range(L):
    for test_data in test_dataloader:
        out = net_agent[i](test_data[0])
        accuracy = sum(torch.max(out, dim=1).indices == test_data[1])
        accuracy_list.append(accuracy / 128)

    print("the average accuracy of agent ", i, " : ", sum(accuracy_list) / len(accuracy_list))
