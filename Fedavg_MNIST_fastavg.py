import torch
import numpy as np
from torchvision.datasets import mnist
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import scipy.sparse as sp
import copy
from scipy.sparse.linalg import eigs, eigsh
import random
import matplotlib.pyplot as plt
torch.manual_seed(0)
L = 12
node_feature = 3

device = 'cpu'
def getBestConstantAlpha(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()

    Laplacian_matrix = adj.dot(adj.transpose(0,1))
    Lap_evals, _ = np.linalg.eig(Laplacian_matrix)
    Lap_evals = sorted(Lap_evals)
    alpha_b_con = 2 / (Lap_evals[1] + Lap_evals[len(Lap_evals)-1])

    return alpha_b_con,torch.Tensor(adj)

def getDMax(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()

    Neighbour = sum(adj)
    alpha_d_max = 1 / max(Neighbour)

    return alpha_d_max,torch.Tensor(adj)

def generateLocalMax(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()

    adj = torch.Tensor(adj)
    Neighbour = sum(adj)
    W = copy.deepcopy(adj)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                W[i][j] = min(1/Neighbour[i],1/Neighbour[j])
    for i in range(adj.shape[0]):
        index_k = 0
        for k in range(adj.shape[1]):
            if adj[i][k] == 1:
                index_k += (max(0, 1 / Neighbour[i] - 1 / Neighbour[k]))
        W[i][i] = index_k

    return W
def getW(alpha,adj):
    W = alpha * copy.deepcopy(adj)
    Neighbour = sum(adj)
    for i in range(adj.shape[0]):
        W[i][i] = 1 - Neighbour[i]*alpha
    return W

def generate_Metropolis_W(adj):
    W = copy.deepcopy(adj)
    Neighbour = sum(adj)

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                W[i][j] = 1 / (1 + max(Neighbour[i], Neighbour[j]))

    Neighbour_sum = sum(W)

    for i in range(adj.shape[0]):
        W[i][i] = 1 - Neighbour_sum[i]
    return W

def generate_Laplacian(step,adj):
    Laplacian = alpha * copy.deepcopy(-adj)
    Neighbour = sum(adj)
    for i in range(adj.shape[0]):
        Laplacian[i][i] = Neighbour[i]
    W = torch.eye(adj.shape[0]) - step * Laplacian
    return W

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = self.softmax(self.fc(x))
        return x
L = 12
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
data = Data(x=torch.randn((L,1)),edge_index=edge_index)
print(edge_index)

target_model = Net()
net_agent = []
optimizer_agent = []
for i in range(L):
    net_agent.append(Net())
    optimizer_agent.append(torch.optim.Adam(net_agent[i].parameters()))

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
param_data_index = torch.zeros((L,7850))

# alpha, adj = getBestConstantAlpha(data)
# alpha, adj = getDMax(data)
# W = getW(alpha, adj)
# W = generate_Metropolis_W(adj)
# W = generate_Laplacian(0.2,adj)
W = generateLocalMax(data)
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

    # for i in range(L):
    #     if i == 0:
    #         for target_param, param in zip(target_model.parameters(), net_agent[i].parameters()):
    #             target_param.data.copy_(param.data)
    #     else:
    #         for target_param, param in zip(target_model.parameters(), net_agent[i].parameters()):
    #             target_param.data.copy_(target_param.data + param.data)
    # for i in range(L):
    #     for target_param, param in zip(target_model.parameters(), net_agent[i].parameters()):
    #         param.data.copy_(target_param.data/L)

    for i in range(L):
        k = 0
        for param in net_agent[i].parameters():
            if k == 0:
                param_data_index[i][:7840] = torch.reshape(param.data,(7840,))
            else:
                param_data_index[i][7840:] = param.data
            k += 1


    for ii in range(10):
        param_data_index = torch.matmul(W,param_data_index)

    for i in range(L):
        k = 0
        for param in net_agent[i].parameters():
            if k == 0:
                param.data.copy_(param_data_index[i][:7840].data.reshape(10,784))
            else:
                param.data.copy_(param_data_index[i][7840:].data)
            k += 1
            # print("#", param.data, "#")

    accuracy_list = []
    for i in range(L):
        for valid_data in valid_dataloader:
            out = net_agent[i](valid_data[0])
            accuracy = sum(torch.max(out, dim=1).indices == valid_data[1])
            accuracy_list.append(accuracy / 64)
    acc_ll.append(sum(accuracy_list)/len(accuracy_list))

# test the GNN
accuracy_list = []
for i in range(L):
    for test_data in test_dataloader:
        out = net_agent[i](test_data[0])
        accuracy = sum(torch.max(out, dim=1).indices == test_data[1])
        accuracy_list.append(accuracy / 128)

    print("the average accuracy of agent ", i, " : ", sum(accuracy_list) / len(accuracy_list))

baseline = torch.load('./baseline.pt')
x_label = [i for i in range(len(acc_ll))]
# plt.plot(x_label,acc_ll,label='BestConstant')
# plt.plot(x_label,acc_ll,label='Metropolis weights')
# plt.plot(x_label,acc_ll,label='Laplacian weights')
# plt.plot(x_label,acc_ll,label='Maximum degree ')
plt.plot(x_label,acc_ll,label='Local degree ')
plt.plot(x_label,baseline[:len(x_label)],label='baseline')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('average accuracy')
plt.title('Fedavg on MNIST dataset')
# plt.yscale('log')
plt.show()
# print(loss_list[len(loss_list)-1])
# torch.save(acc_ll,'./baseline.pt')


