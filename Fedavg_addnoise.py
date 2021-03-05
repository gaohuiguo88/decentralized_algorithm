import torch
import math
import copy
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_geometric.data import Data

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



torch.manual_seed(0)
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

fr = open("output.txt", "r")
line = fr.readline()
line_x = line.split()
noise = torch.randn(L*mi)
U = torch.randn(L*mi*node_feature)

for i in range(len(line_x)):
    if i < L*mi:
        noise[i] = float(line_x[i])
    else:
        U[i-L*mi] = float(line_x[i])

noise = noise.reshape((L,mi))
U = U.reshape((L,mi,node_feature))

# noise = math.sqrt(0.01)*torch.randn((L,3))
# noise = math.sqrt(0)*torch.randn((L,mi))
# U = torch.randn((L,mi,node_feature))
# x = torch.ones((L,node_feature))
# print(U)
v = torch.empty((L,mi))
for i in range(L):
    # v[i] = torch.matmul(U[i],x[i])+noise[i]
    v[i] = torch.matmul(U[i],x[i])

Iteration = 2500
# W = 1/L * torch.ones((L,L))
W = generate_Metropolis_W(data)
for sigma in [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,0]:
    recon_x = torch.zeros((L,node_feature),requires_grad=True)
    loss_list = []
    for k in range(Iteration):
        alpha = 6e-2
        for i in range(L):
            formula = v[i]-torch.matmul(U[i],recon_x[i])
            f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
            f.backward()
            # print(recon_x)
            # loss = torch.norm(recon_x - x)
            # print(loss.data)
        loss = norm_f(recon_x - x)/norm_f(torch.zeros((L,node_feature))-x)
        loss_list.append(loss.data)
        cp_recon_x = copy.deepcopy(recon_x)
        for i in range(L):
            with torch.no_grad():
                # print(recon_x.grad[i])
                recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * recon_x.grad[i]
        for i in range(L):
            with torch.no_grad():
                # print(recon_x.shape)
                # print(sum(recon_x).shape)
                recon_x[i] = sum(recon_x)/L + sigma * torch.rand(1)
    # for i in range(L):
    #     print(recon_x[i])
    print(len(loss_list))
    x_label = [i for i in range(len(loss_list))]
    plt.plot(x_label,loss_list,label='sigma=%f'%sigma)
    plt.xlabel('k')
    plt.ylabel('residual')
    plt.title('Federated Averaging Algorithm Adding Noise to the Aggregation part(Fedavg)')
    plt.yscale('log')
    plt.legend()
    print(loss_list[len(loss_list)-1])
plt.show()