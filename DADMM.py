
import torch
import math
import copy
import matplotlib.pyplot as plt

def norm_f(x):
    m,n = x.shape
    x = x.reshape(m*n)
    total_sum = 0
    for i in range(m*n):
        total_sum += x[i]*x[i]
    total_sum = torch.sqrt(total_sum)
    return total_sum

# problem:
L = 10
mi = 3
node_feature = 3
torch.manual_seed(0)
# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,mi))
U = torch.randn((L,mi,node_feature))
x = 300*torch.ones((L,node_feature))
loss_list = []
v = torch.empty((L,mi))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 5
W = 0.5/(L-1) * torch.ones((L,L))
for i in range(L):
    W[i][i] = 0.5
# W = torch.zeros((L,L))
recon_x = torch.zeros((L,node_feature))
alpha = torch.zeros((L,node_feature))
Neighbour = torch.zeros((L))
c = 6e-2
for i in range(L):
    neighbour = 0
    for j in range(L):
        neighbour += bool(W[i][j])
    Neighbour[i] = neighbour

for k in range(Iteration):
    c = 0
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = 0.5 * torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        # print(recon_x)
    # loss = torch.norm(recon_x - x)
    loss = norm_f(recon_x - x) / norm_f(torch.zeros((L,node_feature)) - x)
    loss_list.append(loss.data)
    print("Loss: ",loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        index = c * Neighbour[i] * cp_recon_x[i] + c*torch.matmul(W[i],cp_recon_x) - alpha[i] + torch.matmul(torch.transpose(U[i],0,1),v[i])
        # print(index)
        recon_x[i] = torch.matmul(torch.inverse(torch.matmul(torch.transpose(U[i], 0, 1), U[i]) + 2*c*Neighbour[i]*torch.eye(node_feature)),index)
    for i in range(L):
        alpha[i] = alpha[i] + c * Neighbour[i] * recon_x[i] - c*torch.matmul(W[i],recon_x)



print(len(loss_list))
x_label = [i for i in range(len(loss_list))]
plt.plot(x_label,loss_list)
plt.yscale('log')
plt.show()
print(loss_list[len(loss_list)-1])

# import torch
# import math
# import copy
# import matplotlib.pyplot as plt
# def norm_f(x):
#     m,n = x.shape
#     x = x.reshape(m*n)
#     total_sum = 0
#     for i in range(m*n):
#         total_sum += x[i]*x[i]
#     total_sum = torch.sqrt(total_sum)
#     return total_sum
# # problem:
# L = 200
# mi = 3
# node_feature = 3
# # torch.manual_seed(0)
# # noise = math.sqrt(0.01)*torch.randn((L,3))
# noise = math.sqrt(0) * torch.randn((L, mi))
# U = torch.randn((L, mi, node_feature))
# x = 300 * torch.randn((L, node_feature))
# loss_list = []
# v = torch.empty((L,mi))
# for i in range(L):
#     v[i] = torch.matmul(U[i], x[i]) + noise[i]
#
# Iteration = 100
# W = 1 / L * torch.ones((L, L))
# recon_x = torch.zeros((L, node_feature))
# alpha = torch.zeros((L, node_feature))
# Neighbour = torch.zeros((L))
# c = 2
# for i in range(L):
#     neighbour = 0
#     for j in range(L):
#         neighbour += bool(W[i][j])
#     Neighbour[i] = neighbour
#
# for k in range(Iteration):
#     # c = 6e-4
#     for i in range(L):
#         formula = (v[i] - torch.matmul(U[i], recon_x[i]))
#         # f = 0.5 * torch.norm(formula) * torch.norm(formula)
#         f = 0.5 * torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
#         # print(recon_x)x
#     loss = torch.norm(recon_x - x)/torch.norm(x)
#     # loss = norm_f(recon_x - x) / norm_f(torch.zeros((L, node_feature)) - x)
#     loss_list.append(loss.data)
#     print("Loss: ", loss.data)
#     cp_recon_x = copy.deepcopy(recon_x)
#     for i in range(L):
#         index = c * Neighbour[i] * cp_recon_x[i] + c * torch.matmul(W[i], cp_recon_x) - alpha[i] + torch.matmul(torch.transpose(U[i],0,1),v[i])
#         # print(index)
#         recon_x[i] = torch.matmul(
#             torch.inverse( 2 * c * Neighbour[i] * torch.eye(node_feature) - torch.matmul(torch.transpose(U[i], 0, 1), U[i])),
#             index)
#     for i in range(L):
#         alpha[i] = alpha[i] + c * Neighbour[i] * recon_x[i] - c * torch.matmul(W[i], recon_x)
#
# print(len(loss_list))
# x_label = [i for i in range(len(loss_list))]
# plt.plot(x_label, loss_list)
# plt.yscale('log')
# plt.show()
# print(loss_list[len(loss_list) - 1])
