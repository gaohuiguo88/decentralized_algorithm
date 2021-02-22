import torch
import math
import copy
import matplotlib.pyplot as plt
# problem:
L = 10
torch.manual_seed(0)
# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,3))
U = torch.randn((L,3,3))
x = 300*torch.ones((L,3))
loss_list = []
v = torch.empty((L,3))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 1000
W = 1/L * torch.ones((L,L))
recon_x = torch.zeros((L,3))
alpha = torch.zeros((L,3))
Neighbour = torch.zeros((L))
c = 6e-19
for i in range(L):
    neighbour = 0
    for j in range(L):
        neighbour += bool(W[i][j])
    Neighbour[i] = neighbour

for k in range(Iteration):
    c = 3e-20
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        # print(recon_x)x
    loss = torch.norm(recon_x - x)
    loss_list.append(loss.data)
    print("Loss: ",loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        index = c * Neighbour[i] * cp_recon_x[i] + c*torch.matmul(W[i],cp_recon_x) - alpha[i] + 2*torch.matmul(torch.transpose(U[i],0,1),v[i])
        # print(index)
        recon_x[i] = torch.matmul(torch.inverse(2*torch.matmul(torch.transpose(U[i], 0, 1), U[i]) + 2*c*Neighbour[i]*torch.eye(3)),index)
    for i in range(L):
        alpha[i] = alpha[i] + c * Neighbour[i] * recon_x[i] - c*torch.matmul(W[i],recon_x)



print(len(loss_list))
x_label = [i for i in range(len(loss_list))]
plt.plot(x_label,loss_list)
plt.yscale('log')
plt.show()
print(loss_list[len(loss_list)-1])
