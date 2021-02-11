import torch
import math
import copy
# problem:
L = 10
# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,3))
U = torch.randn((L,3,3))
x = 1.9*torch.ones((L,3))

v = torch.empty((L,3))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 100
W = 1/L * torch.ones((L,L))
recon_x = torch.zeros((L,3))
alpha = torch.zeros((L,3))
Neighbour = torch.zeros((L))
c = 0.1
for i in range(L):
    neighbour = 0
    for j in range(L):
        neighbour += bool(W[i][j])
    Neighbour[i] = neighbour

for k in range(Iteration):
    c = 0.1/(k+1)
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        # print(recon_x)x
        loss = torch.norm(recon_x - x)
    print("Loss: ",loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        index = c * Neighbour[i] * cp_recon_x[i] + c*torch.matmul(W[i],cp_recon_x) - alpha[i] + 2*torch.matmul(torch.transpose(U[i],0,1),v[i])
        # print(index)
        recon_x[i] = torch.matmul(torch.inverse(2*torch.matmul(torch.transpose(U[i], 0, 1), U[i]) + 2*c*Neighbour[i]*torch.eye(3)),index)
    for i in range(L):
        alpha[i] = alpha[i] + c * Neighbour[i] * recon_x[i] - c*torch.matmul(W[i],recon_x)



