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
recon_x = torch.zeros((L,3),requires_grad=True)
alpha = torch.zeros((L,3))
Neighbour = torch.zeros((L))
c = 1
for i in range(L):
    neighbour = 0
    for j in range(L):
        neighbour += bool(W[i][j])
    Neighbour[i] = neighbour

for k in range(Iteration):
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        f.backward()
        # print(recon_x)
        loss = torch.norm(recon_x - x)
        # print(loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        with torch.no_grad():
            index = c * Neighbour[i] * recon_x[i] + c*torch.matmul(W[i],cp_recon_x) - alpha[i]
            recon_x[i] = torch.matmul(torch.inverse(torch.diag(recon_x.grad[i]) + 2*c*Neighbour[i]*torch.eye(3)),index)
    for i in range(L):
        with torch.no_grad():
            alpha[i] = alpha[i] + c * Neighbour[i] * recon_x[i] - c*torch.matmul(W[i],recon_x)

for i in range(L):
    print(recon_x[i])


