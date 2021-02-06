import torch
import math
import copy
# problem:
L = 10
# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,3))
U = torch.randn((L,3,3))
x = 0.9*torch.ones((L,3))

v = torch.empty((L,3))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 10000
W = 1/L * torch.ones((L,L))
recon_x = torch.zeros((L,3),requires_grad=True)
store_grad = torch.zeros((L,3))
y = torch.zeros((L,3))
for k in range(Iteration):
    alpha = 0.1/(k+1)
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        f.backward()
        # print(recon_x)
        loss = torch.norm(recon_x - x)
        # print(loss.data)
    if k == 0:
        for i in range(L):
            y[i] = recon_x.grad[i]
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        with torch.no_grad():
            # print(recon_x.grad[i])
            recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * y[i]
            y[i] = torch.matmul(W[i],y) + recon_x.grad[i] - store_grad[i]
            store_grad[i] = recon_x.grad[i]

for i in range(L):
    print(recon_x[i])


