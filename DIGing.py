import torch
import math
import copy
L = 10
torch.manual_seed(0)
noise = math.sqrt(0)*torch.randn((L,3))
U = torch.randn((L,3,3))
x = torch.rand((L,3))
v = torch.empty((L,3))
# print(U)
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]
alpha = 0.1
Iteration = 10000
W = 1/L * torch.ones((L,L))
recon_x = torch.zeros((L,3),requires_grad=True)
store_grad = torch.zeros((L,3))
y = torch.zeros((L,3))

loss = torch.norm(recon_x - x)
# print("loos_x: ",loss.data)
# Initialization y
for i in range(L):
    formula = v[i] - torch.matmul(U[i], recon_x[i])
    f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
    f.backward()

with torch.no_grad():
    for i in range(L):
        y[i] = recon_x.grad[i]
        store_grad[i] = recon_x.grad[i]
        # print(recon_x.grad[i])

    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * recon_x.grad[i]
        # print(recon_x[i])

loss = torch.norm(recon_x - x)
# print("loos_x: ",loss.data)

for k in range(Iteration):
    alpha = 0.1/(k+2)
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        f.backward()
        # print(recon_x)

    with torch.no_grad():
        cp_recon_x = copy.deepcopy(recon_x)
        cp_y = copy.deepcopy(y)

        for i in range(L):
            y[i] = torch.matmul(W[i], cp_y) + recon_x.grad[i] - store_grad[i]
            store_grad[i] = recon_x.grad[i]

        loss_y = torch.norm(recon_x.grad-y)
        # print("loss_y: ",loss_y.data)

        for i in range(L):
            recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * y[i]

        loss = torch.norm(recon_x - x)
        # print("loos_x: ",loss.data)

for i in range(L):
    print(recon_x[i]-x[i])


