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

torch.manual_seed(0)
# problem:
L = 10
# noise = math.sqrt(0.01)*torch.randn((L,3))
noise = math.sqrt(0)*torch.randn((L,3))
U = torch.randn((L,3,3))
x = 300*torch.ones((L,3))
# print(U)
v = torch.empty((L,3))
for i in range(L):
    v[i] = torch.matmul(U[i],x[i])+noise[i]

Iteration = 2500
W = 1/L * torch.ones((L,L))
recon_x = torch.zeros((L,3),requires_grad=True)
loss_list = []

for k in range(Iteration):
    alpha = 6e-2
    for i in range(L):
        formula = v[i]-torch.matmul(U[i],recon_x[i])
        f = torch.matmul(torch.transpose(formula.unsqueeze(1), 0, 1), formula.unsqueeze(1))
        f.backward()
        # print(recon_x)
    loss = norm_f(recon_x - x)/norm_f(torch.zeros((L,3))-x)
    loss_list.append(loss.data)
    print("Iteration ", k, " : ",loss.data)
    cp_recon_x = copy.deepcopy(recon_x)
    for i in range(L):
        with torch.no_grad():
            # print(recon_x.grad[i])
            recon_x[i] = torch.matmul(W[i],cp_recon_x) - alpha * recon_x.grad[i]


    # if k == 0:
    #     for i in range(L):
    #         # print(recon_x.grad[i])
    #         print(recon_x[i])

# for i in range(L):
#     print(recon_x[i])
print(len(loss_list))
x_label = [i for i in range(len(loss_list))]
plt.plot(x_label,loss_list)
plt.yscale('log')
plt.show()
print(loss_list[len(loss_list)-1])


