import torch
import matplotlib.pyplot as plt

baseline = torch.load('./baseline.pt')
acc_ll_bestconstant_weights = torch.load('./acc_ll_bestconstant_weights.pt')
acc_ll_metropolis_weights = torch.load('./acc_ll_metropolis_weights.pt')
acc_ll_laplacian_weights = torch.load('./acc_ll_laplacian_weights.pt')
acc_ll_mnist_max_degree = torch.load('./acc_ll_mnist_max_degree.pt')
acc_ll_mnist_local_degree = torch.load('./acc_ll_mnist_local_degree.pt')
acc_ll_mnist_gnn = torch.load('./acc_ll_mnist_gnn.pt')

x_label = [i for i in range(100)]
plt.plot(x_label,baseline[:len(x_label)],label='baseline')

# plt.plot(x_label,acc_ll_bestconstant_weights,label='BestConstant')
# plt.plot(x_label,acc_ll_metropolis_weights,label='Metropolis weights')
# plt.plot(x_label,acc_ll_laplacian_weights,label='Laplacian weights')
# plt.plot(x_label,acc_ll_mnist_max_degree,label='Maximum degree ')
# plt.plot(x_label,acc_ll_mnist_local_degree,label='Local degree ')
# plt.plot(x_label,acc_ll_mnist_gnn,label='GNNavg ')

plt.legend()
plt.xlabel('epochs')
# plt.ylim(0.9140,0.9155)
# plt.xlim(18,19)
plt.ylabel('average accuracy')
plt.title('OtherMthodAvg on MNIST dataset')
plt.show()