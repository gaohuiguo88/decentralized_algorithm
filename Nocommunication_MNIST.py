import torch
from torchvision.datasets import mnist
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
torch.manual_seed(0)
L = 12
node_feature = 3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = self.softmax(self.fc(x))
        return x

net_agent = []
optimizer_agent = []
for i in range(L):
    net_agent.append(Net())
    optimizer_agent.append(torch.optim.Adam(net_agent[i].parameters()))

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

train_set = list(mnist.MNIST('data/mnist', train=True, transform=transform, download=True))
valid_set = train_set[int(0.8*len(train_set)):]
train_set = train_set[:int(0.8*len(train_set))]

sorted_train_set = sorted(train_set, key=lambda t: t[1])
test_set = mnist.MNIST('data/mnist', train=False, transform=transform, download=True)
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False)



#generate sorted training dataset
number = torch.zeros(10)
for i in range(len(sorted_train_set)):
    number[sorted_train_set[i][1]] += 1

# train_set_size = sorted(torch.Tensor([0] + random.sample(range(1,len(train_set)),L-1) + [len(train_set)-1]))
train_set_size = [0, 4456, 6349, 17228, 21839, 25415, 28386, 31255, 35466, 36174, 42703, 45126, 47999]
print(train_set_size)
train_set_agent = []
train_dataloader_agent = []
for i in range(L):
    train_set_agent.append(train_set[int(train_set_size[i]) : int(train_set_size[i+1])])
    train_dataloader_agent.append(DataLoader(train_set_agent[i], batch_size=64, shuffle=True))

# train the GNN
crit = torch.nn.MSELoss()

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
        # print("the loss of agent ",i," : ",sum(loss_list)/(len(loss_list)))

# test the GNN
accuracy_list = []
for i in range(L):
    for test_data in test_dataloader:
        out = net_agent[i](test_data[0])
        accuracy = sum(torch.max(out,dim=1).indices == test_data[1])
        accuracy_list.append(accuracy/128)

    print("the average accuracy of agent ",i," : ",sum(accuracy_list)/len(accuracy_list))