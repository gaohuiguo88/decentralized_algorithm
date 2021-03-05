import torch
from torch_geometric.data import Data
import argparse
import math
# torch.manual_seed(10)
parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=12,
                    help='number of the node')
parser.add_argument('--num_weight', type=int, default=7840,
                    help='number of the weight')
parser.add_argument('--batch_size', type=int, default=25,
                    help='the size of a batch')
args = parser.parse_args()
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
                           [5, 7, 9, 10, 9, 11, 4, 5, 8, 11, 6, 8, 10, 8, 9, 11, 6, 7, 11, 9, 11, 9, 10]], dtype=torch.long)
edge_index = torch.cat((edge_index,torch.cat((edge_index[1],edge_index[0])).reshape(2,-1)),1)
print(edge_index)

data_list = []
for k in range(args.data_size):
    x = torch.cat((torch.randn(args.num_node, args.num_weight), torch.ones(args.num_node,1)), dim=1)
    y = []
    multiplier = torch.transpose(x,0,1)[args.num_weight]

    divider = sum(multiplier)
    for i in range(args.num_weight):
        y.append(sum(torch.transpose(x,0,1)[i] * multiplier) / divider)
    y = torch.Tensor(y) * torch.ones((args.num_node, args.num_weight))

    # for i in range(args.num_node):
    #     x[i] = x[i] * x[i][args.num_weight]
    #     x[i][args.num_weight] = math.sqrt( x[i][args.num_weight])

    data = Data(x=x,y=y,edge_index=edge_index)
    data_list.append(data)

train_data = data_list[:int(0.8 * args.data_size)]
val_data = data_list[int(0.8 * args.data_size):int(0.9* args.data_size)]
test_data = data_list[int(0.9 * args.data_size):]

from torch_geometric.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# define the structure of GNN
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclasses):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(in_channels=nfeat,out_channels=nhid)
        self.gc2 = GCNConv(in_channels=nhid,out_channels=nhid)
        self.lin = nn.Linear(nhid, nclasses)

    def forward(self, x, edge_index):

        x = F.relu(self.gc1(x,edge_index))
        x = F.relu(self.gc2(x,edge_index))
        x = self.lin(x)
        return x


model = GCN(nfeat=args.num_weight, nhid=32,nclasses=args.num_weight)

# train the GNN
k = 0
device = 'cpu'
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
crit = torch.nn.MSELoss()
loss_train_list = []

while(k<1200):
    k += 1
    for data in train_dataloader:
        data.to(device)
        optimizer.zero_grad()

        output = model(data.x, edge_index).squeeze()
        loss_train = crit(output, data.y.squeeze())

        loss_train_list.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        print(loss_train.data)

# test the GNN
output = model(data.x, edge_index).squeeze()
loss_test = crit(output, data.y.squeeze())
print(loss_test.data)
torch.save(model, './model.pt')
