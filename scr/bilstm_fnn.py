import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import random



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = None
        self.dropout2 = None
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def get_index(self, df_avg):
        L = []
        for val in df_avg.values.tolist():
            L.extend(val)
        x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
            sorted((x, i) for i, x in enumerate(L)))))
        ord_index = [max(x) - i for i in list(x)]
        return ord_index

    def node_order(self, weights):
        average = torch.mean(weights, axis=0)
        new_average = pd.DataFrame(average.cpu().detach().numpy())
        ord_index = self.get_index(new_average)
        return ord_index

    def forward(self, x):
        if self.type == 'unlearning':
            prev_drp = 0
            try:
                self.dropout1 = nn.Dropout(0.75 - (0.072 * self.turn) + (0.05 * (self.epoch - 1)))
                self.dropout2 = nn.Dropout(0.80 - (0.072 * self.turn) + (0.05 * (self.epoch - 1)))
            except:
                self.dropout1 = nn.Dropout(0.75 - (0.072 ) + (0.05 ))
                self.dropout2 = nn.Dropout(0.80 - (0.072 ) + (0.05))
        elif self.type == 'learning':
            try:
                self.dropout1 = nn.Dropout(0.65 - (0.09 * self.turn))
                self.dropout2 = nn.Dropout(0.75 - (0.09 * self.turn))
            except:
                self.dropout1 = nn.Dropout(0.65 - (0.09 ))
                self.dropout2 = nn.Dropout(0.75 - (0.09))

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        # if self.type == 'unlearning':
        #     # With ordered numbers
        #     rank = torch.tensor([30 for i in range(x.shape[1])])
        #     rank = rank.to(self.device)

            # With ordered nodes
            # rank = self.node_order(x)
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # With to 30 nodes
            # rank = self.node_order(x)
            # rank = [ind if 30 <= ind else 0 for ind in rank]
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # Random node
            # random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            # rank = self.node_order(x)
            # for rn in random_numbers:
            #     try:
            #         rank[rn] = 1
            #
            #     except:
            #         continue
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)
            # x = x * torch.exp(-(self.epoch / rank))
        self.dropout2 = nn.Dropout(random.uniform(0, 1))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        if self.type == 'unlearning':
            # With ordered numbers
            rank = torch.tensor([30 for i in range(x.shape[1])])
            rank = rank.to(self.device)

            # With ordered nodes
            # rank = self.node_order(x)
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # With to 30 nodes
            # rank = self.node_order(x)
            # rank = [ind if 30 <= ind else 0 for ind in rank]
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)

            # Random node
            # random_numbers = random.sample(range(0, x.shape[1] - 1), random.randint(0, x.shape[1] - 1))
            # rank = self.node_order(x)
            # for rn in random_numbers:
            #     try:
            #         rank[rn] = rn*random()
            #
            #     except:
            #         continue
            # rank = torch.tensor(rank)
            # rank = rank.to(self.device)
            x = x * torch.exp(-(self.epoch / rank))

        output = F.log_softmax(x, dim=1)
        return output