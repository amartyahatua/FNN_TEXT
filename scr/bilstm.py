import torch.nn as nn
import torch
import pandas as pd

class BiLSTM(nn.Module):

    def __init__(self, le, opt, embedding_matrix):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(opt.max_features, opt.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(opt.embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.fc = nn.Linear(64, 64)
        self.out = nn.Linear(64, n_classes)

    def get_index(self, df_avg):
        L = []
        for val in df_avg.values.tolist():
            L.extend(val)
        x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
            sorted((x, i) for i, x in enumerate(L)))))
        ord_index = [max(x) - i+1 for i in list(x)]
        return ord_index

    def node_order(self, weights):
        average = torch.mean(weights, axis=0)
        new_average = pd.DataFrame(average.cpu().detach().numpy())
        ord_index = self.get_index(new_average)
        return ord_index

    def forward(self, x):
        # rint(x.size())
        if self.type == 'learning':
            h_embedding = self.embedding(x)
            # _embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
            h_lstm, _ = self.lstm(h_embedding)
            avg_pool = torch.mean(h_lstm, 1)
            max_pool, _ = torch.max(h_lstm, 1)
            conc = torch.cat((avg_pool, max_pool), 1)
            conc = self.relu(self.linear(conc))
            self.dropout = nn.Dropout(0.1)
            conc = self.dropout(conc)
            fc_out = self.fc(conc)
            out = self.out(fc_out)
            return out
        if self.type == 'unlearning':
            h_embedding = self.embedding(x)
            # _embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
            h_lstm, _ = self.lstm(h_embedding)
            avg_pool = torch.mean(h_lstm, 1)
            max_pool, _ = torch.max(h_lstm, 1)
            conc = torch.cat((avg_pool, max_pool), 1)
            conc = self.relu(self.linear(conc))

            try:
                self.dropout1 = nn.Dropout(0.75 - (0.072 * self.turn) + (0.05 * (self.epoch - 1)))
            except:
                self.dropout1 = nn.Dropout(0.75 - (0.072 ) + (0.05 ))

            conc = self.dropout1(conc)
            fc_out = self.fc(conc)

            if self.rank==1:
                rank = torch.tensor([i for i in range(1, 1+fc_out.shape[1])])
                rank = rank.cuda()
            elif self.rank==2:
                rank = self.node_order(fc_out)
                rank = torch.tensor(rank)
                rank = rank.cuda()
            elif self.rank==3:
                rank = torch.tensor([30 for i in range(fc_out.shape[1])])
                rank = rank.cuda()
            elif self.rank==4:
                rank = self.node_order(fc_out)
                rank = [ind+1 if 30 <= ind else 1 for ind in rank]
                rank = torch.tensor(rank)
                rank = rank.cuda()

            fc_out = fc_out * torch.exp(-(self.epoch)/rank)
            out = self.out(fc_out)
            return out