import torch
import torch.nn.functional as F
import numpy as np

import hps
from data_utils import DataLoader


class LSTMseq2seq(torch.nn.Module):

    def __init__(self, n_cell=hps.n_cell, n_unit=hps.n_unit, dropout=hps.dropout):

        super(LSTMseq2seq, self).__init__()

        self.n_unit = n_unit
        self.n_cell = n_cell
        self.dropout = dropout

        dataloader = DataLoader('train')
        self.data_shape = dataloader.get_datashape()
        del dataloader

        self.lstm = torch.nn.LSTM(
            self.data_shape[1],
            self.n_unit,
            self.n_cell
        )
        self.linear = torch.nn.Linear(self.n_unit, self.data_shape[1])

    def init_hidden(self, batch_size):

        hidden = [
            torch.autograd.Variable(torch.zeros(hps.n_cell, batch_size, self.n_unit)).cuda(),
            torch.autograd.Variable(torch.zeros(hps.n_cell, batch_size, self.n_unit)).cuda()
        ]
        return hidden

    def forward(self, x, mode, future=0):
        #x = x.cuda()
        hidden = self.init_hidden(x.size(1))
        if mode == 'train':
            self.lstm.flatten_parameters()
            lstm_out, hidden = self.lstm(x, hidden)
            linear_out = self.linear(lstm_out)
            relu_out = F.relu(linear_out)
            print(relu_out.shape)
            out = F.dropout(relu_out, self.dropout, training=True)
            out = {
                'asu': F.softmax(out[:, :, :self.data_shape[2][0]]),
                'size': out[:, :, self.data_shape[2][0]:self.data_shape[2][1]],
                'opcode/home/zxi': F.softmax(out[:, :, self.data_shape[2][1]:self.data_shape[2][2]]),
                'time_diff': out[:, :, self.data_shape[2][2]:]
            }
        else:
            self.lstm.flatten_parameters()
            lstm_out, hidden = self.lstm(x, hidden)
            last = torch.unsqueeze(x[-1], 0).cuda()
            out = torch.autograd.Variable(torch.zeros(future, x.size(1), x.size(2))).cuda()
            for cnt in range(future):
                self.lstm.flatten_parameters()
                lstm_out, hidden = self.lstm(last, hidden)
                linear_out = self.linear(lstm_out)
                relu_out = F.relu(linear_out)
                last = relu_out
                last[:, :, :self.data_shape[2][0]] = F.softmax(last[:, :, :self.data_shape[2][0]])
                last[:, :, self.data_shape[2][1]:self.data_shape[2][2]] =\
                    F.softmax(last[:, :, self.data_shape[2][1]:self.data_shape[2][2]])
                out.data[cnt] = last.data
            out = {
                'asu': out[:, :, self.data_shape[2][0]],
                'size': out[:, :, self.data_shape[2][0]:self.data_shape[2][1]],
                'opcode': out[:, :, self.data_shape[2][1]:self.data_shape[2][2]],
                'time_diff': out[:, :, self.data_shape[2][2]:]
            }
        return out


if __name__ == '__main__':
    lstm = LSTMseq2seq()
    pass