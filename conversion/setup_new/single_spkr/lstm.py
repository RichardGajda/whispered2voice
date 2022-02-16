import torch
import torch.nn as nn
from torch.autograd import Variable


class MyBLSTM(nn.Module):
    def __init__(self, dim_in, hidden_size, dim_out, num_layers=1, bidirectional=True):
        super(MyBLSTM, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(dim_in, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True, dropout=0.1)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, dim_out)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)))
        return h, c

    def forward(self, sequence, lengths, h, c):

        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        return output



