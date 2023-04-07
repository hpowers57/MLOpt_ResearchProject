import numpy as np
from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, nonlinearity="tanh"):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlinearity, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.rnn(x, hidden)
        
        out = self.out(out[:,-1,:])
        
        return out
      
      
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        out, _ = self.gru(x, hidden.detach())
        
        out = self.out(out[:,-1,:])
        
        return out

    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size[0], num_layers, batch_first=True)
        self.y1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.6)
        self.y2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.dropout = nn.Dropout()
        self.y3 = nn.Linear(hidden_size[2], output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size[0])
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size[0])

        hiddenStates, _ = self.lstm(x, (h0, c0))

        lstm_out = self.y1(hiddenStates[:, -1, :])
        lstm_out = self.elu(lstm_out)

        dense_out = self.y2(lstm_out)
        dense_out = self.elu(dense_out)
        dense_out = self.dropout(dense_out)

        out = self.y3(dense_out)

        return out
