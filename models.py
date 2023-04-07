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
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.y1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        hidden, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        lstm_out = self.y1(hidden[:, -1, :])
        
        return lstm_out 

    
# class stackedLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=2):
#         super(stackedLSTM, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.y1 = nn.Linear(hidden_size, output_size)
#         self.dropout1 = nn.Dropout(0.3)
#         self.y2 = nn.Linear(hidden_size[1], hidden_size[2])
#         self.elu = nn.ELU(alpha=0.5)
#         self.dropout2 = nn.Dropout(p=0.6)
#         self.y3 = nn.Linear(hidden_size[2], output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
#         hidden, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
#         lstm_out = self.y1(hidden[:, -1, :])
#         lstm_out = self.dropout1(lstm_out)

#         dense_out = self.y2(lstm_out)
#         dense_out = self.elu(dense_out)
#         dense_out = self.dropout2(dense_out)
        
#         return self.y3(dense_out)
