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
