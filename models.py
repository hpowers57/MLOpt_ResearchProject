from torch import nn
import torch


class Martingale:
    # Martingale model as discussed in Elliot et al (2017)
    def __init__(self):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, x):
        return x[:,-1]


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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.out(out[:, -1, :])
        
        return out 


class stack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(stack, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        _, hn1 = self.gru(x, h0.detach())
        _, hn2 = self.gru(x, hn1.detach())
        _, hn3 = self.gru(x, hn2.detach())

        _, (hn4, cn1) = self.lstm(x, (hn3.detach(), c0.detach()))
        _, (hn5, _) = self.lstm(x, (hn4.detach(), cn1.detach()))

        _, hn6 = self.gru(x, hn5.detach())
        _, hn7 = self.gru(x, hn6.detach())
        out, _ = self.gru(x, hn7.detach())

        out = self.out(out[:, -1, :])
        
        return out
