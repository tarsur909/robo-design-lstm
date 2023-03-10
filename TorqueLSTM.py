import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class LSTM(nn.Module):
  def __init__(self, num_classes = 1 , input_size = 2, hidden_size = 1024, num_layers = 4):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first= True,dropout = 0.2)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
  def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
            
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        _, (hn, cn) = self.lstm(x, (h_0, c_0))
        
        y = hn.view(-1, self.hidden_size)
        
        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        
        out = self.fc(final_state)
       
        return out



