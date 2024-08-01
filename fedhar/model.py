
import torch.nn as nn
import torch
from torch.nn.modules.module import Module

# har model: lstm

class HAR_lstm(nn.Module):
    
    def __init__(self,input_dim,args):
        super().__init__()
        self.embedding = nn.Linear(in_features=input_dim, out_features= args.hidden_dim)
        self.lstm_layer = nn.LSTM(input_size = args.hidden_dim, hidden_size = args.hidden_dim)
        self.output = nn.Linear(in_features=args.hidden_dim, out_features=args.class_num)

    def encoder(self,input):
        
        h = self.embedding(input)
        lstm_out,_ = self.lstm_layer(h)
        return lstm_out[:,-1,:]

    def forward(self,input):
        self.lstm_layer.flatten_parameters() 
        lstm_out = self.encoder(input)
        predict_y = self.output(lstm_out)
        return predict_y
