import torch.nn as nn
import torch.nn.functional as F
import torch


class selfattention(nn.Module):
    def __init__(self,d_in,d_out,device):
       super(selfattention,self).__init__()
       self.d_in = d_in
       self.d_out = d_out
       # Linear has better initialisation compared to other ways of iniitilisation
       self.device = device
       self.query = nn.Linear(d_in,d_out,device=device) ## d_in,d_out
       self.key = nn.Linear(d_in,d_out,device = device) ## d_in ,d_out
       self.value = nn.Linear(d_in,d_out, device = device) ## d_in,d_out
       self.softmax = nn.Softmax(dim = -1 )
    def forward(self,X):
        qx = self.query(X)
        kx = self.key(X)
        vx = self.value(X)
        # K.T
        kx = torch.transpose(kx,1,2)
        y = qx @ kx
        y /= (self.d_out)**0.5
        # creating a mask for causal attention
        mask = torch.ones(y.shape, device=self.device)
        mask = torch.triu(mask,diagonal=1)
        mask = mask.bool()
        # applying mask
        y  = torch.masked_fill(y,mask,-torch.inf)
        # applying softmax
        y = self.softmax(y)
        # y = (QK.T)@ V
        y = y @ vx  
        return y
