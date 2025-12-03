import torch
import torch.nn as nn


class multiheadkvcache(nn.Module):
    def __init__(self,din,dout,n_heads):
        super(multiheadkvcache,self).__init__()
        self.din = din
        self.dout = dout
        self.n_heads = n_heads
        self.dheads = int(dout/n_heads)
        self.query = nn.Linear(din,dout)
        self.key = nn.Linear(din,dout)
        self.value = nn.Linear(din,dout)
        self.softmax = nn.Softmax(dim = -1)
        self.out_proj = nn.Linear(dout,dout)
        self.register_buffer("cachedk",None,persistent=False)
        self.register_buffer("cachedv", None,persistent=False)
        self.register_buffer("mask", None,persistent=False)
    def forward(self,X,training = False):
        batch_size,n,din = X.shape
        # getting the q,k,v matrices
        q = self.query(X).view(batch_size,n,self.dout)
        k = self.key(X).view(batch_size,n,self.dout)
        v = self.value(X).view(batch_size,n,self.dout) ## (batch_size,n,din)
        # splitting accross heads
        q = q.view(batch_size,n,self.n_heads, self.dheads)
        k = k.view(batch_size,n,self.n_heads, self.dheads)
        v = v.view(batch_size,n,self.n_heads, self.dheads)
        # if training is true we use cached K and V
        if training is False:
            if self.cachedk is None:
                self.cachedk = k
                self.cachedv = v
            else :
                self.cachedk = torch.concat([self.cachedk,k], dim = 1)
                self.cachedv = torch.concat([self.cachedv,v], dim = 1)
                k = self.cachedk
                v = self.cachedv
            # this is to prevent breaking the graph structure 
            self.cached_k = k.detach()
            self.cached_v = v.detach()


        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        k = k.transpose(2,3)
        attn_scores = q @ k
        attn_scores /= (self.dheads**0.5)
        if training is True:
            if self.mask is None:
                curr_device = X.device
                self.mask = torch.triu(torch.ones(n,n,device = curr_device),diagonal=1).bool()
            attn_scores = torch.masked_fill(attn_scores,self.mask,-torch.inf)
        attn_weights = self.softmax(attn_scores)
        context_vector = attn_weights @ v

        context_vector = context_vector.view(batch_size,self.n_heads,n,self.dheads).transpose(1,2).reshape(batch_size,n,self.dout)

        return self.out_proj(context_vector)
    def reset_cache(self):
        self.cachedk = None
        self.cachedv = None