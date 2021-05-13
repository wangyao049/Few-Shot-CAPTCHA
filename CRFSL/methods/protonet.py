# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__(model_func,  n_way, n_support)
        self.var_weight = nn.Parameter(torch.randn([1,1], requires_grad=True))
        self.var_b = nn.Parameter(torch.randn([1], requires_grad=True))
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,is_feature = False):
#         print("ORIGINAL")
        z_support, z_query  = self.parse_feature(x,is_feature)
        # 把tensor变成在内存中连续分布的形式
        z_support   = z_support.contiguous()
        
        # the shape of z is [n_data, n_dim]
        # z_support [5,5,64]; z_query [5,15,64]
        
        # prototype & query point
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) 
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1)
        
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
    
    def set_forward_amend(self,x,is_feature = False):
#         print("set_forward_amend")
        z_support, z_query  = self.parse_feature(x,is_feature)
        # 把tensor变成在内存中连续分布的形式
        z_support   = z_support.contiguous()
        # prototype & query point
        # [5, 16, 64]
        z_proto     = z_support.view(self.n_way, self.n_support, -1).mean(1) 
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1)    
        
        dists = euclidean_dist(z_query, z_proto)
        
        emb_var = torch.std(z_query.view(self.n_way, self.n_support, -1),1)
        emb_var = emb_var.view(self.n_way,-1).mean(1)
#         print(emb_var)
        # 𝑣𝑘 = w ∗ 𝑣′𝑘 + 𝑏
        
        emb_var = emb_var*self.var_weight+self.var_b
        
        m = nn.Softmax(dim=0)
        emb_var = m(emb_var)
#         print(emb_var)
        emb_var = emb_var*(torch.tensor(self.n_way,dtype=torch.float))
        emb_var = emb_var.view(-1,self.n_way).repeat(self.n_way* self.n_query,1)
        amend_dists = torch.mul(emb_var,dists)
        scores = -amend_dists
        return scores
        

    def set_forward_loss(self, x):
#         print("set_forward_loss")
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
#         scores = self.set_forward_amend(x)
#         print(scores.shape)
#         print(y_query.shape)
        return self.loss_fn(scores, y_query)
        


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)





