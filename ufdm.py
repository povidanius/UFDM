import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np


class UFDM(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005,  input_projection_dim = 0, output_projection_dim=0, weight_decay=0.01, orthogonality_enforcer = 1.0, device="cpu", init_scale_shift=[1,0]):
        super(UFDM, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim
        self.output_projection_dim = output_projection_dim
        self.weight_decay = weight_decay
        self.orthogonality_enforcer = orthogonality_enforcer
        self.device = device
        self.init_scale_shift = init_scale_shift
        self.tanh =  torch.nn.Tanh()
        self.reset()

    def reset(self):


        param_list = []
        if self.input_projection_dim > 0:
            self.a = Variable(self.init_scale_shift[0]*torch.rand(self.input_projection_dim,device=self.device)+self.init_scale_shift[1], requires_grad=True)
            self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_x.parameters()) 
        else:
            self.a = Variable(self.init_scale_shift[0]*torch.rand(self.dim_x, device=self.device)+self.init_scale_shift[1], requires_grad=True)

        if self.output_projection_dim > 0:
            self.b = Variable(self.init_scale_shift[0]*torch.rand(self.output_projection_dim,device=self.device)+self.init_scale_shift[1], requires_grad=True)
            self.projection_y = nn.Linear(self.dim_y, self.output_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_y.parameters()) 
        else:            
            self.b = Variable(self.init_scale_shift[0]*torch.rand(self.dim_y,device=self.device)+self.init_scale_shift[1], requires_grad=True)
        
        self.bnx = nn.BatchNorm1d(self.dim_x, affine=True).to(self.device)
        self.bny = nn.BatchNorm1d(self.dim_y, affine=True).to(self.device)


        self.trainable_parameters = param_list  + [self.a, self.b] + list(self.bnx.parameters()) + list(self.bny.parameters())
        
        self.optimizer = torch.optim.AdamW(param_list  + [self.a, self.b] + list(self.bnx.parameters()) + list(self.bny.parameters()), lr=self.lr, weight_decay=self.weight_decay) 


    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = self.bnx(x)

        proj = self.projection_x(x)            
        return proj


    def forward(self, x, y, update = True, normalize=True):
        x = x.to(self.device)
        y = y.to(self.device)
        if normalize:
            x = self.bnx(x)
            y = self.bny(y)
        

        if self.input_projection_dim > 0:                
                x = self.projection_x(x)
        if self.output_projection_dim > 0:
                y = self.projection_y(y)


        
       
        dimx = self.a.shape[0]
        dimy = self.b.shape[0]
        xa = x @ self.a /  (dimx*torch.norm(self.a))
        yb = y @ self.b/ (dimy*torch.norm(self.b))

        #xa = x @ (self.a   / dimx)
        #yb = y @ (self.b  / dimy)
        #xa = xa / torch.norm(xa)
        #yb = yb / torch.norm(yb)

     
        f1 = torch.exp(1j*(xa + yb)).mean() - torch.exp(1j*xa).mean() * torch.exp(1j*yb).mean()
        ufdm = torch.norm(f1)

        if update:
            loss = -ufdm 
            if self.input_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_x.weight,self.projection_x.weight.T) - torch.eye(self.input_projection_dim).to(self.device)) # maximise => negative
            
            if self.output_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_y.weight,self.projection_y.weight.T) - torch.eye(self.output_projection_dim).to(self.device)) # maximise => negative

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

        return ufdm
        