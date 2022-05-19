import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools


class KacIndependenceMeasure(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005,  input_projection_dim = 0, output_projection_dim=0, weight_decay=0.01, orthogonality_enforcer = 1.0, device="cpu"):
        super(KacIndependenceMeasure, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim
        self.output_projection_dim = output_projection_dim
        self.weight_decay = weight_decay
        self.orthogonality_enforcer = orthogonality_enforcer
        self.device = device
        self.reset()

    def reset(self):


        param_list = []
        if self.input_projection_dim > 0:
            self.a = Variable(torch.rand(self.input_projection_dim,device=self.device), requires_grad=True)
            self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_x.parameters()) 
        else:
            self.a = Variable(torch.rand(self.dim_x, device=self.device), requires_grad=True)

        if self.output_projection_dim > 0:
            self.b = Variable(torch.rand(self.output_projection_dim,device=self.device), requires_grad=True)
            self.projection_y = nn.Linear(self.dim_y, self.output_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_y.parameters()) 
        else:            
            self.b = Variable(torch.rand(self.dim_y,device=self.device), requires_grad=True)
        
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

        xa = (x @ (self.a/torch.norm(self.a)))
        yb = (y @ (self.b/torch.norm(self.b)))

        f = torch.exp(1j*(xa + yb)).mean() - torch.exp(1j*xa).mean() * torch.exp(1j*yb).mean()
        kim = torch.norm(f)

        if update:
            loss = -kim 
            if self.input_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_x.weight,self.projection_x.weight.T) - torch.eye(self.input_projection_dim).to(self.device)) # maximise => negative
            
            if self.output_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_y.weight,self.projection_y.weight.T) - torch.eye(self.output_projection_dim).to(self.device)) # maximise => negative

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

        return kim