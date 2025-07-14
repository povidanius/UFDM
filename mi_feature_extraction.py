import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import sys
from MI import calculate_MI
import torch.nn.functional as F


class MiFeatureExtraction(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005,  input_projection_dim = 0, output_projection_dim=0, weight_decay=0.01, orthogonality_enforcer = 1.0, device="cpu", init_scale_shift=[1,0]):
        super(MiFeatureExtraction, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim
        self.output_projection_dim = output_projection_dim
        self.weight_decay = weight_decay
        self.orthogonality_enforcer = orthogonality_enforcer
        self.device = device
        self.init_scale_shift = init_scale_shift
        self.reset()

    def reset(self):


        param_list = []
        self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
        param_list = param_list + list(self.projection_x.parameters()) 
        
        self.bnx = nn.BatchNorm1d(self.dim_x, affine=True).to(self.device)
        self.bny = nn.BatchNorm1d(self.dim_y, affine=True).to(self.device)
        param_list +=  list(self.bnx.parameters()) + list(self.bny.parameters())

        self.optimizer = torch.optim.AdamW(param_list, lr=self.lr) 

    def pairwise_distances(self, x):
        #x should be two dimensional
        instances_norm = torch.sum(x**2,-1).reshape((-1,1))
        return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

    def GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ /sigma)


    def HSIC(self, x, y, s_x, s_y):
        """ calculate HSIC from https://github.com/danielgreenfeld3/XIC"""
        m,_ = x.shape #batch size
        K = self.GaussianKernelMatrix(x,s_x)
        L = self.GaussianKernelMatrix(y,s_y)
        H = torch.eye(m) - 1.0/m * torch.ones((m,m))
        H = H.float().cuda()
        HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        return HSIC


    def DCOR(self, x, y, exponent=1.0):
        """
        Calculate unbiased distance correlation between x and y.
        
        Args:
            x: Tensor of shape (n, d1), input data
            y: Tensor of shape (n, d2), input data
            exponent: float, power applied to Euclidean distances (default 1.0)
        
        Returns:
            Unbiased distance correlation (scalar)
        """
        
        # Input validation
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")
        if x.shape[0] < 4:
            raise ValueError("Need at least 4 samples for unbiased estimator")
        
        n = x.shape[0]
        
        # Compute pairwise Euclidean distances and apply exponent
        def pairwise_dist(z):
            return torch.cdist(z, z, p=2).pow(exponent)
        
        # Unbiased double-centering
        def u_center(D):
            row_means = D.mean(dim=1, keepdim=True)
            col_means = D.mean(dim=0, keepdim=True)
            grand_mean = D.mean()
            c = 1.0 / (n - 2)
            A = D - c * (row_means + col_means) + grand_mean / ((n - 1) * (n - 2))
            A.fill_diagonal_(0.0)
            return A
        
        # Compute distance covariance
        a = u_center(pairwise_dist(x))
        b = u_center(pairwise_dist(y))
        dcov2 = (a * b).sum() / (n * (n - 3))
        
        # Compute distance variances
        dvar_x = (a * a).sum() / (n * (n - 3))
        dvar_y = (b * b).sum() / (n * (n - 3))
        
        # Compute distance correlation
        denom = torch.sqrt(dvar_x * dvar_y)
        dcor = torch.zeros_like(dcov2) if denom == 0 else dcov2 / denom
        
        return dcor

    def get_sigma(self, x):
        pd = self.pairwise_distances(x)
        return torch.median(pd)

    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = self.bnx(x)

        proj = self.projection_x(x)            
        return proj


    def forward(self, x, y, update = True, normalize=True, measure='mi'):
        x = x.to(self.device)
        y = y.to(self.device)
        if normalize:
            x = self.bnx(x)
            y = self.bny(y)
        
              
        x = self.projection_x(x)
        sx = max(0.000001 , self.get_sigma(x).detach())
        sy = max(0.000001 , self.get_sigma(y).detach())

        #print("sx sy {} {}".format(sx, sy))
        if measure == 'mi':
            mi = calculate_MI(x,y, sx, sy, 1.01, normalize=False, epsilon = 1e-5)
        elif measure == 'hsic':
            mi = self.HSIC(x,y, sx, sy)
        elif measure == 'dcor':
            mi = self.DCOR(x,y)

        
        if update:
            loss = -mi 
            if self.input_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_x.weight,self.projection_x.weight.T) - torch.eye(self.input_projection_dim).to(self.device)) # maximise => negative
            
            if self.output_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_y.weight,self.projection_y.weight.T) - torch.eye(self.output_projection_dim).to(self.device)) # maximise => negative

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

        return mi


