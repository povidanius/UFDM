import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np
from torch import Tensor, linalg as LA


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

        self.max_norm = 25.0

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
        self.optimizer = torch.optim.AdamW(self.trainable_parameters, lr=self.lr, weight_decay=self.weight_decay) 

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def svd_initialise0(self, X: Tensor, Y: Tensor) -> None:
        """Initialise a, b with top singular vectors of Cov[X,Y]."""
        C = X.T @ Y / X.shape[0]
        U, _, Vh = LA.svd(C, full_matrices=False)
        self.a.copy_(U[:, 0].real[: self.a.numel()])
        self.b.copy_(Vh[0].real[: self.b.numel()])

    @torch.no_grad()
    def svd_initialise(self, X: Tensor, Y: Tensor) -> None:
        """Initialise a, b with top canonical correlation directions."""
        epsilon = 1e-6  # Small constant to avoid division by zero
        # Compute standard deviations for each dimension
        std_X = X.std(dim=0)
        std_Y = Y.std(dim=0)
        # Inverse square root of standard deviations for normalization
        D_X_inv_sqrt = 1.0 / (std_X + epsilon)
        D_Y_inv_sqrt = 1.0 / (std_Y + epsilon)
        # Normalize X and Y
        X_norm = X * D_X_inv_sqrt
        Y_norm = Y * D_Y_inv_sqrt
        # Compute normalized cross-covariance matrix
        C_norm = X_norm.T @ Y_norm / X.shape[0]
        # Perform SVD on the normalized matrix
        U, _, Vh = LA.svd(C_norm, full_matrices=False)
        # Adjust the singular vectors to the original space
        a_init = D_X_inv_sqrt * U[:, 0]
        b_init = D_Y_inv_sqrt * Vh[0, :]
        # Copy to parameters a and b
        self.a.copy_(a_init.real[: self.a.numel()])
        self.b.copy_(b_init.real[: self.b.numel()])
        #print(f'{self.a}, {self.b}')

    @torch.no_grad()
    def bootstrap_svd_init(self, X, Y, sample_frac=0.5, seed=None):
        #print("bootstrap")
        rng = torch.Generator() #device=X.device)
        if seed is not None: rng.manual_seed(seed)

        n = X.shape[0]
        idx = torch.randperm(n)[: int(sample_frac*n)]
        C = X[idx].T @ Y[idx] / idx.numel()

    
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)
        self.a.copy_(U[:, 0].real[: self.a.numel()])
        self.b.copy_(Vh[0].real[: self.b.numel()])

    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = self.bnx(x)

        proj = self.projection_x(x)            
        return proj

    def set_params(self, a,b):
        self.a = a
        self.b = b


    def clamp_params(self):
        """Clamp [a, b] to max_norm ball."""
        if self.max_norm is None:
            return
        gamma = torch.cat([self.a, self.b])
        norm_gamma = LA.norm(gamma)
        if norm_gamma > self.max_norm:
            gamma = gamma / norm_gamma * self.max_norm
        # Split back with data
        self.a.data.copy_(gamma[:self.a.numel()])
        self.b.data.copy_(gamma[self.a.numel():])

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

        
        xa = (x @ self.a)
        yb = (y @ self.b)
     
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

            # Clamp after step
            self.clamp_params()

        return ufdm
        