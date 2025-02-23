import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np

def median_heuristic_bandwidth(X_data, Y_data):
    """
    Compute median of pairwise distances in R^2, treating each sample as (X_i, Y_i).
    Both X_data, Y_data are 1D Tensors or numpy arrays of same length n.
    
    Returns a float h_est.
    """
    # Convert to CPU numpy if needed
    x_np = X_data.detach().cpu().numpy() if isinstance(X_data, torch.Tensor) else np.array(X_data)
    y_np = Y_data.detach().cpu().numpy() if isinstance(Y_data, torch.Tensor) else np.array(Y_data)

    data_2d = np.column_stack((x_np, y_np))  # shape (n,2)
    # pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(data_2d, metric='euclidean')
    h_est = np.median(dists)
    return h_est

def gaussian_smoothing_factor(h, x):
    """
    Basic 1D Gaussian smoothing factor: exp(-0.5*h^2 * x^2)
    If x is multi-dimensional, x^2 => x^2 + ... etc.

    But in your usage, you'll apply 'x' = (xa + yb) or so.
    """
    # x can be a Torch tensor
    # we compute exp(-0.5*h^2 * x^2) elementwise
    return torch.exp(-0.5 * (h**2) * (x**2))

class UFDM_Smoothed(nn.Module):
    """
    A smoothed version of UFDM, using Gaussian smoothing with a bandwidth h.
    We pick h via median-heuristic or keep it as a parameter.
    """

    def __init__(
        self, 
        dim_x, 
        dim_y, 
        lr=0.005,  
        input_projection_dim=0, 
        output_projection_dim=0,
        weight_decay=0.01, 
        orthogonality_enforcer=1.0, 
        device="cpu", 
        init_scale_shift=[1,0],
        auto_bandwidth=True
    ):
        """
        auto_bandwidth : if True, we'll estimate h via median heuristic once.
        Otherwise, you can set self.h manually after initialization.
        """
        super(UFDM_Smoothed, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim
        self.output_projection_dim = output_projection_dim
        self.weight_decay = weight_decay
        self.orthogonality_enforcer = orthogonality_enforcer
        self.device = device
        self.init_scale_shift = init_scale_shift
        # We'll store h as a torch parameter or buffer
        # Let h be a parameter, or a buffer (requires_grad=False).
        # Here let's do "requires_grad=False" by default.
        self.h = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=False)
        self.auto_bandwidth = auto_bandwidth
        self.reset()

    def reset(self):
        param_list = []
        # param a
        if self.input_projection_dim > 0:
            self.a = Variable(
                self.init_scale_shift[0]*torch.rand(self.input_projection_dim, device=self.device) + self.init_scale_shift[1], 
                requires_grad=True
            )
            self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
            param_list += list(self.projection_x.parameters()) 
        else:
            self.a = Variable(
                self.init_scale_shift[0]*torch.rand(self.dim_x, device=self.device) + self.init_scale_shift[1], 
                requires_grad=True
            )

        # param b
        if self.output_projection_dim > 0:
            self.b = Variable(
                self.init_scale_shift[0]*torch.rand(self.output_projection_dim, device=self.device) + self.init_scale_shift[1],
                requires_grad=True
            )
            self.projection_y = nn.Linear(self.dim_y, self.output_projection_dim).to(self.device)
            param_list += list(self.projection_y.parameters()) 
        else:            
            self.b = Variable(
                self.init_scale_shift[0]*torch.rand(self.dim_y, device=self.device) + self.init_scale_shift[1], 
                requires_grad=True
            )
        
        self.bnx = nn.BatchNorm1d(self.dim_x, affine=True).to(self.device)
        self.bny = nn.BatchNorm1d(self.dim_y, affine=True).to(self.device)

        self.trainable_parameters = param_list + [self.a, self.b] + list(self.bnx.parameters()) + list(self.bny.parameters())
        
        # define optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

    def estimate_bandwidth_from_data(self, x_data, y_data):
        """
        Use median heuristic to set self.h
        This function can be called externally once you have the data loaded.
        """
        with torch.no_grad():
            h_est = median_heuristic_bandwidth(x_data, y_data)
            self.h[...] = h_est  # assign the float to self.h parameter

    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = self.bnx(x)
        proj = self.projection_x(x)            
        return proj

    def forward(self, x, y, update=True, normalize=True):
        """
        x,y: (batch_size, dim_x or dim_y)
        We do a single iteration step:
           1) Possibly BN,
           2) Possibly projection
           3) compute smoothed CF difference
           4) do gradient step if update==True
        returns the "ufdm" measure
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if normalize:
            x = self.bnx(x)
            y = self.bny(y)
        
        # Possibly reduce dimension
        if self.input_projection_dim > 0:
            x = self.projection_x(x)
        if self.output_projection_dim > 0:
            y = self.projection_y(y)

        dimx = self.a.shape[0]
        dimy = self.b.shape[0]
        xa = (x @ self.a) / (dimx*torch.norm(self.a))
        yb = (y @ self.b) / (dimy*torch.norm(self.b))

        # Smoothing factor: e^(-0.5 * h^2 * (xa+yb)^2) for joint
        # But let's break it down more carefully:
        
        # "joint" CF = E[exp(i*(xa + yb))]
        # smoothed: multiply by exp(-0.5*h^2*(xa+yb)^2)
        
        c_joint = torch.exp(1j*(xa + yb)) * torch.exp(-0.5*(self.h**2)*((xa+yb)**2))
        # average => empirical smoothed joint
        smoothed_joint = c_joint.mean()

        # For marginals:
        #   X-part:  E[exp(i*xa)*exp(-0.5*h^2*(xa)^2)]
        c_x = torch.exp(1j*xa) * torch.exp(-0.5*(self.h**2)*(xa**2))
        smoothed_x = c_x.mean()

        c_y = torch.exp(1j*yb) * torch.exp(-0.5*(self.h**2)*(yb**2))
        smoothed_y = c_y.mean()

        f_smooth = smoothed_joint - (smoothed_x * smoothed_y)
        ufdm_val = torch.norm(f_smooth)  # magnitude

        if update:
            loss = -ufdm_val
            # enforce orthonormal constraints if set
            if self.input_projection_dim > 0.0:
                # We want W_x^T * W_x ~ I
                penalty_x = torch.matmul(self.projection_x.weight, self.projection_x.weight.T)
                eye_x = torch.eye(self.input_projection_dim, device=self.device)
                loss = loss + self.orthogonality_enforcer*torch.norm(penalty_x - eye_x)

            if self.output_projection_dim > 0.0:
                penalty_y = torch.matmul(self.projection_y.weight, self.projection_y.weight.T)
                eye_y = torch.eye(self.output_projection_dim, device=self.device)
                loss = loss + self.orthogonality_enforcer*torch.norm(penalty_y - eye_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return ufdm_val
