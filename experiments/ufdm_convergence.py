import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import itertools
from torch.nn.utils import weight_norm
import sys
from scipy.signal import medfilt as mf
import os
sys.path.insert(0, "../")

from ufdm import UFDM

device = "cuda:0"

# Insert path to the system
sys.path.insert(0, "../")

def sample_multivariate_gaussian(cov_matrix, n):
    """
    Sample `n` samples from a multivariate Gaussian distribution given a covariance matrix.
    
    Parameters:
    - cov_matrix (torch.Tensor): Covariance matrix of shape (d, d) where d is the dimension.
    - n (int): Number of samples to generate.
    
    Returns:
    - samples (torch.Tensor): Samples of shape (n, d).
    """
    d = cov_matrix.shape[0]  # Get the dimension from the covariance matrix
    mean = torch.zeros(d)     # Mean vector of zeros
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)  # Multivariate normal distribution
    samples = mvn.sample((n,))  # Sample `n` samples
    return samples

class LossGaussian(nn.Module):
    def __init__(self, cov_xx, cov_yy, cov_xy, device="cuda:0"):        
        super(LossGaussian, self).__init__()
        self.device = device
        self.cov_xx = cov_xx.to(self.device)
        self.cov_yy = cov_yy.to(self.device)
        self.cov_xy = cov_xy.to(self.device)

    def forward(self, alpha, beta):
        dimx = alpha.shape[0]
        dimy = beta.shape[0]

        #a = alpha + 0.00001*torch.randn(alpha.shape).to(self.device) #/ (dimx*torch.norm(alpha))
        #b = beta + 0.00001*torch.randn(beta.shape).to(self.device) # / (dimy*torch.norm(beta))

        a = alpha /  (dimx*torch.norm(alpha))
        b = beta / (dimy*torch.norm(beta))

        #print(f"{a} {b}")

        aSxxa = torch.matmul(torch.matmul(a.t(), self.cov_xx), a)
        bSxxb = torch.matmul(torch.matmul(b.t(), self.cov_yy), b)       
        aSxyb = torch.matmul(torch.matmul(a.t(), self.cov_xy), b)

        v1 = torch.exp(-0.5 * (aSxxa + bSxxb))
        v2 = torch.abs(torch.exp(-aSxyb) - 1.0)
        ret_val = v1 * v2

        return ret_val

def random_covariance_matrix(d):
    """
    Generate a random covariance matrix of dimension `d`.
    
    Parameters:
    - d (int): Dimension of the covariance matrix.
    
    Returns:
    - cov_matrix (torch.Tensor): A random covariance matrix of shape (d, d).
    """
    A = torch.randn(d, d)  # Generate a random matrix A of shape (d, d)
    cov_matrix = torch.mm(A, A.t())  # Create positive semi-definite matrix by multiplying A with its transpose
    return cov_matrix

def sample_random_batch(X, Y, batch_size):
    """
    Sample a random batch from datasets X and Y.
    
    Parameters:
    - X (torch.Tensor): Input dataset.
    - Y (torch.Tensor): Target dataset.
    - batch_size (int): Number of samples in the batch.
    
    Returns:
    - Batch from X and Y.
    """
    n = X.shape[0]  # Total number of rows in the arrays
    indices = np.random.choice(n, batch_size, replace=False)  # Randomly select `batch_size` indices
    return X[indices], Y[indices]  # Return the sampled batch for both X and Y

def covariance_XY(Sigma_X, Sigma_E, W):
    """
    Returns the covariance matrix of (X, Y).
    
    Parameters:
    - Sigma_X (numpy.ndarray): Covariance matrix of X.
    - Sigma_E (numpy.ndarray): Covariance matrix of the noise (ε).
    - W (numpy.ndarray): Projection matrix W.
    
    Returns:
    - Covariance matrices (torch.Tensor): Covariance matrices of (X, Y) and Σ_Y.
    """
    Sigma_X_W = np.dot(Sigma_X, W)  # Sigma_X * W^T
    WT_Sigma_X_W = np.dot(W.T, Sigma_X_W)
   
    top = np.concatenate((Sigma_X, Sigma_X_W), axis=1)  # Concatenate horizontally
    bottom = np.concatenate((Sigma_X_W.T, WT_Sigma_X_W + Sigma_E), axis=1)  # Concatenate horizontally
    
    Sigma_XY = np.concatenate((top, bottom), axis=0)  # Concatenate vertically
    
    return torch.from_numpy(Sigma_X_W), torch.from_numpy(WT_Sigma_X_W + Sigma_E)

if __name__ == "__main__":
    n_batch = 128
    n = n_batch * 100
    dim_x = 32
    dim_y = 32
    num_iter = 1000 
    input_proj_dim = 0 
    lr = 0.01

    model = UFDM(dim_x, dim_y, lr=lr, input_projection_dim=input_proj_dim, weight_decay=0.000, device=device)

    cov_x = random_covariance_matrix(dim_x)
    cov_e = random_covariance_matrix(dim_y)

    X = sample_multivariate_gaussian(cov_x, n)
    E = sample_multivariate_gaussian(cov_e, n)    
    W = torch.randn(dim_x, dim_y)

    Px = torch.matmul(X, W)
    Y = Px + E

    print(f"{X.shape} {Y.shape} {E.shape} {W.shape}")

    cov_xy, cov_y = covariance_XY(cov_x.detach().cpu().numpy(), cov_e.detach().cpu().numpy(), W.detach().cpu().numpy())

    print("Estimating UFDM with no distributional assumption")

    history_gradient_estimator = []
    for i in range(num_iter):
        x, y = sample_random_batch(X, Y, n_batch)
        dep = model(x, y, normalize=False)
        history_gradient_estimator.append(dep.cpu().detach().numpy())

    plt.plot(history_gradient_estimator)

    a = Variable(1.0 * torch.rand(dim_x, device=device) + 0.0, requires_grad=True).to(device)
    b = Variable(1.0 * torch.rand(dim_y, device=device) + 0.0, requires_grad=True).to(device)

    history_gaussian_estimator = []
    loss_gaussian = LossGaussian(cov_x, cov_y, cov_xy)

    optimizer = torch.optim.AdamW([a, b], lr=lr, weight_decay=0.000)

    print("Estimating UFDM with gaussian assumption")
    for i in range(num_iter):   
        loss = -loss_gaussian(a, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val = loss_gaussian(a, b)
        #print(f"{i} {a} {b} {val}")
        history_gaussian_estimator.append(val.detach().cpu().numpy())

    plt.plot(history_gaussian_estimator)
    plt.show()