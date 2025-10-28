# -*- coding: utf-8 -*-
# forked from https://github.com/SJYuCNEL/Matrix-based-Dependence/blob/main/MI.py

"""
Created on Sun Dec 13 21:03:51 2020

@author: Shujian Yu, Xi Yu

Calculate Mutual inforamtion based on Matrix-based Entropy Functional
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 1.01
upper=True


def calculate_gram_mat(x, sigma):
    """Calculate gram matrix for variables x
        Args:
            x: random variable with two dimensional (N,d).
            sigma: kernel size of x (Gaussian kernel)
        Returns:
            Gram matrix (N,N)
    """
    x = x.view(x.shape[0], -1)
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    dist = -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    return torch.exp(-dist / (2*sigma*sigma))

def renyi_entropy(x, sigma, alpha, epsilon = 0.0):
    """Calculate entropy for single variables x (Eq.(9) in paper)
        Args:
            x: random variable with two dimensional (N,d).
            sigma: kernel size of x (Gaussian kernel)
            alpha: alpha value of Renyi entropy
        Returns:
            Renyi alpha entropy of x.
    """
    k = calculate_gram_mat(x, sigma) + epsilon * torch.eye(x.shape[0]).to(x.device)

    k = k / torch.trace(k) 
    try:
        eigv = torch.abs(torch.linalg.eigvalsh(k))
        eig_pow = eigv**alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    except torch._C._LinAlgError as err:      # ← catch the specific error
        print("eigvalsh failed:", err)   
        return 0.0
         	    
    return entropy

def joint_entropy(x, y, s_x, s_y, alpha, epsilon = 0.0):
    """Calculate joint entropy for random variable x and y (Eq.(10) in paper)
        Args:
            x: random variable with two dimensional (N,d).
            y: random variable with two dimensional (N,d).
            s_x: kernel size of x
            s_y: kernel size of y
            alpha: alpha value of Renyi entropy
        Returns:
            Joint entropy of x and y.
    """
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)

    k = torch.mul(x, y) + epsilon * torch.eye(x.shape[0]).to(x.device)
    k = k / torch.trace(k)
    try:
        eigv = torch.abs(torch.linalg.eigvalsh(k))
        eig_pow = eigv**alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    except torch._C._LinAlgError as err:      # ← catch the specific error
        print("eigvalsh failed:", err)   
        return 0.0
        
    return entropy

def calculate_MI(x, y, s_x, s_y, alpha, normalize=True, epsilon = 0.0):
    """Calculate Mutual information between random variables x and y
        Args:
            x: random variable with two dimensional (N,d).
            y: random variable with two dimensional (N,d).
            s_x: kernel size of x
            s_y: kernel size of y
            alpha: alpha value of Renyi entropy
            normalize: bool True or False, normalize value between (0,1)
        Returns:
            Mutual information between x and y (scalar)
    """
    Hx = renyi_entropy(x, sigma=s_x, alpha=alpha, epsilon=epsilon)
    Hy = renyi_entropy(y, sigma=s_y, alpha=alpha, epsilon=epsilon)
    Hxy = joint_entropy(x, y, s_x, s_y, alpha=alpha, epsilon=epsilon)
    if normalize:
        Ixy = Hx + Hy - Hxy
        Ixy = Ixy / torch.max(Hx, Hy) 
    else:
        Ixy = Hx + Hy - Hxy
    return Ixy

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /(2*sigma*sigma))

def get_sigma(x):
    pd = pairwise_distances(x)
    return torch.sqrt(0.5 * torch.median(pd) )

def HSIC(x, y, s_x, s_y):
    
    """ calculate HSIC from https://github.com/danielgreenfeld3/XIC"""
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC



if __name__ == "__main__":
  x = torch.randn(1024, 32).cuda()
  y = torch.randn(1024, 32).cuda()
  sx = get_sigma(x)
  sy = get_sigma(y)
  mi = calculate_MI(x,y, sx, sy, ALPHA, False)
  print(mi.detach().cpu().numpy())
  mi = calculate_MI(x,x, sx, sx, ALPHA, False)
  print(mi.detach().cpu().numpy())

  hsic = HSIC(x,y, sx, sy)
  print(hsic.detach().cpu().numpy())
  hsic = HSIC(x,x, sx, sx)
  print(hsic.detach().cpu().numpy())

