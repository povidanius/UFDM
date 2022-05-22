import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "../")
from scipy.signal import medfilt as mf
from kac_independence_measure import KacIndependenceMeasure

"""
This is code to reproduce right part of Figure 1.
"""


def get_y(proj_x, epsilon, fun_type):
        if fun_type == 'trigonometric':
            print(fun_type)
            y  = (torch.sin(proj_x) + torch.cos(proj_x)) + 0.2*epsilon
        elif fun_type == 'logarithmic':            
            print(fun_type)
            y = torch.log(1.0 + 0.5*proj_x*proj_x) + 0.2*epsilon
        elif fun_type == 'linear':                        
            print(fun_type)
            y = proj_x + 0.2*epsilon
        elif fun_type == 'quadratic':            
            print(fun_type)
            y  = torch.mul(proj_x,proj_x) + 0.2*epsilon
        else:
            print("Error")
            sys.exit(0)
        return y

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_batch = 4096 
    dim_x = 512 
    dim_y = 512
    num_iter = 1000 
    input_proj_dim = 0 
    lr = 0.01

    if len(sys.argv) != 2:
        print("Usage {} fun_type".format(sys.argv[0]))
        sys.exit(0)
    else:
        fun_type = sys.argv[1]    

    # simulate inedependent data
    print("Independent")
    model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device=device)   
   
    history_indep = []
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        y = torch.randn(n_batch, dim_y)
        dep = model(x.to(device),y.to(device))
        history_indep.append(dep.clone().cpu().detach().numpy())
    plt.plot(history_indep)  

    # simulate dependent data (additive noise model)
    print("Dependent")
    model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device=device)    
    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y) # random projection
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y) # noise distribution
        y = get_y(proj_x, noise, fun_type)  # dependence
        dep = model(x.to(device),y.to(device))
        history_dep.append(dep.clone().cpu().detach().numpy())
    
    plt.plot(history_dep)  
    plt.savefig('./basic_demonstration/dependence_detection_{}_{}_{}_lr{}_nbatch{}.png'.format(fun_type, dim_x, dim_y, lr, n_batch))
    plt.savefig('./basic_demonstration/figures/1_1/dependence_detection_{}.png'.format(fun_type))

