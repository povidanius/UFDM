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
import os
from scipy.ndimage import gaussian_filter1d

"""
This is code to reproduce right part of Figure 1.
"""

num_experiments = 25

def get_y(proj_x, epsilon, fun_type):
        if fun_type == 'trigonometric':
            #print(fun_type)
            y  = (torch.sin(proj_x) + torch.cos(proj_x)) + 0.2*epsilon
        elif fun_type == 'logarithmic':            
            #print(fun_type)
            y = torch.log(1.0 + 0.5*proj_x*proj_x) + 0.2*epsilon
        elif fun_type == 'linear':                        
            #print(fun_type)
            y = proj_x + 0.2*epsilon
        elif fun_type == 'quadratic':            
            #print(fun_type)
            y  = torch.mul(proj_x,proj_x) + 0.2*epsilon
        else:
            #print("Error")
            sys.exit(0)
        return y

def get_file_name(fun_type):
    id = 0
    while os.path.isfile('./basic_demonstration/fig11_data_{}_{}.npy'.format(fun_type, id)):
        id = id + 1

    file_name = './basic_demonstration/fig11_data_{}_{}.npy'.format(fun_type, id)
    return file_name

def produce_plots(fun_type):
    z_indep = []
    z_dep = []
    for id in range(num_experiments):
            file_name = './basic_demonstration/fig11_data_{}_{}.npy'.format(fun_type, id)
            if os.path.isfile(file_name):
                data = np.array(np.load(file_name))
                z_indep.append(data[0,:])
                z_dep.append(data[1,:])         
                if np.isnan(data[1,-1]):
                    print(file_name)
                    #breakpoint()

    
    z_indep = np.array(z_indep)            
    z_dep = np.array(z_dep)        
    if z_indep.shape[0] < 2:
        return
    #z_indep = z_indep[:,::10]    
    #z_dep = z_dep[:,::10]     
    print(z_dep.shape)
    #breakpoint()
    mean_indep = gaussian_filter1d(np.mean(z_indep,axis=0), sigma=2.0)       
    std_indep = np.std(z_indep, axis=0)
    mean_dep = gaussian_filter1d(np.mean(z_dep,axis=0), sigma=2.0)
    std_dep = np.std(z_dep, axis=0)    
    #breakpoint()
    f1 = plt.figure()   
    plt.errorbar(np.array(range(0,1000)), mean_indep, yerr=std_indep, fmt='-.', errorevery=20)
    plt.errorbar(np.array(range(0,1000)), mean_dep, yerr=std_dep, fmt='-.', errorevery=20)
    plt.savefig('./basic_demonstration/figures/1_1/dependence_detection_{}_error_bars.png'.format(fun_type))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_batch = 4096 
    dim_x = 512 
    dim_y = 512
    num_iter = 1000 
    input_proj_dim = 0 
    lr = 0.001 # 0.01



    if len(sys.argv) != 2:
        print("Usage {} fun_type".format(sys.argv[0]))
        sys.exit(0)
    else:
        fun_type = sys.argv[1]    

    #produce_plots(fun_type)        
    #sys.exit(0)

    for e in range(num_experiments):
        print(e)
        # simulate inedependent data
        print("Independent")
        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device=device, init_scale_shift=[-2.0,2.0])    
    
        history_indep = []
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            y = torch.randn(n_batch, dim_y)
            dep = model(x.to(device),y.to(device))
            history_indep.append(dep.clone().cpu().detach().numpy())
        plt.plot(history_indep)  

        # simulate dependent data (additive noise model)
        print("Dependent")
        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device=device, init_scale_shift=[-2.0,2.0])    
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

        fname = get_file_name(fun_type)
        data_arr = np.array([np.array(history_indep), np.array(history_dep)])    
        np.save(fname, data_arr)
        print(fname)
        produce_plots(fun_type)     

    



