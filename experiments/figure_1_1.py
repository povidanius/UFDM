from locale import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "../")
from ufdm import *
import os

"""
This is code to reproduce right part of Figure 1.
"""

num_experiments = 20

def get_y(proj_x, epsilon, fun_type):
        if fun_type == 'independence':
            y = epsilon        
        elif fun_type == 'trigonometric':
            #print(fun_type)
            y  = torch.cos(proj_x)  + 0.1*epsilon
        elif fun_type == 'logarithmic':            
            #print(fun_type)
            y = torch.log(1.0 + proj_x*proj_x) + 0.1*epsilon
        elif fun_type == 'linear':                        
            #print(fun_type)
            y = proj_x + 0.1*epsilon
        elif fun_type == 'quadratic':            
            #print(fun_type)
            y  = torch.mul(proj_x,proj_x) + 0.1*epsilon
        else:
            print(f"Error in get_y {fun_type}")
            sys.exit(0)
        return y

def get_file_name(fun_type):
    id = 0
    while os.path.isfile('./fig11_data_{}_{}.npy'.format(fun_type, id)):
        id = id + 1

    file_name = './fig11_data_{}_{}.npy'.format(fun_type, id)
    return file_name

def produce_plots(fun_type):
    z = []
    for id in range(num_experiments):
            file_name = './fig11_data_{}_{}.npy'.format(fun_type, id)
            if os.path.isfile(file_name):
                data = np.array(np.load(file_name))
                z.append(data)

    
    z = np.array(z)        
    print(z.shape)
    #breakpoint()
    mean_dep = np.mean(z,axis=0)
    std_dep = np.std(z, axis=0)    
    #std_dep = np.quantile(z, q=0.9, axis=0)
    f1 = plt.figure()   
    plt.errorbar(np.array(range(0,len(mean_dep))), mean_dep, yerr=std_dep, fmt='-.', errorevery=20)
    plt.savefig('./1_{}_error_bars.png'.format(fun_type))
    #plt.box(mean_dep)
    #plt.savefig('./1_{}_box_plot.png'.format(fun_type))
    #plt.plot()
    #breakpoint()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_batch = 1024 
    dim_x = 32 
    dim_y = 32
    num_iter = 500 
    lr = 0.1


    if len(sys.argv) != 2:
            print("Usage {} fun_type".format(sys.argv[0]))
            sys.exit(0)
    else:
            fun_type = sys.argv[1]    
        
    if True:    

        os.system("rm *.npy")
        
        random_proj = nn.Linear(dim_x, dim_y) # random projection

        for e in range(num_experiments):
            print(e)

            model = UFDM(dim_x, dim_y, lr=lr,  weight_decay=0.00001, device=device, init_scale_shift=[1.0,1.0])    
            model.reset()

            # simulate dependent data (additive noise model)
            print(fun_type)
            history_dep = []
            for i in range(num_iter):
                x = torch.randn(n_batch, dim_x)
        
                #random_proj = nn.Linear(dim_x, dim_y) # random projection

                proj_x = random_proj(x)
                noise = torch.randn(n_batch, dim_y) # noise distribution
                y = get_y(proj_x, noise, fun_type)  # dependence
                dep = model(x.to(device),y.to(device), normalize=False)
                history_dep.append(dep.clone().cpu().detach().numpy())
            plt.plot(history_dep)  
            #plt.show()
            #breakpoint()


            fname = get_file_name(fun_type)
            np.save(fname, history_dep)
        produce_plots(fun_type)     

    #produce_box_plots(fun_type)

    



