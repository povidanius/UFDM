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

num_experiments = 25

def get_y(proj_x, epsilon, fun_type):
        if fun_type == 'trigonometric':
            #print(fun_type)
            y  = (torch.sin(proj_x) + torch.cos(proj_x)) + 0.1*epsilon
        elif fun_type == 'logarithmic':            
            #print(fun_type)
            y = torch.log(1.0 + 0.5*proj_x*proj_x) + 0.1*epsilon
        elif fun_type == 'linear':                        
            #print(fun_type)
            y = proj_x + 0.1*epsilon
        elif fun_type == 'quadratic':            
            #print(fun_type)
            y  = torch.mul(proj_x,proj_x) + 0.1*epsilon
        else:
            print("Error in get_y")
            sys.exit(0)
        return y

def get_file_name(fun_type):
    id = 0
    while os.path.isfile('./fig11_data_{}_{}.npy'.format(fun_type, id)):
        id = id + 1

    file_name = './fig11_data_{}_{}.npy'.format(fun_type, id)
    return file_name

def produce_plots(fun_type):
    z_indep = []
    z_dep = []
    for id in range(num_experiments):
            file_name = './fig11_data_{}_{}.npy'.format(fun_type, id)
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
    print(z_dep.shape)
    #breakpoint()
    mean_indep = np.mean(z_indep,axis=0)
    std_indep = np.std(z_indep, axis=0)
    mean_dep = np.mean(z_dep,axis=0)
    std_dep = np.std(z_dep, axis=0)    
    #breakpoint()
    f1 = plt.figure()   
    plt.errorbar(np.array(range(0,len(mean_indep))), mean_indep, yerr=std_indep, fmt='-.', errorevery=20)
    plt.errorbar(np.array(range(0,len(mean_dep))), mean_dep, yerr=std_dep, fmt='-.', errorevery=20)
    plt.savefig('./1_{}_error_bars.png'.format(fun_type))
    #plt.plot()


def produce_box_plots(fun_type):
    """
    This function loads ONLY the last valid experiment file for the given fun_type
    and produces a box plot comparing the distribution of the independent vs 
    dependent data from that single experiment.
    """
    last_id = -1
    # Find the highest ID for which the file exists
    for i in range(num_experiments):
        file_name = './fig11_data_{}_{}.npy'.format(fun_type, i)
        if os.path.isfile(file_name):
            last_id = i

    # If no files found, just return
    if last_id < 0:
        print("No experiment files found for fun_type =", fun_type)
        return
    
    # Load the data for the last experiment
    data = np.load('./fig11_data_{}_{}.npy'.format(fun_type, last_id))
    # data shape is (2, num_iterations)
    # data[0, :] -> history of 'independent' measure
    # data[1, :] -> history of 'dependent' measure

    plt.figure()
    plt.boxplot(
        [data[0, :], data[1, :]],
        labels=['Independent', 'Dependent']
    )
    plt.title(f"Box Plots for Last Experiment (ID={last_id}, {fun_type})")
    plt.savefig(f'./1_{fun_type}_box_plots_{last_id}.png')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_batch = 256 
    dim_x = 32 
    dim_y = 32
    num_iter = 1000 
    lr = 0.05

    #fun_type = 'trigonometric'

    
    if len(sys.argv) != 2:
        print("Usage {} fun_type".format(sys.argv[0]))
        sys.exit(0)
    else:
        fun_type = sys.argv[1]    
    

    os.system("rm *.npy")
    
    random_proj = nn.Linear(dim_x, dim_y) # random projection

    for e in range(num_experiments):
        print(e)
        # simulate inedependent data
        print("Independent")
        model = UFDM(dim_x, dim_y, lr=lr,  weight_decay=0.001, device=device, init_scale_shift=[1.0,0.0])    
    
        history_indep = []
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            y = torch.randn(n_batch, dim_y)
            dep = model(x.to(device),y.to(device))
            history_indep.append(dep.clone().cpu().detach().numpy())
        plt.plot(history_indep)  
        
        model.reset()

        # simulate dependent data (additive noise model)
        print("Dependent")
        history_dep = []
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
    

            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y) # noise distribution
            y = get_y(proj_x, noise, fun_type)  # dependence
            dep = model(x.to(device),y.to(device))
            history_dep.append(dep.clone().cpu().detach().numpy())
        plt.plot(history_dep)  
        #plt.show()
        #breakpoint()


        fname = get_file_name(fun_type)
        data_arr = np.array([np.array(history_indep), np.array(history_dep)])    
        np.save(fname, data_arr)
    produce_plots(fun_type)     
    produce_box_plots(fun_type)

    



