import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "../")
import os
import dcor
from kac_independence_measure import KacIndependenceMeasure
from scipy.ndimage import gaussian_filter1d

unbiassed = 0
num_experiments = 20

def get_y(proj_x, epsilon, fun_type):
        print(fun_type)
        if fun_type == 'trigonometric':
            y  = (torch.sin(proj_x) + torch.cos(proj_x))  + 0.2*epsilon
        elif fun_type == 'logarithmic':            
            y = torch.log(1.0 + 0.5*proj_x*proj_x) + 0.2*epsilon
        elif fun_type == 'linear':                        
            y = proj_x + 0.2*epsilon
        elif fun_type == 'quadratic':            
            y  = torch.mul(proj_x,proj_x) + 0.2*epsilon
        else:
            print("Error")
            sys.exit(0)
        return y

def get_file_name(fun_type):
    id = 0
    while os.path.isfile('./basic_demonstration/fig21_data{}_{}_{}.npy'.format(unbiassed, fun_type, id)):
        id = id + 1

    file_name = './basic_demonstration/fig21_data{}_{}_{}.npy'.format(unbiassed,fun_type, id)
    return file_name

def produce_plots(fun_type):
    z_indep = []
    z_dep = []
    for id in range(num_experiments):
            file_name = './basic_demonstration/fig21_data{}_{}_{}.npy'.format(unbiassed,fun_type, id)
            if os.path.isfile(file_name):
                data = np.array(np.load(file_name))
                z_indep.append(data[0,:])
                z_dep.append(data[1,:])         
                #breakpoint()
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
    mean_indep = np.mean(z_indep,axis=0) #gaussian_filter1d(np.mean(z_indep,axis=0), sigma=2.0)       
    std_indep = np.std(z_indep, axis=0)
    mean_dep = np.mean(z_dep,axis=0) #gaussian_filter1d(np.mean(z_dep,axis=0), sigma=2.0)
    std_dep = np.std(z_dep, axis=0)    
    #breakpoint()
    f1 = plt.figure()   
    plt.errorbar(np.array(range(1,512,30)), mean_indep, yerr=std_indep, fmt='-.', errorevery=1)
    plt.errorbar(np.array(range(1,512,30)), mean_dep, yerr=std_dep, fmt='-.', errorevery=1)
    plt.savefig('./basic_demonstration/figures/2_1/dependence_detection_dcor{}_{}_error_bars.png'.format(unbiassed,fun_type))


if __name__ == "__main__":

    if not os.path.exists('basic_demonstration'):
        os.makedirs('basic_demonstration')

    if len(sys.argv) != 2:
        print("Usage {} fun_type".format(sys.argv[0]))
        sys.exit(0)
    else:
        fun_type = sys.argv[1]        

    #produce_plots(fun_type)        
    #sys.exit(0)

    n_batch = 4096 
    dim_x = 512 
    dim_y = 512

    x = torch.randn(n_batch, dim_x)
    random_proj = nn.Linear(dim_x, dim_y)
    proj_x = random_proj(x)
    noise = torch.randn(n_batch, dim_y)
    y = get_y(proj_x, noise, fun_type)    
    x = x.detach().numpy()
    y = y.detach().numpy()
    z = torch.randn(n_batch, dim_y).detach().numpy()

    deps_dep = []
    deps_indep = []
    for dim in range(1,dim_x,30):
        print(dim)
        if unbiassed == 0:
            dep_dep =  dcor.distance_correlation(x[:,:dim], y[:,:dim])
            dep_indep =  dcor.distance_correlation(x[:,:dim], z[:,:dim])
        else:            
            dep_dep =  dcor.u_distance_correlation_sqr(x[:,:dim], y[:,:dim])
            dep_indep =  dcor.u_distance_correlation_sqr(x[:,:dim], z[:,:dim])
        #breakpoint()

        deps_dep.append(dep_dep)
        deps_indep.append(dep_indep)
    
    xx = np.array(range(1,dim_x,30))
    fig = plt.figure()
    plt.figure().clear()
    plt.plot(xx,deps_dep, '-', color='orange', label='Dependent')        
    plt.plot(xx,deps_indep,'-', color='blue', label='Independent')
    plt.legend(['Dependent','Independent'])

    #plt.show()
    plt.savefig('./basic_demonstration/figures/2_1/dependence_detection_dcor{}_{}.png'.format(unbiassed, fun_type))

    fname = get_file_name(fun_type)
    data_arr = np.array([np.array(deps_indep), np.array(deps_dep)])    
    np.save(fname, data_arr)
    print(fname)
    produce_plots(fun_type)    


    #breakpoint()
