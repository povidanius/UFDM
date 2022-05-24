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


def get_y(proj_x, epsilon, fun_type):
        print(fun_type)
        if fun_type == 'trigonometric':
            y  = (torch.sin(proj_x) + torch.cos(proj_x))  + 1.0*epsilon
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

if __name__ == "__main__":

    if not os.path.exists('basic_demonstration'):
        os.makedirs('basic_demonstration')

    if len(sys.argv) != 2:
        print("Usage {} fun_type".format(sys.argv[0]))
        sys.exit(0)
    else:
        fun_type = sys.argv[1]        

    n_batch = 4096 
    dim_x = 512 
    dim_y = 4

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
        dep_dep =  dcor.distance_correlation(x[:,:dim], y[:,:dim])
        dep_indep =  dcor.distance_correlation(x[:,:dim], z[:,:dim])
        #dep_dep =  dcor.u_distance_covariance_sqr(x[:,:dim], y[:,:dim])
        #dep_indep =  dcor.u_distance_covariance_sqr(x[:,:dim], z[:,:dim])
        #dep_dep =  dcor.u_distance_correlation_sqr(x[:,:dim], y[:,:dim])
        #dep_indep =  dcor.u_distance_correlation_sqr(x[:,:dim], z[:,:dim])
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
    plt.savefig('./basic_demonstration/figures/2_1/dependence_detection_udcorrelation_{}.png'.format(fun_type))



    #breakpoint()
