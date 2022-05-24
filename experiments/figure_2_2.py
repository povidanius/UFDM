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
import os
from scipy.ndimage import gaussian_filter1d


from kac_independence_measure import KacIndependenceMeasure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_y(proj_x, epsilon, fun_type):
        #print(fun_type)
        if fun_type == 'trigonometric':
            y  = (torch.sin(proj_x) + torch.cos(proj_x)) + 0.2*epsilon
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

    n_batch = 4096 #8192
    num_iter = 500 #250
    lr = 0.05 #0.05

    if len(sys.argv) != 2:
        print("Usage {} fun_type".format(sys.argv[0]))
        sys.exit(0)
    else:
        fun_type = sys.argv[1]   

    final_values_indep = []
    final_values_additive = []
    final_values_multiplicative = []
    xx = []

    for dim_x in range(10,512,30):
        print("dim_x = {}".format(dim_x))

        dim_y = dim_x
        fig = plt.figure()
        plt.figure().clear()

        xx.append(dim_x)

        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = 0, weight_decay=0.1, device=device)

        
    
        # inedependent data
        history_indep = []
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            y = torch.randn(n_batch, dim_y)
            dep = model(x,y)
            history_indep.append(dep.cpu().detach().numpy())


        final_values_indep.append(history_indep[-1])
        plt.plot(xx, final_values_indep, label="Independent")

        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = 0, weight_decay=0.1, init_scale_shift=[2.0,-1.0], device=device)        

        # dependent data (additive noise)
        history_dep = []
        random_proj = nn.Linear(dim_x, dim_y)
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x) 
            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y)
            #y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise    
            y = get_y(proj_x, noise, fun_type)
            dep = model(x,y, normalize=False)
            history_dep.append(dep.cpu().detach().numpy())
        


        final_values_additive.append(history_dep[-1])


        plt.plot(xx, final_values_additive, label="Dependent")

        #plt.savefig('./basic_demonstration/aaa_dependence_detection_kacim_by_dim.png')
        plt.savefig('./basic_demonstration/figures/2_2/kacim_vs_dim_{}.png'.format(fun_type))

        plt.close(fig)

        fig = plt.figure()        
        plt.figure().clear()
        plt.plot(xx, gaussian_filter1d(final_values_indep, sigma=1.5), label="Independent")
        plt.plot(xx, gaussian_filter1d(final_values_additive, sigma=1.5), label="Dependent")
        plt.savefig('./basic_demonstration/figures/2_2/smoothed_kacim_vs_dim_{}.png'.format(fun_type))
        plt.close(fig)

        np.save('./basic_demonstration/figures/2_2/kacim_vs_dim_{}_indep.npy'.format(fun_type),final_values_indep)
        np.save('./basic_demonstration/figures/2_2/kacim_vs_dim_{}_dep.npy'.format(fun_type),final_values_additive)
