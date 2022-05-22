import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy import loadtxt
import numpy as np
from torch.distributions.laplace import Laplace

import sys
sys.path.insert(0, "../")

from kac_independence_measure import KacIndependenceMeasure
import os

def get_file_name():
    id = 0
    while os.path.isfile('./noise_effect/noise_effect_{}.txt'.format(id)):
        id = id + 1

    file_name = './noise_effect/noise_effect_{}.txt'.format(id)
    return file_name

def produce_plots():
    z = []
    for id in range(25):
            file_name = './noise_effect/noise_effect_{}.txt'.format(id)
            lines = loadtxt(file_name, comments="#", delimiter=",", unpack=False)
            z.append(lines)
            #breakpoint()
    z = np.array(z)     
    mean = np.mean(z,axis=0)       
    std = np.std(z, axis=0)
    plt.errorbar(0.1*np.array(range(0,30)), mean, yerr=std, fmt='-o')
    plt.savefig('./noise_effect/noise_effect_laplace.png')
    breakpoint()            


if __name__ == "__main__":

    if not os.path.exists('noise_effect'):
        os.makedirs('noise_effect')

    if len(sys.argv) == 2 and sys.argv[1] == 'plots':
        produce_plots();
        sys.exit(0)

    n_batch = 2048 
    dim_x = 512 
    dim_y = 4 
    num_iter = 600 
    input_proj_dim = 0


    file_object = open(get_file_name(), 'w')
    lap = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))

    to_plot = []
    step = 0

    random_proj = nn.Linear(dim_x, dim_y)

    noise_levels = np.arange(0.0, 3.0, 0.1)

    for noise_level in noise_levels:
        print("noise_level: {}".format(noise_level))
        history_dep = []
        #model.reset()

        model = KacIndependenceMeasure(dim_x, dim_y, lr=0.01,  input_projection_dim = input_proj_dim, weight_decay=0.001, device="cuda:0")

        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            proj_x = random_proj(x)
            #noise = 2*torch.rand(n_batch, dim_y)-1.0
            #noise = torch.exp(torch.randn(n_batch, dim_y)) # lognormal
            #noise = (noise - noise.mean(axis=0, keepdim=True))
            noise = lap.sample((n_batch, dim_y)).squeeze()
            y = torch.sin(proj_x) + torch.cos(proj_x)  + noise_level*noise           
            dep = model(x.to("cuda:0"),y.to("cuda:0"))
            history_dep.append(dep.cpu().detach().numpy())
            #print("{} {}".format(i, dep))
        to_plot.append(history_dep[-1])
        file_object.write(str(history_dep[-1]))    
        file_object.write('\n')
        #plt.plot(history_dep, label="Scale: ".format(str(noise_level)))
        #plt.savefig('./noise_effect/lognormal_{}.png'.format(step))
        step = step + 1
        plt.clf()

   
    #plt.plot(noise_levels, to_plot)    
    #plt.savefig('./noise_effect/summary_lognormal.png')
    #breakpoint()

    file_object.close()

