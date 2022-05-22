import pandas as pd
import glob
from numpy import genfromtxt
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt



all_files = glob.glob('./feature_extraction/*.npy')
method_raw = 1
method_kacimfe = 2
method_nca = 3

fig, ax = plt.subplots(nrows=2, ncols=4, sharex=False)
i = 0
row = 0
col = 0

for filename in all_files:
    print(filename)



    print("{} {}".format(row, col))

    data = np.load(filename)
    ind = np.where(data[:,0] == 10)[0]
    num_experiments = len(ind)
    num_dims_to_check = np.diff(ind)[0]
    print(data.shape / num_dims_to_check)
    z = []
    for i in range(num_experiments):        
        z.append(data[i*num_dims_to_check:(i+1)*num_dims_to_check,:])
    #z.shape - num_experiments, x axis, methods
    mn = np.mean(z,axis=0)
    std = np.std(z,axis=0)
    
    dbname = Path(filename).stem
    if dbname == 'one-hundred-plants-shape':
        dbname = 'ohps'

    ax[row,col].set_title(dbname)
    ax[row,col].errorbar(mn[:,0], mn[:,method_kacimfe], yerr=std[:,method_kacimfe], fmt='--o',color='b')
    ax[row,col].errorbar(mn[:,0], mn[:,method_nca], yerr=std[:,method_nca], fmt='-o',color='tab:orange')
           
    
    i = i + 1
    col = col + 1
    if col == 4:
        col = 0
        row = 1
 

fig.tight_layout()    
#plt.show()    
plt.savefig('./feature_extraction/dimension_vs_accuracy.png')

