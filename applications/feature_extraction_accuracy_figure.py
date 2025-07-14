import pandas as pd
import glob
from numpy import genfromtxt
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import os
from matplotlib.transforms import Affine2D

all_files = glob.glob('./feature_extraction/*.npy')

method_raw = 1
method_kacimfe = 2
method_mi = 3
method_hsic = 4
method_nca = 5

fig, ax = plt.subplots(nrows=7, ncols=3, sharex=False, constrained_layout=True)
i = 0
row = 0
col = 0

for filename in sorted(all_files):
    print(filename)
    #breakpoint()
    xx = genfromtxt(os.path.splitext(filename)[0] + '.csv', delimiter=',')

    if xx.shape[0] < 15 or len(xx.shape) == 1:
        print("Continue")
        continue

    print("row {} col {}".format(row, col))

    data = np.load(filename)
    #breakpoint()
    ind = np.where(data[:, 0] == 10)[0]
    num_experiments = len(ind)
    num_dims_to_check = np.diff(ind)[0]
    #print(data.shape / num_dims_to_check)

    if (num_experiments < 15):
        print("num_exp {}".format(num_experiments))
        continue
    z = []
    for i in range(num_experiments):
        z.append(data[i * num_dims_to_check:(i + 1) * num_dims_to_check, :])
    # z.shape - num_experiments, x axis, methods
    mn = np.mean(z, axis=0)
    std = np.std(z, axis=0)
    
    dbname = Path(filename).stem
    if dbname == 'one-hundred-plants-shape':
        dbname = 'ohps'

    if dbname == 'amazon-commerce-reviews':
        dbname = 'amazon-com-rev'
    if dbname == 'Internet-Advertisements':
        dbname = 'internet-advert'

    if dbname == 'Smartphone-Based_Recognition_of_Human_Activities':
        dbname = 'sbrha'

    trans1 = Affine2D().translate(-0.1, 0.0) + ax[row, col].transData
    trans2 = Affine2D().translate(0.1, 0.0) + ax[row, col].transData
    trans3 = Affine2D().translate(-0.2, 0.0) + ax[row, col].transData
    trans4 = Affine2D().translate(0.2, 0.0) + ax[row, col].transData

    ax[row, col].set_xticks([])
    ax[row, col].set_title(dbname.lower().title())
    ax[row, col].errorbar(mn[:, 0], mn[:, method_kacimfe], yerr=std[:, method_kacimfe], fmt='--.', color='b', transform=trans1, alpha=1.0)
    ax[row, col].errorbar(mn[:, 0], mn[:, method_mi], yerr=std[:, method_mi], fmt='--.', color='r', transform=trans2, alpha=1.0)
    ax[row, col].errorbar(mn[:, 0], mn[:, method_hsic], yerr=std[:, method_hsic], fmt='--.', color='tab:green', transform=trans3, alpha=1.0)
    ax[row, col].errorbar(mn[:, 0], mn[:, method_nca], yerr=std[:, method_nca], fmt='--.', color='tab:orange', transform=trans4, alpha=1.0)
               
    i = i + 1
    col = col + 1
    if col == 3:
        col = 0
        row = row + 1
 
ect = [0, 0.00, 0.98, 1.0]
fig.tight_layout()  # [0, 0.00, 0.98, 1.0])
plt.subplot_tool()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.92)
plt.show()
# plt.savefig('./feature_extraction/dimension_vs_accuracy1.png')
