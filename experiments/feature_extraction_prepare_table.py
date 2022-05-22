import pandas as pd
import glob
from numpy import genfromtxt
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
from sklearn.datasets import fetch_openml


#path = r'./feature_extraciton' # use your path
all_files = glob.glob('./feature_extraction/*.csv')

li = []

for filename in all_files:
    #print(filename)
    x = genfromtxt(filename, delimiter=',')
    #print(x.shape)
    acc_avg = np.mean(x,axis=0)
    #print("{} {}".format(filename, acc_avg))
    #breakpoint()
    max_ind = np.argmax(acc_avg)
    
    #print(acc_avg[max_ind])
    p_vals = []
    for ind in range(3):
        if ind != max_ind:
            w, p = wilcoxon(x[:,max_ind], x[:, ind], alternative="greater")
            p_vals.append(p)
    significant = False            
    if np.max(p_vals)  < 0.01:
        significant = True

    db_name = Path(filename).stem
    print(db_name, end = " ")

    X,y = fetch_openml(name=db_name, as_frame=True, return_X_y=True, version=1)
    categories = pd.unique(y.to_numpy().ravel())
    print("& (%d,%d,%d) " % (X.shape[0], X.shape[1], len(categories)), end = " ")
    for ind in range(3):
        print(" & ", end = " ")
        if significant and ind == max_ind:
            print(r'\textbf{%2.4f}' % acc_avg[ind], end = " ")
        else:    
            print(r'%2.4f' % (acc_avg[ind]), end = " ")
    
    #print(np.max(p_vals))
    print("\\\\")
    #df = pd.read_csv(filename, index_col=None, header=0)
    #li.append(df)
    #print(df.shape)
    #x = df[1:]
    #print(x)

#frame = pd.concat(li, axis=0, ignore_index=True)