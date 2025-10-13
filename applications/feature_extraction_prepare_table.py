import pandas as pd
import glob
from numpy import genfromtxt
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
from sklearn.datasets import fetch_openml
import warnings
import scikit_posthocs as sp

warnings.filterwarnings("ignore")


NUM_BASELINES = 6
PVAL_THRESHOLD = 0.05
ALGOS = ['RAW','UFDM','DCOR','MEF','HSIC','NCA']
# path = r'./feature_extraciton' # use your path
all_files = glob.glob('./feature_extraction_cv2/*.csv')

li = []
win_counter = np.zeros((NUM_BASELINES, NUM_BASELINES))
loose_counter = np.zeros((NUM_BASELINES, NUM_BASELINES))

idx = 0
accuracies = []
for filename in all_files:
    # print(filename)
    x = genfromtxt(filename, delimiter=',')

    if x.shape[0] < 3 or len(x.shape) == 1:
        continue
    # print("{} {} {}".format(idx, filename, x.shape))
    idx += 1

    acc_avg = np.mean(x, axis=0)
    accuracies.append(acc_avg)

    # print("{} {}".format(filename, acc_avg))
    # breakpoint()
    max_ind = np.argmax(acc_avg)
    
    # print(acc_avg[max_ind])
    p_vals = []
    for ind in range(NUM_BASELINES):
        if ind != max_ind:
            w, p = wilcoxon(x[:, max_ind], x[:, ind], alternative="greater")
            p_vals.append(p)

        for ind1 in range(NUM_BASELINES):
            if ind1 != ind:
                w, p = wilcoxon(x[:, ind], x[:, ind1], alternative="greater")
                if p < PVAL_THRESHOLD:
                    win_counter[ind, ind1] += 1.0

                w, p = wilcoxon(x[:, ind], x[:, ind1], alternative="less")
                if p < PVAL_THRESHOLD:
                    loose_counter[ind, ind1] += 1.0

                    #if ind1 != 0 and ind != 0:
                    #    print("{} > {}".format(ALGOS[ind],ALGOS[ind1]))

    significant = False
    if np.max(p_vals) < PVAL_THRESHOLD:
        significant = True

    db_name = Path(filename).stem
    print(db_name, end=" ")
    
    X, y = fetch_openml(name=db_name, as_frame=True, return_X_y=True)  # , version=1)
    categories = pd.unique(y.to_numpy().ravel())
    print("& (%d,%d,%d) " % (X.shape[0], X.shape[1], len(categories)), end=" ")

    for ind in range(NUM_BASELINES):
        print(" & ", end=" ")
        if significant and ind == max_ind:
            print(r'\textbf{%2.3f}' % acc_avg[ind], end=" ")
        elif not significant and ind == max_ind:
            print(r'\underline{%2.3f}' % acc_avg[ind], end=" ")
        else:
            print(r'%2.3f' % (acc_avg[ind]), end=" ")
    

    # print(np.max(p_vals))
    print("\\\\")
    # df = pd.read_csv(filename, index_col=None, header=0)
    # li.append(df)
    # print(df.shape)
    # x = df[1:]
    # print(x)
accuracies = np.array(accuracies)    
print(win_counter[1:,1:])
win_counter = win_counter[1:, 1:]
wins = np.sum(win_counter, axis=1)
methods = ['UFDM', 'DCOR', 'MEF', 'HSIC', 'NCA']
for i, m in enumerate(methods):
    print("{} {}".format(m, wins[i]))
print('---')    
print(loose_counter[1:,1:])
loose_counter = loose_counter[1:, 1:]
looses = np.sum(loose_counter, axis=1)
for i, m in enumerate(methods):
    print("{} {}".format(m, looses[i]))
print(f'{idx} data sets.')    

cd = sp.posthoc_nemenyi_friedman(accuracies)
#breakpoint()

