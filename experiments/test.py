from numpy import loadtxt
import numpy as np
from scipy.stats import wilcoxon


#acc_reg = loadtxt("./7result_True_0.1.txt", comments="#", delimiter=",", unpack=False)


#acc_reg = loadtxt("./8result_kb_True_0.1.txt", comments="#", delimiter=",", unpack=False)
#acc = loadtxt("./8result_kb_False_0.1.txt", comments="#", delimiter=",", unpack=False)

#acc_reg = loadtxt("./10result_kb_True_0.1.txt", comments="#", delimiter=",", unpack=False)
#acc = loadtxt("./10result_kb_False_0.1.txt", comments="#", delimiter=",", unpack=False)

#acc_reg = loadtxt("./14result_chest_True_0.15.txt", comments="#", delimiter=",", unpack=False)


#acc_reg = loadtxt("./18result_chest_True_9.0.txt", comments="#", delimiter=",", unpack=False)
#acc = loadtxt("./18result_chest_False_9.0.txt", comments="#", delimiter=",", unpack=False)

#acc_reg = loadtxt("./21aresult_chest_True_0.2.txt", comments="#", delimiter=",", unpack=False) # ok
#acc = loadtxt("./backup/21result_chest_False_0.2.txt", comments="#", delimiter=",", unpack=False)


#acc_reg = loadtxt("aaa_22result_chest_True_0.15.txt", comments="#", delimiter=",", unpack=False)
#acc = loadtxt("aaa_22result_chest_False_0.15.txt", comments="#", delimiter=",", unpack=False)

acc_reg =  loadtxt("./qmelanoma2_True_0.2.txt", comments="#", delimiter=",", unpack=False)
acc = loadtxt("./qmelanoma2_False_0.2.txt", comments="#", delimiter=",", unpack=False)
print("{} {}".format(acc_reg.shape[0], acc.shape[0]))
mean1 = 100*np.mean(acc_reg, axis=0)
mean0 = 100*np.mean(acc, axis=0)
#print("Reg {}, nonreg {}".format(mean1, mean0))
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(mean1)
print(mean0)
#breakpoint()
ww = []
pp = []
for i in range(6):
    #print(i)
    nn = np.min([len(acc_reg), len(acc)])
    w, p = wilcoxon(acc_reg[:nn,i], acc[:nn,i], alternative="greater")
    ww.append(p < 0.04)
    pp.append(p)
print(np.array(ww))
print(np.array(pp))

ww = []
pp = []
for i in range(6):
    #print(i)
    nn = np.min([len(acc_reg), len(acc)])
    w, p = wilcoxon(acc_reg[:nn,i], acc[:nn,i], alternative="less")
    ww.append(p < 0.04)
    pp.append(p)    
print(np.array(ww))
print(np.array(pp))

