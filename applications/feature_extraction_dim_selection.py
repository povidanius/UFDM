import torch
import numpy as np
import math
import sys
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
sys.path.insert(0, "../")
from ufdm import UFDM
from mi_feature_extraction import MiFeatureExtraction
from MI import *


random_state = 0

LR = 0.007 

def one_hot(x, num_classes=2):
  return np.squeeze(np.eye(num_classes)[x.reshape(-1)])

def load_data(db_name):

    X,y = fetch_openml(name=db_name, as_frame=True, return_X_y=True)

    print("X shape: {}".format(X.shape))
    dim_x = X.shape[1]
    categories = pd.unique(y.to_numpy().ravel())
    y_num = np.zeros(len(y))
    category_id = 0
    for cat in categories:
        ind = np.where(y == cat)
        for ii in ind:
            y_num[ii] = category_id
        category_id = category_id + 1

    y = y_num.astype(np.int32)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, random_state=None, shuffle=True, train_size=0.6)  
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, random_state=None, shuffle=True, train_size=0.8)  

    X_train = preprocessing.normalize(X_train)
    X_val = preprocessing.normalize(X_val)
    X_test = preprocessing.normalize(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[1], len(categories),X.shape[0]

def benchmark(X_train, y_train, X_test, y_test, method_name = 'R'):

    logistic = linear_model.LogisticRegression(max_iter=10000)
    result = logistic.fit(X_train, y_train).score(X_test, y_test)
    return result

if __name__ == "__main__":

    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')

    if len(sys.argv) != 2:
        print("Usage {} OpenML_dbname".format(sys.argv[0]))
        sys.exit(0)

    num_epochs = 100 #250  
    normalize = True


    X_train, y_train, X_val, y_val, X_test, y_test, dim_x, num_classes, num_samples = load_data(sys.argv[1])

    dim_y = num_classes 

    # now select feature dimension, which maximize validation accuracy

    #n_batch = 1024 
    n_batch = X_train.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zz = []
    zz_test = []

    for num_features in range(10, int(1.0*dim_x), max(1,int(0.1*dim_x))):

        kim = UFDM(dim_x, dim_y, lr=LR, input_projection_dim = num_features, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) 
        mim = MiFeatureExtraction(dim_x, dim_y, lr=LR, input_projection_dim = num_features, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) 
        hsic = MiFeatureExtraction(dim_x, dim_y, lr=LR, input_projection_dim = num_features, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) 
        dcor = MiFeatureExtraction(dim_x, dim_y, lr=LR, input_projection_dim = num_features, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) 

        n = X_train.shape[0]
        ytr = one_hot(y_train, num_classes)
        yte = one_hot(y_test, num_classes)
        dep_history = []
        for i in range(num_epochs):
            print("Epoch: {}, dim {}/{}".format(i, num_features, dim_x))
            shuffled_indices = np.arange(n)
            np.random.shuffle(shuffled_indices)
            num_batches = math.ceil(n/n_batch)
            for j in np.arange(num_batches):
                batch_indices = shuffled_indices[n_batch*j:n_batch*(j+1)]
                Xb = torch.from_numpy(X_train[batch_indices, :].astype(np.float32))
                yb = torch.from_numpy(ytr[batch_indices, :].astype(np.float32)) #.unsqueeze(1)
                #print("{} {}".format(Xb.shape, yb.shape))
                if i == 0 and j == 0:
                    kim.svd_initialise0(Xb,yb)
                dep = kim.forward(Xb, yb, normalize=normalize)
                dep_history.append(dep.detach().cpu().numpy())
                dep_mef = mim.forward(Xb, yb, normalize=normalize, measure='mi')
                dep_hsic = hsic.forward(Xb, yb, normalize=normalize, measure='hsic')
                dep_dcor = dcor.forward(Xb, yb, normalize=normalize, measure='dcor')

                #mivalue = mife.forward(Xb,yb, normalize=normalize)
                print("epoch {} batch {} dep {}, {}".format(i, j, dep, Xb.shape[0]))

    

        F_train = kim.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        F_val = kim.project(torch.from_numpy(X_val.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        F_test = kim.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

        FMI_train = mim.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        FMI_val = mim.project(torch.from_numpy(X_val.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        FMI_test = mim.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

        HSIC_train = hsic.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        HSIC_val = hsic.project(torch.from_numpy(X_val.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        HSIC_test = hsic.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

        DCOR_train = hsic.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        DCOR_val = hsic.project(torch.from_numpy(X_val.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
        DCOR_test = hsic.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

        num_features_nca = num_features
        max_num_features_nca = min(X_train.shape[0], X_train.shape[1])
        num_features_nca = min(num_features_nca, max_num_features_nca)

       

        num_features_nca = num_features
        nca = make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis(n_components=num_features_nca, random_state=random_state))

        nca.fit(X_train, y_train)
        NCA_X_train = nca.transform(X_train)
        NCA_X_val = nca.transform(X_val)
        NCA_X_test = nca.transform(X_test)


        rez_raw = benchmark(X_train, y_train, X_val, y_val, 'R')
        rez_kacIMFE = benchmark(F_train, y_train, F_val, y_val, 'KacIMF')
        rez_DCOR = benchmark(F_train, y_train, F_val, y_val, 'DCORMF')

        rez_MIMFE = benchmark(FMI_train, y_train, FMI_val, y_val, 'MIMF')
        rez_HSIC= benchmark(HSIC_train, y_train, HSIC_val, y_val, 'HSICF')
        rez_NCA = benchmark(NCA_X_train, y_train, NCA_X_val, y_val, 'NCA')

        zz.append([num_features, rez_raw, rez_kacIMFE, rez_DCOR, rez_MIMFE, rez_HSIC, rez_NCA])

        rez_raw_test = benchmark(X_train, y_train, X_test, y_test, 'R')
        rez_kacIMFE_test = benchmark(F_train, y_train, F_test, y_test, 'KacIMF')
        rez_DCOR_test = benchmark(F_train, y_train, F_test, y_test, 'DCORMF')

        rez_MIMFE_test = benchmark(FMI_train, y_train, FMI_test, y_test, 'KacIMF')
        rez_HSIC_test = benchmark(HSIC_train, y_train, HSIC_test, y_test, 'HSICF')
        rez_NCA_test = benchmark(NCA_X_train, y_train, NCA_X_test, y_test, 'NCA')

        zz_test.append([num_features, rez_raw_test, rez_kacIMFE_test, rez_DCOR_test, rez_MIMFE_test, rez_HSIC_test, rez_NCA_test])

    zz = np.array(zz)
    zz_test = np.array(zz_test)

    if not os.path.isfile('./feature_extraction/' + sys.argv[1] + '.npy'):
        np.save('./feature_extraction/' + sys.argv[1] + '.npy', zz_test)
    else:

        zzz = np.load('./feature_extraction/' + sys.argv[1] + '.npy')
        zzz = np.concatenate((zzz,zz_test), axis=0)    
        np.save('./feature_extraction/' + sys.argv[1] + '.npy', zzz)

    dim_kacimfe = int(zz[int(np.argmax(zz[:,2])),0])
    dim_dcor = int(zz[int(np.argmax(zz[:,3])),0])
    dim_mi = int(zz[int(np.argmax(zz[:,4])),0])
    dim_hsic = int(zz[int(np.argmax(zz[:,5])),0])
    dim_nca = int(zz[int(np.argmax(zz[:,6])),0])


    kim = UFDM(dim_x, dim_y, lr=LR, input_projection_dim = dim_kacimfe, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) #0.007
    mim = MiFeatureExtraction(dim_x, dim_y, lr=LR, input_projection_dim = dim_mi, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) #0.007
    hsic = MiFeatureExtraction(dim_x, dim_y, lr=LR, input_projection_dim = dim_hsic, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) #0.007
    dcor = MiFeatureExtraction(dim_x, dim_y, lr=LR, input_projection_dim = dim_dcor, weight_decay=0.01, orthogonality_enforcer=1.0, device=device) #0.007

    n = X_train.shape[0]
    ytr = one_hot(y_train, num_classes)
    yte = one_hot(y_test, num_classes)
    dep_history = []
    for i in range(num_epochs):
            shuffled_indices = np.arange(n)
            np.random.shuffle(shuffled_indices)
            num_batches = math.ceil(n/n_batch)
            for j in np.arange(num_batches):
                batch_indices = shuffled_indices[n_batch*j:n_batch*(j+1)]
                Xb = torch.from_numpy(X_train[batch_indices, :].astype(np.float32))
                yb = torch.from_numpy(ytr[batch_indices, :].astype(np.float32)) #.unsqueeze(1)
                if i == 0 and j == 0:
                    kim.svd_initialise0(Xb,yb)       
                dep = kim.forward(Xb, yb, normalize=normalize)
                dep_history.append(dep.detach().cpu().numpy())

                dep1 = mim.forward(Xb, yb, normalize=normalize, measure='mi')  
                dep2 = hsic.forward(Xb, yb, normalize=normalize, measure='hsic')      
                dep3 = dcor.forward(Xb, yb, normalize=normalize, measure='dcor')            

            print("epoch {} batch {} {}, {}".format(i, j, dep, Xb.shape[0]))
    

    F_train = kim.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
    F_test = kim.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

    DCOR_train = dcor.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
    DCOR_test = dcor.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

    FMI_train = mim.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
    FMI_test = mim.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

    HSIC_train = hsic.project(torch.from_numpy(X_train.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy()
    HSIC_test = hsic.project(torch.from_numpy(X_test.astype(np.float32)).to(device), normalize=normalize).detach().cpu().numpy() 

    num_features_nca = dim_nca
    max_num_features_nca = min(X_train.shape[0], X_train.shape[1])
    num_features_nca = min(num_features_nca, max_num_features_nca)
    nca = make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis(n_components=num_features_nca, random_state=random_state))
    nca.fit(X_train, y_train)
    NCA_X_train = nca.transform(X_train)
    NCA_X_test = nca.transform(X_test)


    rez_raw = benchmark(X_train, y_train, X_test, y_test, 'R')
    rez_kacIMFE = benchmark(F_train, y_train, F_test, y_test, 'KacIMF')
    rez_DCOR = benchmark(DCOR_train, y_train, DCOR_test, y_test, 'DCORF')
    rez_MIMFE = benchmark(FMI_train, y_train, FMI_test, y_test, 'MIMF')
    rez_HSIC = benchmark(HSIC_train, y_train, HSIC_test, y_test, 'HSICF')
    rez_NCA = benchmark(NCA_X_train, y_train, NCA_X_test, y_test, 'NCA')

    #breakpoint()
    
    #from os.path import exists
    handle_exception = True
    try:
        train = pd.read_csv('feature_extraction/{}.csv'.format(sys.argv[1]))
        train_tensor = torch.tensor(train.values)
        if train_tensor.shape[0] >= 25:
            print("Enough!")    
            handle_exception = False         
        else: 
            print("A")
            with open('feature_extraction/{}.csv'.format(sys.argv[1]),'a') as fd:
                print(sys.argv[1],end =" ")
                print(" & ",end =" ")
                print("({},{},{})".format(num_samples,X_train.shape[1],num_classes),end =" ")
                print(" & ",end =" ")
                #rez_raw = benchmark(X_train, y_train, X_test, y_test, 'R')
                print("%5.3f" % (rez_raw), end=" ")
                print(" & ",end =" ")
                print("%5.3f" % (rez_kacIMFE), end=" ")
                print(" & ",end =" ")
                print("%5.3f" % (rez_DCOR), end=" ")
                print(" & ",end =" ")                
                print("%5.3f" % (rez_MIMFE), end=" ") 
                print(" & ",end =" ")
                print("%5.3f" % (rez_HSIC), end=" ")                                
                #benchmark(PCA_X_train, y_train, PCA_X_test, y_test, 'PCA')
                #rez_NCA = benchmark(NCA_X_train, y_train, NCA_X_test, y_test, 'NCA')
                print(" & ",end =" ")
                print("%5.3f" % (rez_NCA), end=" ")
                print("\\\\",end ="\n")
                #print("num_classes {}".format(num_classes))

                result_row = [rez_raw, rez_kacIMFE, rez_DCOR, rez_MIMFE, rez_HSIC, rez_NCA] 
                writer = csv.writer(fd)
                writer.writerow(result_row)
    except:         
            if handle_exception == False:      
                sys.exit(0)
            else:
                with open('feature_extraction/{}.csv'.format(sys.argv[1]),'a') as fd:
                    print(sys.argv[1],end =" ")
                    print(" & ",end =" ")
                    print("({},{},{})".format(num_samples,X_train.shape[1],num_classes),end =" ")
                    print(" & ",end =" ")
                    #rez_raw = benchmark(X_train, y_train, X_test, y_test, 'R')
                    print("%5.3f" % (rez_raw), end=" ")
                    print(" & ",end =" ")
                    #rez_kacIMFE = benchmark(F_train, y_train, F_test, y_test, 'KacIMF')
                    print("%5.3f" % (rez_kacIMFE), end=" ")
                    #benchmark(PCA_X_train, y_train, PCA_X_test, y_test, 'PCA')
                    #rez_NCA = benchmark(NCA_X_train, y_train, NCA_X_test, y_test, 'NCA')
                    print(" & ",end =" ")
                    print("%5.3f" % (rez_DCOR), end=" ")
                    print(" & ",end =" ")                        
                    print("%5.3f" % (rez_MIMFE), end=" ") 
                    print(" & ",end =" ")
                    print("%5.3f" % (rez_HSIC), end=" ")                    
                    print(" & ",end =" ")
                    print("%5.3f" % (rez_NCA), end=" ")
                    print("\\\\",end ="\n")
                    #print("num_classes {}".format(num_classes))

                    result_row = [rez_raw, rez_kacIMFE, rez_DCOR, rez_MIMFE, rez_HSIC, rez_NCA] 
                    writer = csv.writer(fd)
                    writer.writerow(result_row)
