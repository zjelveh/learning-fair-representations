import pickle
import os
import numpy as np
import csv
import scipy.optimize as optim
from helper import *
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

# k = number of propotypes
k = 10
firstLine = True # assume csv has header column
with open('', 'rb') as f:
    filk = csv.reader(f)
    dat = []
    for row in filk:
        if firstLine:
            firstLine = False
            continue
        dat.append([float(r) for r in row])

print('finished reading data')
data = np.array(dat)
y = np.array(data[:,-1]).flatten()
data = data[:,:-1]
sensitive = data[:,-1]
data = preprocessing.scale(data)
data = data[:,:-1]
#data = data[:, :387]
sensitive_idx = np.array(np.where(sensitive==1))[0].flatten()
nonsensitive_idx = np.array(np.where(sensitive!=1))[0].flatten()
data_sensitive = data[sensitive_idx,:]
data_nonsensitive = data[nonsensitive_idx,:]
y_sensitive = y[sensitive_idx]
y_nonsensitive = y[nonsensitive_idx]

with open('', 'rb') as f:
  indices = pickle.load(f)

idx=indices[0]
training_sensitive = data_sensitive[idx,:]
ytrain_sensitive = y_sensitive[idx]
idx2=indices[1]
test_sensitive = data_sensitive[idx2,:]
ytest_sensitive = y_sensitive[idx2]
indices.append(idx)
indices.append(idx2)

idx=indices[2]
training_nonsensitive = data_nonsensitive[idx,:]
ytrain_nonsensitive = y_nonsensitive[idx]
idx2=indices[3]
test_nonsensitive = data_nonsensitive[idx2,:]
ytest_nonsensitive = y_nonsensitive[idx2]

#indices = []
#
#idx=np.array(list(set(np.random.randint(0, data_sensitive.shape[0], 3000))))
#training_sensitive = data_sensitive[idx,:]
#ytrain_sensitive = y_sensitive[idx]
#idx2=np.array([i for i in range(data_sensitive.shape[0]) if i not in idx])
#test_sensitive = data_sensitive[idx2,:]
#ytest_sensitive = y_sensitive[idx2]
#indices.append(idx)
#indices.append(idx2)
#
#idx=np.array(list(set(np.random.randint(0, data_nonsensitive.shape[0], 6000))))
#training_nonsensitive = data_nonsensitive[idx,:]
#ytrain_nonsensitive = y_nonsensitive[idx]
#idx2=np.array([i for i in range(data_nonsensitive.shape[0]) if i not in idx])
#test_nonsensitive = data_nonsensitive[idx2,:]
#ytest_nonsensitive = y_nonsensitive[idx2]
#indices.append(idx)
#indices.append(idx2)
# 
##with open('d:/dropbox/crime_lab_ny/fair_algorithms/data/indices_zemel.csv', 'wb') as f:
##  pickle.dump(indices, f)

training = np.concatenate((training_sensitive, training_nonsensitive))
ytrain = np.concatenate((ytrain_sensitive, ytrain_nonsensitive))

test = np.concatenate((test_sensitive, test_nonsensitive))
ytest = np.concatenate((ytest_sensitive, ytest_nonsensitive))




src= ''
if os.path.isfile(src):
    with open(src, 'rb') as f:
        rez = f.read().split('\n')[:-1]
    rez = np.array([float(r) for r in rez])
    print LFR(rez, training_sensitive, training_nonsensitive, ytrain_sensitive, 
              ytrain_nonsensitive, k, 1e-4, 0.1, 1000, 0)
else:
    print 'not loading'
    rez = np.random.uniform(size=data.shape[1] * 2 + k + data.shape[1] * k)

bnd = []
for i, k2 in enumerate(rez):
    if i < data.shape[1] * 2 or i >= data.shape[1] * 2 + k:
        bnd.append((None, None))
    else:
        bnd.append((0, 1))

rez = optim.fmin_l_bfgs_b(LFR, x0=rez, epsilon=1e-5, 
                          args=(training_sensitive, training_nonsensitive, 
                                ytrain_sensitive, ytrain_nonsensitive, k, 1e-4,
                                0.1, 1000, 0),
                          bounds = bnd, approx_grad=True, maxfun=150000, 
                          maxiter=150000)
