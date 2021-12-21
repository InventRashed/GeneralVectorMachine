# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:40:03 2021

@author: RMUSLEM
"""


import tensorflow as tf
import random
import numpy as np
import pandas as pd
import keras as keras 
import keras.backend as K
from CGVM import train as CGVM
from CGVM import MetricsClass
from BALOfirst import bALO
import time




#%%
def CFGVM(N_C, N_F, Max_iter_C, Max_iter_F, X_train, y_train, X_test, y_test, X_all, y_all, paramsC):
    
    start = time.time()
    
    fitnessC, positionC, metricsC, posC, GC = bALO(N_C, Max_iter_C, X_train, y_train, X_test, y_test, paramsC, 'cost', 'CFGVM')
    
    Ctime = time.time()
    
    C1 = int("".join(str(int(x)) for x in positionC[0:4]),2)
    C2 = int("".join(str(int(x)) for x in positionC[4:11]),2)
    C = float(str(C1)+"."+str(C2))
    paramsC[3] = C
    
    fitness, position, metricsF, posF, GF= bALO(N_F, Max_iter_F, X_train, y_train, X_test, y_test, paramsC, 'feature', 'CFGVM')
    
    Ftime = time.time()
    
    x=[i>0.5 for i in position]
    # X_train = X_train[:, x]
    # X_test = X_test[:, x]
    
    # model = CGVM(X_train, y_train, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
    # y_pred = model(X_test)
    # weightsList.append(model.get_weights())
    # metrics = MetricsClass(y_test, y_pred) #put GMean in dictt
    # metricsList.append(metrics['Gmean'].numpy())
    
    Mtime = time.time()
    fitness = [fitnessC, fitness]
    position = [positionC, x]
    gmean = [GC, GF]
    metrics = [metricsC, metricsF]

    positions = [posC, posF]

    print('Time to run bALO for cost is:'+str((Ctime-start)/60)+' minutes' )
    print('Time to run bALO for feature is:'+str((Ftime-Ctime)/60)+' minutes')
    #print('Time to run CVGM is:'+str((Mtime-Ftime)/60)+' minutes')
    print('Total run time is:'+str((Mtime-start)/60)+' minutes')    
    return fitness, position, paramsC, metrics, positions, gmean


#%%
# def CFGVM(N_C, N_F, Max_iter_C, Max_iter_F, X_train, y_train, X_test, y_test, X_all, y_all, paramsC):
    
#     start = time.time()
    
#     fitnessC, positionC, metricsC, posC = bALO(N_C, Max_iter_C, X_train, y_train, X_test, y_test, paramsC, 'cost')
    
#     Ctime = time.time()
    
#     C1 = int("".join(str(int(x)) for x in positionC[0:4]),2)
#     C2 = int("".join(str(int(x)) for x in positionC[4:11]),2)
#     C = float(str(C1)+"."+str(C2))
#     paramsC[3] = C
    
#     fitness, position, metricsF, posF= bALO(N_F, Max_iter_F, X_train, y_train, X_test, y_test, paramsC, 'feature')
    
#     Ftime = time.time()
    
#     x=[i>0.5 for i in position]
#     # X_train = X_train[:, x]
#     # X_test = X_test[:, x]
    
#     # model = CGVM(X_train, y_train, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
#     # y_pred = model(X_test)
#     # weightsList.append(model.get_weights())
#     # metrics = MetricsClass(y_test, y_pred) #put GMean in dictt
#     # metricsList.append(metrics['Gmean'].numpy())
    
#     Mtime = time.time()
#     fitness = [fitnessC, fitness]
#     position = [positionC, x]

#     metrics = [metricsC, metricsF]

#     positions = [posC, posF]

#     print('Time to run bALO for cost is:'+str((Ctime-start)/60)+' minutes' )
#     print('Time to run bALO for feature is:'+str((Ftime-Ctime)/60)+' minutes')
#     #print('Time to run CVGM is:'+str((Mtime-Ftime)/60)+' minutes')
#     print('Total run time is:'+str((Mtime-start)/60)+' minutes')    
#     return fitness, position, paramsC, metrics, positions




#%%

# def CFGVM(N_C, N_F, Max_iter_C, Max_iter_F, X_train, y_train, X_test, y_test, params,):
    
#     start = time.time()
    
#     fitnessC, positionC, bestBetaC, bestWeightC, weightsListC, metricsC, featuresC, betaC = bALO(N_C, Max_iter_C, X_train, y_train, X_test, y_test, params, 'cost')
    
#     Ctime = time.time()
    
#     C1 = int("".join(str(int(x)) for x in positionC[0:4]),2)
#     C2 = int("".join(str(int(x)) for x in positionC[4:11]),2)
#     C = float(str(C1)+"."+str(C2))
#     params[3] = C
    
#     fitness, position, bestBeta, bestWeight, weightsList, metricsList, features, beta = bALO(N_F, Max_iter_F, X_train, y_train, X_test, y_test, params, 'feature')
    
#     Ftime = time.time()
    
#     x=[i>0.5 for i in position]
#     # X_train = X_train[:, x]
#     # X_test = X_test[:, x]
    
#     # model = CGVM(X_train, y_train, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
#     # y_pred = model(X_test)
#     # weightsList.append(model.get_weights())
#     # metrics = MetricsClass(y_test, y_pred) #put GMean in dictt
#     # metricsList.append(metrics['Gmean'].numpy())
    
#     Mtime = time.time()
#     fitness = [fitnessC, fitness]
#     position = [positionC, x]
#     weighttss = [weightsListC, weightsList]
#     metricss = [metricsC, metricsList]
#     features = [featuresC, features]
#     betas = [betaC, beta]
#     print('Time to run bALO for cost is:'+str((Ctime-start)/60)+' minutes' )
#     print('Time to run bALO for feature is:'+str((Ftime-Ctime)/60)+' minutes')
#     #print('Time to run CVGM is:'+str((Mtime-Ftime)/60)+' minutes')
#     print('Total run time is:'+str((Mtime-start)/60)+' minutes')    
#     return fitness, position, bestBeta, bestWeight, weighttss, metricss, features, params, betas


