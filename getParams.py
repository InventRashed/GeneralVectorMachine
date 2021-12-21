# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:02:25 2021

@author: RMUSLEM
"""

import tensorflow as tf
import random
import numpy as np
import pandas as pd
import keras as keras 
import keras.backend as K
# from CGVM import train
from CGVM import MetricsClass
from BALOfirst import bALO
from CFGVM import CFGVM
import sklearn.preprocessing as skpre
import sklearn.model_selection as skmodel
from completeCFGVM import train
from GVM import train as GVM


def getParams(X_train, y_train, X_test, y_test, ds, ranBs, nodess, R, params, typef = 'CGVM'):
    """

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    ds : TYPE
        DESCRIPTION.
    ranBs : TYPE
        DESCRIPTION.
    nodess : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.
    params : List
        0: input dimension, 1: beta, 2: batch size, 3: cost, 4: learning rate, 5: epochs

    Returns
    -------
    Dbest : TYPE
        DESCRIPTION.
    Nbest : TYPE
        DESCRIPTION.
    Bbest : TYPE
        DESCRIPTION.

    """
    metsD = []
    metsN = []
    metsB = []
    allD = []
    allN = []
    allB = []
    for i in ds:
        metsDD = np.zeros((R,1))
        for j in range(R):
            if typef=='CGVM':
                model = train(X_train, y_train, i, params[0], params[1],params[2], params[3], params[4], params[5],400, 1)
            else:
                model = GVM(X_train, y_train, i, params[0], params[1],params[2], params[3], params[4], params[5],400, 1)
            y_pred = model(X_test)
            mets = MetricsClass(y_test, y_pred)
            metsDD[j] = mets['Gmean']
            allD.append(mets)
        j = j+1
        metsD.append(np.mean(metsDD))
    Dbest = ds[np.argmax(metsD)]
    
    for i in nodess:
        metsNN = np.zeros((R,1))
        for j in range(R):
            b = np.random.uniform(-1,1,i)
            if typef=='CGVM':
                model = train(X_train, y_train, Dbest, params[0], b,params[2], params[3], params[4],params[5],i, 1)
            else:
                model = GVM(X_train, y_train, Dbest, params[0], b,params[2], params[3], params[4],params[5],i, 1)
            y_pred = model(X_test)
            mets = MetricsClass(y_test, y_pred)
            metsNN[j] = mets['Gmean']
            allN.append(mets)
        j = j+1
        metsN.append(np.mean(metsNN))
    Nbest = nodess[np.argmax(metsN)]
    
    for i in ranBs:
        metsBB = np.zeros((R,1))
        for j in range(R):
            b=np.random.uniform(-1,1,Nbest)
            if typef=='CGVM':
                model = train(X_train, y_train, Dbest, params[0], b, params[2], params[3], params[4], params[5], Nbest, i)
            else:    
                model = GVM(X_train, y_train, Dbest, params[0], b, params[2], params[3], params[4], params[5], Nbest, i)
            y_pred = model(X_test)
            mets = MetricsClass(y_test, y_pred)
            metsBB[j] = mets['Gmean']
            allB.append(mets)
        j = j+1
        metsB.append(np.mean(metsBB))
    Bbest = ranBs[np.argmax(metsB)]

    return Dbest, Nbest, Bbest, metsD, metsN, metsB, allD, allN, allB
        
        