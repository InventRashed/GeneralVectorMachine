# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:17:50 2021

@author: RMUSLEM
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:09:34 2021

@author: RMUSLEM

"""
import tensorflow as tf
from random import random
import numpy as np
import pandas as pd
import keras as keras 
import keras.backend as K
from CGVM import train 
from CGVM import MetricsClass
import copy
from sklearn.metrics import confusion_matrix


#%%
def Fitness(listG, listP,  cols, X_train, y_train, X_test, y_test, clf):
    #fitness function
    alpha=0.01
    # d = params[0]
    # input_dim = params[1]
    # batch_size = params[2]
    # C = params[3]
    # lr = params[4]
    # epochs = params[5]
    # nodes = params[6]
    # b = params[7]
    # ranB = params[8]
    # model_train = None
   

    if np.sum(cols)==0:
        y=100000
    else: 
        # if method=="feature":
        x=[i>0.5 for i in cols]
        listP.append(x)
        X_tr = X_train[:, x]
        X_t = X_test[:, x]
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_t)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        G = np.sqrt(np.divide((tp*tn), ((tp+fn)*(tn+fp))))
        # dictt['Gmean'] = tf.keras.backend.sqrt(tf.divide((dictt['TP']*dictt['TN']),((dictt['TP']+dictt['FN'])*(dictt['TN']+dictt['FP'])))).numpy()


        listG.append(G)
        y=(1-alpha)*(1-G)+alpha*np.sum(x)/(len(x))
        # elif method=="cost":
        #     C1 = int("".join(str(x) for x in cols[0:4]),2)
        #     C2 = int("".join(str(x) for x in cols[4:11]),2)
        #     C = float(str(C1)+"."+str(C2))
        #     if C<1:
        #         C=C+1
        #     model_train = train(X_train, y_train, d, input_dim, b, batch_size, C, lr, epochs, nodes, ranB)
            
        #     y_pred = model_train(X_test)
            
        #     listP.append(cols)
        #     model_train.save(r"C:\Users\rmuslem\Capgemini\Stage Rashed - General\models"+'\\'+'modelC'+str(len(listP)))

        #     metrics = MetricsClass(y_test, y_pred)

        #     listG.append(metrics)
        #     y = (1-metrics['Gmean'])
        # else:
        #     raise ValueError('Incorrect input for method, possible values: feature or cost')
        
    return y, listG, listP

def CrossOverU(RA,RE):
    out = np.zeros(RA.shape)
    r=(np.random.uniform(0,1,size=(1,len(RA)))<0.5).astype(int)[0]
    out[(r>0.5)]=RA[(r>0.5)]
    out[(r<0.5)]=RE[(r<0.5)]
    return out

def MutationU(dim,Max_iter, inp,Current_iter):
    r=np.random.uniform(0,1,size=(1,dim))
    r_new=(r>(Current_iter/Max_iter)).astype(int)
    RA=copy.deepcopy(inp) #check
    #RA(r)=1-RA(r);%set it to inverted value
    ran = (np.random.uniform(0,1,size=(1,np.sum(r_new)))>0.5).astype(int)[0]
    RA[(r_new>0.5)[0]]=ran#%set it to new values

    return RA


def RWS(weights):
#roulette wheel selection

  accumulation = np.cumsum(weights)
  p = np.random.uniform(0,1)*accumulation[len(accumulation)-1]
  chosen_index = -1
  for index in range(len(accumulation)):
    if accumulation[index] > p:
      chosen_index = index
      break

  return chosen_index

def initialization(SearchAgents_no,dim,ub=1,lb=0):

    Positions=np.random.uniform(lb, ub, size=(SearchAgents_no,dim))
    
    return (Positions >0.5).astype(int)

def bALO(N,Max_iter,X_train, y_train, X_test, y_test, classifier):
    """
    

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    Max_iter : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    GVMparams : Array
        consists of the parameters needed for the CGVM, in the following order [d, input_dim, batch_size, cost, learning rate, epochs, nodes]
    method : TYPE, optional
        DESCRIPTION. The default is 'feature'.

    Returns
    -------
    Elite_antlion_fitness : TYPE
        DESCRIPTION.
    Elite_antlion_position : TYPE
        DESCRIPTION.

    """
    #binary ALO inspired from the cALO
    #this code is based on the paper
    # E. Emary, Hossam M. Zawbaa, Aboul Ella Hassanien, "Binary ant lion approaches for feature selection", 
    # Neurocomputing, 2016. DOI: 10.1016/j.neucom.2016.03.101
    # params = [9, 9, 343, 4, 0.1, 300, 150]
    # N= 2
    dim = X_train.shape[1]
    
    # weight_antlions = [None]*N
    # weight_ants = [None]*N
    # weightList = []
    metricsList = []
    # modelsList = []
    posList = []

    # beta_antlions = [None]*N
    # beta_ants = [None]*N
    # elite_beta = []
    # elite_weight = []

    # N2 = 2*N
    # double_sorted_models = [None]*N2
    # Max_iter = 10
    # Current_iter=1    
    
    ant_position=initialization(N,dim,1,0)
    antlion_position=initialization(N,dim,1,0)
    
    # Initialize variables to save the position of elite, sorted antlions, 
    # convergence curve, antlions fitness, and ants fitness
    Sorted_antlions=np.zeros((N,dim))
    Elite_antlion_position=np.zeros((dim))
    Elite_antlion_fitness=0
    Convergence_curve=np.zeros((Max_iter))
    antlions_fitness=np.zeros((N))
    ants_fitness=np.zeros((N))
    
    # Calculate the fitness of initial antlions and sort them
    for i in range(N):
        antlions_fitness[i], metricsList, posList=Fitness(metricsList,  posList, antlion_position[i], X_train, y_train, X_test, y_test, classifier)
    
    sorted_antlion_fitness = np.sort(antlions_fitness)
    sorted_indexes = np.argsort(antlions_fitness)


    
    for newindex in range(N):
         Sorted_antlions[newindex]=antlion_position[sorted_indexes[newindex]]
         
    
        
    Elite_antlion_position=Sorted_antlions[0]
    Elite_antlion_fitness=sorted_antlion_fitness[0]


    
    Convergence_curve[0]=Elite_antlion_fitness
    # Main loop start from the second iteration since the first iteration 
    # was dedicated to calculating the fitness of antlions
    Current_iter=1; 
    while Current_iter<Max_iter:
        print('----------------------------------------------------')
        print('Current iteration is: ', Current_iter)
        print('----------------------------------------------------')
        # This for loop simulate random walks
        for i in range(len(ant_position)):
            # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
            weight = 1/sorted_antlion_fitness
            Rolette_index=RWS(weight)
            if Rolette_index==-1:
                Rolette_index=1
            
          
            # RA is the random walk around the selected antlion by rolette wheel
            #RA=RandWalk(dim,Max_iter,lb,ub, Sorted_antlions(Rolette_index,:),Current_iter);
            antlion_pos = Sorted_antlions[:][:]
            RA=MutationU(dim,Max_iter,antlion_pos[Rolette_index,:],Current_iter)
            # RE is the random walk around the elite (best antlion so far)
           # [RE]=RandWalk(dim,Max_iter,Elite_antlion_position(1,:),Current_iter);
            elite_pos = Elite_antlion_position.copy()
            RE=MutationU(dim,Max_iter,elite_pos,Current_iter)
            #ant_position(i,:)= (RA(Current_iter,:)+RE(Current_iter,:))/2; # Equation (2.13) in the paper   
            ant_position[i,:]= CrossOverU(RA,RE)
        
        
        for i in range(len(ant_position)):
            
            # Boundar checking (bring back the antlions of ants inside search
            # space if they go beyoud the boundaries
    #         Flag4ub=ant_position(i,:)>ub;
    #         Flag4lb=ant_position(i,:)<lb;
    #         ant_position(i,:)=(ant_position(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;  
    #         
            ants_fitness[i], metricsList, posList = Fitness(metricsList, posList, ant_position[i], X_train, y_train, X_test, y_test, classifier )  
            #All_fitness[0,i]=ants_fitness[0,i]
        
        
        # Update antlion positions and fitnesses based of the ants (if an ant 
        # becomes fitter than an antlion we assume it was cought by the antlion  
        # and the antlion update goes to its position to build the trap)
        double_population=np.concatenate((Sorted_antlions, ant_position))
        double_fitness=np.concatenate((sorted_antlion_fitness, ants_fitness))


           
        double_fitness_sorted=np.sort(double_fitness)
        I = np.argsort(double_fitness)
        double_sorted_population=double_population[I,:]

        # for i in range(2*N):
        #     double_sorted_models[i] = double_models[I[i]]    
        
        sorted_antlions_fitness=double_fitness_sorted[0:N]
        Sorted_antlions=double_sorted_population[0:N,:]

        # Update the position of elite if any antlinons becomes fitter than it
        if sorted_antlions_fitness[0]<Elite_antlion_fitness:
            Elite_antlion_position=Sorted_antlions[0]
            Elite_antlion_fitness=sorted_antlions_fitness[0]


    #         fprintf('\t');
        
        me=np.mean(sorted_antlions_fitness)
        den=np.max(abs(sorted_antlions_fitness-me))
        if den<1:
            den=1
        
        sig=np.sum(np.square((sorted_antlions_fitness-me)/den))/N
        print('bALO: ',Elite_antlion_fitness)
        print(Elite_antlion_position)
        print('\n')
        print(sig)
        # Keep the elite in the population
        Sorted_antlions[0,:]=Elite_antlion_position
        sorted_antlions_fitness[0]=Elite_antlion_fitness
 

      
        # Update the convergence curve
        Convergence_curve[Current_iter]=Elite_antlion_fitness    
        Current_iter=Current_iter+1;
    
    # fprintf('\n');
    return Elite_antlion_fitness,Elite_antlion_position, metricsList, posList
