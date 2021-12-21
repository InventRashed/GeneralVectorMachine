# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:08:51 2021

@author: RMUSLEM
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:01:27 2021

@author: RMUSLEM
"""

import tensorflow as tf
import random
import numpy as np
import pandas as pd
import keras as keras 
import keras.backend as K

#%%
def initializeOutputWeights(shape, dtype=None):
    """
    Function that initializes weights for the output which are between -1 and 1

    Parameters
    ----------
    shape : TYPE
        The shape that you want the weights to have, for example (9,1) 
    dtype : string, optional
        The output type the data needs to have. The default is None.

    Returns
    -------
    tensorflow variable
        with the output weights

    """
    #output weights are initialized as 1 or -1 and not changed afterwards
    randoms = np.random.randint(low=2, size=shape)
    new = np.where(randoms==0, -1, randoms)
    return tf.keras.backend.variable(new, dtype=dtype)

class CustomModel(tf.keras.Model):
   def __init__(self, b, input_dim, nodes):
       """
       Initialize a keras neural network model with custom weight initialization and custom activation function

       Parameters
       ----------
       b : Array
           An array with the values for the custom transfer coefficient 
       input_dim : Integer
           The number of input nodes for the model
       nodes : Integer
           The number of hidden nodes for the model

       Returns
       -------
       None.

       """
       super(CustomModel, self).__init__()
        # self.model=model

       initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
       self.dense = tf.keras.layers.Dense(nodes, name='hidden', kernel_initializer=initializer, 
                  bias_initializer=initializer, activation = lambda x: tf.tanh(b*x), input_shape=(input_dim,))
       self._output = tf.keras.layers.Dense(2, activation='linear', name='output', use_bias=False, trainable=False,kernel_initializer= lambda shape, 
                  dtype: initializeOutputWeights(shape, dtype))

   def call(self, inputs):
       """
       Calls the model and calculates the output

       Parameters
       ----------
       inputs : Keras dataset
           A keras dataset where the values will be used as inputs for the model to calculate the ouput

       Returns
       -------
       tensorflow tensor consisting of an array with floats which are the output estimations
           

       """
       # x = self.model(inputs)
       # return x
       x =  self.dense(inputs)
       return self._output(x)
    
#%%
def rewriteY(y_r, ones=0):
    """
    A function that rewrites the values of the 2D output y to a 2D output with only 1's and 0's 
    1 corresponding with the minority class and 0 with the majority class

    Parameters
    ----------
    y_r : Tensorflow tensor
        the output y from the model

    Returns
    -------
    y_new2 : Tensorflow tensor
        the new output with only 1's and 0's

    """
    #retrieve the indexes of the highest values in every row 
    #if the value in column 1 is higher than column 2 than the value is assigned to the positive class/minority
    #otherwise to the negative class
    indexesMax = tf.reshape(tf.cast(tf.argmax(y_r, axis=-1), "int32"),shape=(-1,1))
    indexesMin = tf.reshape(tf.cast(tf.argmin(y_r, axis=-1), "int32"),shape=(-1,1))
    
    #retrieve an array with indexes going from 1 till the number of observations 
    #and concatenate with the index of Max and Min
    rangeind = tf.reshape(tf.range(len(y_r)), shape=(-1,1))
    full_indexMax = tf.concat([rangeind, indexesMax], 1)
    full_indexMin = tf.concat([rangeind, indexesMin], 1)
    
    #change the highest value in every row to a 1 and the lowest value to a 0 
    y_new2 = tf.tensor_scatter_nd_update(tf.cast(y_r, dtype='float64'), full_indexMax, tf.cast(tf.ones(len(y_r)),dtype='float64'))
    if ones==0:
        y_new2 = tf.tensor_scatter_nd_update(tf.cast(y_new2, dtype='float64'), full_indexMin, tf.cast(tf.zeros(len(y_r)), dtype='float64'))
    else:
        y_new2 = tf.tensor_scatter_nd_update(tf.cast(y_new2, dtype='float64'), full_indexMin, tf.cast(-tf.ones(len(y_r)), dtype='float64'))
    
    return y_new2
    

def GVMLoss(d, y_pred, y_true):
    y_true = tf.cast(y_true, dtype='float32')
    N = y_true.shape[0]
    L = y_pred.shape[1]
    y_dot = y_pred*y_true
    y_d = y_dot-d
    y_square= y_d*y_d
    index_replace = y_dot>d
    idx_replace=tf.where(index_replace==True)
    y_loss = tf.tensor_scatter_nd_update(y_square, idx_replace, tf.zeros(idx_replace.shape[0]))
    return tf.divide(K.sum(K.sum(y_loss, axis=1)),tf.cast(N*L, tf.float32))
#%%
def MetricsClass(y_true, y_pred):
    """
    Calculate diffrent metrics for the predicted values
    consisting of: Accuracy, Precision, Recall, TP, TN, FP, FN
    returning them in a dictionary

    Parameters
    ----------
    y_true : numpy array
        The true values of the data
    y_pred : tensorflow tensor
        the predicted values from the model

    Returns
    -------
    dictt : dictionary
        dictionary with the metrics

    """
    #both the y's need to be in values of 1's and 0's otherwise keras can't calculate
    y_predict = rewriteY(y_pred)
    y_test = rewriteY(y_true)
    
    #initialize metrics
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    TP = tf.keras.metrics.TruePositives()
    TN = tf.keras.metrics.TrueNegatives()
    FP = tf.keras.metrics.FalsePositives()
    FN = tf.keras.metrics.FalseNegatives()
    Acc = tf.keras.metrics.Accuracy()
    dictt = {}
    metrics = [Acc, precision, recall, TP, TN, FP, FN]
    keys = ['Accuraat', 'Precision', 'Recall', 'TP', 'TN', 'FP', 'FN']
    
    #calculate metrics
    for i in range(len(metrics)):
        metrics[i].update_state(y_true = y_test[:,0], y_pred=y_predict[:,0])
        dictt[keys[i]] = metrics[i].result().numpy()
        metrics[i].reset_state()
    dictt['Gmean'] = tf.keras.backend.sqrt(tf.divide((dictt['TP']*dictt['TN']),((dictt['TP']+dictt['FN'])*(dictt['TN']+dictt['FP'])))).numpy()
    dictt['Fmeasure'] = tf.divide((2*dictt['Recall']*dictt['Precision']),(dictt['Recall']+dictt['Precision'])).numpy()
    return dictt
#%%
def new_weight(weights, step, boundaries, select):
    """
    Randomly retrieve a new weight for a specific index based on the step or random choices and random numbers
    within the boundaries of the weight that is selected

    Parameters
    ----------
    weights : Tensorflow tensor or numpy array
        the selected weights that need to be updated
    step : float
        the step for changing the weights
    boundaries : array
        the boundaries of the selected weight written in an array
    select : integer
        representing the selected weight, 0 for the hidden nodes weights, 1 for the bias and 2 for the beta coefficient

    Returns
    -------
    index : integer
        the index of the changed weight
    origin : float
        the original weight
    weights : tensorflow tensor or numpy array
        an array with the weights including the update

    """
    index=[]
    origin = 0
    #the hidden nodes weights is the only matrix thus needs two random indexes to be chosen
    if select == 0:
        ind1 = np.random.choice(weights.shape[0],1)[0] #row 
        ind2 = np.random.choice(weights.shape[1],1)[0] #column
        #combine the two to use as an index
        index = tuple([ind1,ind2])  
        #retrieve original
        origin = weights[index]
    else:
        #both beta and bias are vectors
        ind1 = np.random.choice(len(weights),1)
        index = ind1
        origin = weights[index][0]
    
    #change the original by the step
    new1 = origin + step
    new2 = origin - step
    
    #if both values are out of bounds select new random numbers within the boundaries
    if new1 < boundaries[0] or new1 > boundaries[1] and new2 < boundaries[0] or new2 > boundaries[1]:    
        new1 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
        new2 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
        # weights[index] = np.random.choice([new1[0], new2[0]])
    
    #if only the new1 is out of bounds only select new random number for this one
    elif new1 < boundaries[0] or new1 > boundaries[1]:
        new1 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
        # weights[index] = np.random.choice([new1[0], new2[0]])
    
    #if only the new2 is out of bounds only select new random number for this one
    elif new2 < boundaries[0] or new2 > boundaries[1]:
        new2 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
    
    #randomly choose the new value 
    weights[index] = np.random.choice([new1, new2])
    
    
    return index, origin, weights
#%%
#@tf.function
def train_step(epoch, b, batch, loss, weights, d, input_dim, lr, C, nodes, ranB):
    """
    The train step that is performed for every batch.
    Here the model is optimized/trained, the weights are being updated.
    Weights only get updated if they lower the loss

    Parameters
    ----------
    b : numpy array
        the beta transfer function coefficient
    batch : tensorflow batch data
        the current batch data used
    loss : float
        the most recent lowest loss value
    weights : tensorflow tensor with arrays
        the weights of the model.
    d : integer
        the steep seperating margin.
    input_dim : integer
        the input dimensions corresponding with the number of features/variables
    lr : float
        the learning rate used to calculate the step.
    C : float
        the costs associated with FN.
    nodes : integer
        the number of hidden nodes used in the model

    Returns
    -------
    loss_min : float
        the minimum loss value in this train step after optimizing
    b : array
        if selected the updated beta transfer function coefficient.
    weights : tensorflow tensor 
        if selected the updated weights
    lr_new : float
        the new learning rate used to calculate the step.

    """
    
    #retrieve the data from the batch
    x, y = batch
    
    #initialize parameters
    loss_min = loss
    lr = lr#*K.sqrt(loss_min)
    boundariesW = [-1,1]
    boundariesB = [-10,10]
    boundariesBeta = [-ranB,ranB]
    y_pred=[]
    lr_new=lr
    
    #randomly select beta, bias or weights
    select = np.random.randint(3)
    
    #if beta is selected calculate new weight
    if select == 2:
        ind,original, b = new_weight(b, lr, boundariesBeta, select)
        #create new model with the new weight
        model = CustomModel(b, input_dim, nodes)
        #model needs to be used once before we can update weights
        y_pred = model(x)
        model.set_weights(weights)
        #predict y to calculate loss
        y_pred = model(x)
        loss = GVMLoss(d, y_pred, y)
        #if loss is lower than minimum loss than select the weight is selected and minimum loss is updated
        if loss<loss_min:
            loss_min = loss
            #new learning rate based on the new minimum loss
            lr_new = lr*np.sqrt(loss_min)
        else:
            #else return the weight to normal
            b[ind] = original
    else:
        #for 0 en 1 we have different boundaries but both used in weights array thus the distinction here
        if select == 0:
            ind, original, weights[select] = new_weight(weights[select], lr, boundariesW, select)
        else:
            ind, original, weights[select] = new_weight(weights[select], lr, boundariesB, select)
        model = CustomModel(b, input_dim, nodes)
        y_pred = model(x)
        model.set_weights(weights)
        y_pred = model(x)
        loss = GVMLoss(d, y_pred, y)
        if loss<loss_min:
            loss_min = loss
            lr_new = lr*np.sqrt(loss_min)
        else:
            weights[select][ind]=original
        
    # with tf.GradientTape() as tape:
    #     x, y = batch
    #     d = 16
    #     y_pred = model(x)
    #     loss = GVMLoss(d, y_pred, y)

    # gradients = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # update your metrics here how you want.
    #diction = MetricsClass(y_true=y, y_pred=y_pred)
    #acc_metric.update_state(y, y_pred)
    
    #calculate metrics and print out loss and metrics
    dictt = MetricsClass(y, y_pred)
    # tf.print("Training loss (for one batch): ", loss_min)
    # tf.print("Training metrics (for one batch): ", dictt.values())
    if epoch==200 and dictt['Accuraat']<0.5:
        model = CustomModel(b, input_dim, nodes)
        y_pred = model(x)
        weights = model.get_weights()
        
    return loss_min, b, weights, lr_new, dictt
    

def train(X_train, y_train, d, input_dim, b=np.random.uniform(-1,1, size=150), batch_size=300, C=2.5, lr=0.01, epochs=300, nodes=150, ranB=1):
   """
    Trains the model by running train step for every epoch and every batch
    and returns the optimized model ready to be used

    Parameters
    ----------
    dataset : Tensorflow batched dataset
        Dataset that is split into batches from tensor slices
    d : integer
        the steep seperating margin.
    input_dim : integer
        the input dimensions corresponding to the number of variables
    C : float, optional
        Costs associated with FN. The default is 2.5.
    lr : float, optional
        Learning rate used to change the weights. The default is 0.01.
    epochs : integer, optional
        the number of time the model is run over the data. The default is 300.
    nodes : integer, optional
        the number of hidden nodes in the model. The default is 150.

    Returns
    -------
    custom_model : Tensorflow keras model
        The optimized model.

    """
   dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
   X_train.shape[0]).batch(batch_size)

    #initialize the beta coefficient
   b = b
   #initialize parameters and model to get weights
   lr = lr
   model = CustomModel(b,input_dim, nodes)
   x = list(dataset)[0][0]
   y = list(dataset)[0][1]
   y_p = model(x)
   loss = GVMLoss(d, y_p, y)
   weights = model.get_weights()
   
   #run the train step for every batch per epoch
   for epoch in range(epochs):
     tf.print(str(epoch)+'/'+str(epochs)+'-----------------------------')
     
     for batch in dataset:
         loss, b, weights, lr, metric = train_step(epoch, b, batch, loss, weights, d, input_dim, lr, C, nodes, ranB)
     tf.print("Training loss (for one epoch): ", loss)
     tf.print("Training metrics (for one epoch): ", metric.values())
    #use the weights to get the optimal model and set the weights
   custom_model = CustomModel(b,input_dim, nodes)
   custom_model(x)
   custom_model.set_weights(weights)
   
   return custom_model

     #train_acc = acc_metric.result()
     #tf.print("Training acc over epoch: %.4f" % (float(train_acc),))

     # Reset training metrics at the end of each epoch
     #acc_metric.reset_states()
     
     # -*- coding: utf-8 -*-
#%%
# def initializeOutputWeights(shape, dtype=None):
#     """
#     Function that initializes weights for the output which are between -1 and 1

#     Parameters
#     ----------
#     shape : TYPE
#         The shape that you want the weights to have, for example (9,1) 
#     dtype : string, optional
#         The output type the data needs to have. The default is None.

#     Returns
#     -------
#     tensorflow variable
#         with the output weights

#     """
#     #output weights are initialized as 1 or -1 and not changed afterwards
#     randoms = np.random.randint(low=2, size=shape)
#     new = np.where(randoms==0, -1, randoms)
#     return tf.keras.backend.variable(new, dtype=dtype)

# class CustomModel(tf.keras.Model):
#    def __init__(self, b, input_dim, nodes):
#        """
#        Initialize a keras neural network model with custom weight initialization and custom activation function

#        Parameters
#        ----------
#        b : Array
#            An array with the values for the custom transfer coefficient 
#        input_dim : Integer
#            The number of input nodes for the model
#        nodes : Integer
#            The number of hidden nodes for the model

#        Returns
#        -------
#        None.

#        """
#        super(CustomModel, self).__init__()
#         # self.model=model

#        initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
#        self.dense = tf.keras.layers.Dense(nodes, name='hidden', kernel_initializer=initializer, 
#                   bias_initializer=initializer, activation = lambda x: tf.tanh(b*x), input_shape=(input_dim,))
#        self._output = tf.keras.layers.Dense(2, activation='linear', name='output', use_bias=False, trainable=False,kernel_initializer= lambda shape, 
#                   dtype: initializeOutputWeights(shape, dtype))

#    def call(self, inputs):
#        """
#        Calls the model and calculates the output

#        Parameters
#        ----------
#        inputs : Keras dataset
#            A keras dataset where the values will be used as inputs for the model to calculate the ouput

#        Returns
#        -------
#        tensorflow tensor consisting of an array with floats which are the output estimations
           

#        """
#        # x = self.model(inputs)
#        # return x
#        x =  self.dense(inputs)
#        return self._output(x)
    
# #%%
# def rewriteY(y_r):
#     """
#     A function that rewrites the values of the 2D output y to a 2D output with only 1's and 0's 
#     1 corresponding with the minority class and 0 with the majority class

#     Parameters
#     ----------
#     y_r : Tensorflow tensor
#         the output y from the model

#     Returns
#     -------
#     y_new2 : Tensorflow tensor
#         the new output with only 1's and 0's

#     """
#     #retrieve the indexes of the highest values in every row 
#     #if the value in column 1 is higher than column 2 than the value is assigned to the positive class/minority
#     #otherwise to the negative class
#     indexesMax = tf.reshape(tf.cast(tf.argmax(y_r, axis=-1), "int32"),shape=(-1,1))
#     indexesMin = tf.reshape(tf.cast(tf.argmin(y_r, axis=-1), "int32"),shape=(-1,1))
    
#     #retrieve an array with indexes going from 1 till the number of observations 
#     #and concatenate with the index of Max and Min
#     rangeind = tf.reshape(tf.range(len(y_r)), shape=(-1,1))
#     full_indexMax = tf.concat([rangeind, indexesMax], 1)
#     full_indexMin = tf.concat([rangeind, indexesMin], 1)
    
#     #change the highest value in every row to a 1 and the lowest value to a 0 
#     y_new2 = tf.tensor_scatter_nd_update(tf.cast(y_r, dtype='float64'), full_indexMax, tf.cast(tf.ones(len(y_r)),dtype='float64'))
#     y_new2 = tf.tensor_scatter_nd_update(tf.cast(y_new2, dtype='float64'), full_indexMin, tf.cast(tf.zeros(len(y_r)), dtype='float64'))
    
#     return y_new2
    
# def GVMLoss(d, y_pred, y_true, C):
#     """
#     Cost-sensitive loss function used for training the model

#     Parameters
#     ----------
#     d : integer
#         the steep-margin to seperate the two types
#     y_pred : Tensorflow tensor
#         The predicted output by the model
#     y_true : numpy array
#         the true values of the data
#     C : float or integer
#         the costs associated with missclassifying the positive class, False Negatives

#     Returns
#     -------
#     float
#         the loss of the current prediction

#     """
#     #calculate the loss
#     # y_true = tf.cast(y_true, dtype='float32')
#     # N = y_true.shape[0]
#     # L = y_pred.shape[1]
#     # y_dot = y_pred*y_true
#     # y_d = y_dot-d
#     # y_square= y_d*y_d
#     # index_replace = y_dot>d
#     # idx_replace=tf.where(index_replace==True)
#     # y_loss = tf.tensor_scatter_nd_update(y_square, idx_replace, tf.zeros(idx_replace.shape[0]))
    
#     #Select the first column, this is enough to retrieve the False Negatives
#     y_NL = rewriteY(y_true)[:,0]
#     y_PL = rewriteY(y_pred)[:,0]
    
#     #by substracting the true values from the predicted we know that the False Negatives are in the
#     #positions where the -1 is located
#     y_sub = y_PL-y_NL
    
#     #replace the -1 with the costs associated with FN and replace the 0's with 1's
#     #1 is the cost of classifying FP and true predictions
#     costs = np.where(y_sub==-1, C, y_sub)
#     costs = np.where(costs==0, 1, costs)
    
#     #cast to a tensorflow tensor
#     costs = tf.cast(costs, dtype='float32')
#     y_true = tf.cast(y_true, dtype='float32')
    
#     #calculate loss function
#     N = y_true.shape[0]
#     L = y_pred.shape[1]
#     y_dot = y_pred*y_true
#     y_d = y_dot-d
#     y_square= y_d*y_d
#     index_replace = y_dot>d #for y_dot>d the value will be zero
#     idx_replace=tf.where(index_replace==True)
#     y_loss = tf.tensor_scatter_nd_update(y_square, idx_replace, tf.zeros(idx_replace.shape[0]))
    
#     #dot product of costs and the updated y_squared to calculate loss
#     y_dotted = tf.tensordot(tf.transpose(costs), y_loss,1)
   
#     return tf.divide(tf.keras.backend.sum(y_dotted),tf.cast(N*L, tf.float32))
# #%%
# def MetricsClass(y_true, y_pred):
#     """
#     Calculate diffrent metrics for the predicted values
#     consisting of: Accuracy, Precision, Recall, TP, TN, FP, FN
#     returning them in a dictionary

#     Parameters
#     ----------
#     y_true : numpy array
#         The true values of the data
#     y_pred : tensorflow tensor
#         the predicted values from the model

#     Returns
#     -------
#     dictt : dictionary
#         dictionary with the metrics

#     """
#     #both the y's need to be in values of 1's and 0's otherwise keras can't calculate
#     y_predict = rewriteY(y_pred)
#     y_test = rewriteY(y_true)
    
#     #initialize metrics
#     precision = tf.keras.metrics.Precision()
#     recall = tf.keras.metrics.Recall()
#     TP = tf.keras.metrics.TruePositives()
#     TN = tf.keras.metrics.TrueNegatives()
#     FP = tf.keras.metrics.FalsePositives()
#     FN = tf.keras.metrics.FalseNegatives()
#     Acc = tf.keras.metrics.Accuracy()
#     dictt = {}
#     metrics = [Acc, precision, recall, TP, TN, FP, FN]
#     keys = ['Accuraat', 'Precision', 'Recall', 'TP', 'TN', 'FP', 'FN']
    
#     #calculate metrics
#     for i in range(len(metrics)):
#         metrics[i].update_state(y_true = y_test[:,0], y_pred=y_predict[:,0])
#         dictt[keys[i]] = metrics[i].result().numpy()
#         metrics[i].reset_state()
#     dictt['Gmean'] = tf.keras.backend.sqrt(tf.divide((dictt['TP']*dictt['TN']),((dictt['TP']+dictt['FN'])*(dictt['TN']+dictt['FP']))))
#     dictt['Fmeasure'] = tf.divide((2*dictt['Recall']*dictt['Precision']),(dictt['Recall']+dictt['Precision']))
#     return dictt
# #%%
# def new_weight(weights, step, boundaries, select):
#     """
#     Randomly retrieve a new weight for a specific index based on the step or random choices and random numbers
#     within the boundaries of the weight that is selected

#     Parameters
#     ----------
#     weights : Tensorflow tensor or numpy array
#         the selected weights that need to be updated
#     step : float
#         the step for changing the weights
#     boundaries : array
#         the boundaries of the selected weight written in an array
#     select : integer
#         representing the selected weight, 0 for the hidden nodes weights, 1 for the bias and 2 for the beta coefficient

#     Returns
#     -------
#     index : integer
#         the index of the changed weight
#     origin : float
#         the original weight
#     weights : tensorflow tensor or numpy array
#         an array with the weights including the update

#     """
#     index=[]
#     origin = 0
#     #the hidden nodes weights is the only matrix thus needs two random indexes to be chosen
#     if select == 0:
#         ind1 = np.random.choice(weights.shape[0],1)[0] #row 
#         ind2 = np.random.choice(weights.shape[1],1)[0] #column
#         #combine the two to use as an index
#         index = tuple([ind1,ind2])  
#         #retrieve original
#         origin = weights[index]
#     else:
#         #both beta and bias are vectors
#         ind1 = np.random.choice(len(weights),1)
#         index = ind1
#         origin = weights[index][0]
    
#     #change the original by the step
#     new1 = origin + step
#     new2 = origin - step
    
#     #if both values are out of bounds select new random numbers within the boundaries
#     if new1 < boundaries[0] or new1 > boundaries[1] and new2 < boundaries[0] or new2 > boundaries[1]:    
#         new1 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
#         new2 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
#         #weights[index] = np.random.choice([new1[0], new2[0]])
    
#     #if only the new1 is out of bounds only select new random number for this one
#     elif new1 < boundaries[0] or new1 > boundaries[1]:
#         new1 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
#         #weights[index] = np.random.choice([new1[0], new2[0]])
    
#     #if only the new2 is out of bounds only select new random number for this one
#     elif new2 < boundaries[0] or new2 > boundaries[1]:
#         new2 = np.random.uniform(boundaries[0],boundaries[1],1)[0]
    
#     #randomly choose the new value 
#     weights[index] = np.random.choice([new1, new2])
    
    
#     return index, origin, weights
# #%%
# #@tf.function
# def train_step(epoch, b, batch, loss, weights, d, input_dim, lr, C, nodes):
#     """
#     The train step that is performed for every batch.
#     Here the model is optimized/trained, the weights are being updated.
#     Weights only get updated if they lower the loss

#     Parameters
#     ----------
#     b : numpy array
#         the beta transfer function coefficient
#     batch : tensorflow batch data
#         the current batch data used
#     loss : float
#         the most recent lowest loss value
#     weights : tensorflow tensor with arrays
#         the weights of the model.
#     d : integer
#         the steep seperating margin.
#     input_dim : integer
#         the input dimensions corresponding with the number of features/variables
#     lr : float
#         the learning rate used to calculate the step.
#     C : float
#         the costs associated with FN.
#     nodes : integer
#         the number of hidden nodes used in the model

#     Returns
#     -------
#     loss_min : float
#         the minimum loss value in this train step after optimizing
#     b : array
#         if selected the updated beta transfer function coefficient.
#     weights : tensorflow tensor 
#         if selected the updated weights
#     lr_new : float
#         the new learning rate used to calculate the step.

#     """
    
#     #retrieve the data from the batch
#     x, y = batch
    
#     #initialize parameters
#     loss_min = loss
#     lr = lr#*K.sqrt(loss_min)
#     boundariesW = [-1,1]
#     boundariesB = [-10,10]
#     boundariesBeta = [-1,1]
#     y_pred=[]
#     lr_new=lr
    
#     #randomly select beta, bias or weights
#     select = np.random.randint(3)
    
#     #if beta is selected calculate new weight
#     if select == 2:
#         ind,original, b = new_weight(b, lr, boundariesBeta, select)
#         #create new model with the new weight
#         model = CustomModel(b, input_dim, nodes)
#         #model needs to be used once before we can update weights
#         y_pred = model(x)
#         model.set_weights(weights)
#         #predict y to calculate loss
#         y_pred = model(x)
#         loss = compute_loss(d, y_pred, y, C)
#         #if loss is lower than minimum loss than select the weight is selected and minimum loss is updated
#         if loss<loss_min:
#             loss_min = loss
#             #new learning rate based on the new minimum loss
#             lr_new = lr*np.sqrt(loss_min)
#         else:
#             #else return the weight to normal
#             b[ind] = original
#     else:
#         #for 0 en 1 we have different boundaries but both used in weights array thus the distinction here
#         if select == 0:
#             ind, original, weights[select] = new_weight(weights[select], lr, boundariesW, select)
#         else:
#             ind, original, weights[select] = new_weight(weights[select], lr, boundariesB, select)
#         model = CustomModel(b, input_dim, nodes)
#         y_pred = model(x)
#         model.set_weights(weights)
#         y_pred = model(x)
#         loss = compute_loss(d, y_pred, y, C)
#         if loss<loss_min:
#             loss_min = loss
#             lr_new = lr*np.sqrt(loss_min)
#         else:
#             weights[select][ind]=original
        
#     # with tf.GradientTape() as tape:
#     #     x, y = batch
#     #     d = 16
#     #     y_pred = model(x)
#     #     loss = compute_loss(d, y_pred, y)

#     # gradients = tape.gradient(loss, model.trainable_variables)
#     # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     # update your metrics here how you want.
#     #diction = MetricsClass(y_true=y, y_pred=y_pred)
#     #acc_metric.update_state(y, y_pred)
    
#     #calculate metrics and print out loss and metrics
#     dictt = MetricsClass(y, y_pred)
#     # tf.print("Training loss (for one batch): ", loss_min)
#     # tf.print("Training metrics (for one batch): ", dictt.values())
#     if epoch==200 and dictt['Accuraat']<0.5:
#         model = CustomModel(b, input_dim, nodes)
#         y_pred = model(x)
#         weights = model.get_weights()
        
#     return loss_min, b, weights, lr_new, dictt
    

# def train(X_train, y_train, d, input_dim, b=np.random.uniform(-1,1, size=150), batch_size=300, C=2.5, lr=0.01, epochs=300, nodes=150):
#    """
#     Trains the model by running train step for every epoch and every batch
#     and returns the optimized model ready to be used

#     Parameters
#     ----------
#     dataset : Tensorflow batched dataset
#         Dataset that is split into batches from tensor slices
#     d : integer
#         the steep seperating margin.
#     input_dim : integer
#         the input dimensions corresponding to the number of variables
#     C : float, optional
#         Costs associated with FN. The default is 2.5.
#     lr : float, optional
#         Learning rate used to change the weights. The default is 0.01.
#     epochs : integer, optional
#         the number of time the model is run over the data. The default is 300.
#     nodes : integer, optional
#         the number of hidden nodes in the model. The default is 150.

#     Returns
#     -------
#     custom_model : Tensorflow keras model
#         The optimized model.

#     """
#    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
#    X_train.shape[0]).batch(batch_size)

#     #initialize the beta coefficient
#    b = b
#    #initialize parameters and model to get weights
#    lr = lr
#    model = CustomModel(b,input_dim, nodes)
#    x = list(dataset)[0][0]
#    y = list(dataset)[0][1]
#    y_p = model(x)
#    loss = compute_loss(d, y_p, y, C)
#    weights = model.get_weights()
   
#    #run the train step for every batch per epoch
#    for epoch in range(epochs):
#      tf.print(str(epoch)+'/'+str(epochs)+'-----------------------------')
     
#      for batch in dataset:
#          loss, b, weights, lr, metric = train_step(epoch, b, batch, loss, weights, d, input_dim, lr, C, nodes)
#      tf.print("Training loss (for one epoch): ", loss)
#      tf.print("Training metrics (for one epoch): ", metric.values())
#     #use the weights to get the optimal model and set the weights
#    custom_model = CustomModel(b,input_dim, nodes)
#    custom_model(x)
#    custom_model.set_weights(weights)
   
#    return custom_model, b
   

#      #train_acc = acc_metric.result()
#      #tf.print("Training acc over epoch: %.4f" % (float(train_acc),))

#      # Reset training metrics at the end of each epoch
#      #acc_metric.reset_states()

#%%
# #Load data
# location = r"C:\Users\rmuslem\Capgemini\Stage Rashed - General\breast-cancer-wisconsin.csv"
# df_BC = pd.read_csv(location, header=None)
# #input column names
# df_BC.columns = ['ID', 'Clump thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape',
#                  'Marginal Adhesion', 'Single Epithelial Cell Size',
#                  'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 
#                  'Mitoses', 'Class']
# #change index to ID
# df_BC.index = df_BC['ID']
# df_BC.drop(['ID'], axis=1, inplace=True)
# df_BC = df_BC[df_BC['Bare Nuclei']!='?']
# #split X and y
# X = df_BC.iloc[:,0:9]
# X_t = np.asarray(X).astype(np.int)
# #preprocess X
# scaler = sk.preprocessing.MinMaxScaler()
# Xscaled = scaler.fit_transform(X_t)
# #change y to a 2D array
# y = df_BC.iloc[:,-1]
# y = np.where(y==4, -1, y) #malignent, bad, minority class
# y = np.where(y==2, 1, y) #benign, good, majority class
# zeros = np.zeros(len(y))
# y_new = np.c_[y, zeros]
# for i in range(len(y)):
#     if y_new[i,0]==1:
#         y_new[i,1] = -1
#     else:
#         y_new[i,1]= 1

    
# #split in training and test data
# X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
#     Xscaled, y_new, test_size=0.333)
# #initialize model
# #need custom activation and loss function also custom initializer
# #need to initialize the network with hidden weights and bias between -1 and +1
# #Output weight is set to -1 or +1, transfer function coefficient to 1
# #Set number of nodes in hidden layer

# def initializeOutputWeights(shape, dtype=None):
#     #initialize the weights of the output node
#     randoms = np.random.randint(low=2, size=shape)
#     new = np.where(randoms==0, -1, randoms)
#     return K.variable(new, dtype=dtype)

# class customLoss(keras.losses.Loss):
#     def __init__(self, d=10, name = "CustomLoss"):
#         super().__init__(name=name)
#         self.d = d    
        
#     def call(self,y_true, y_pred):
#         N = len(y_true)
#         L = len(y_pred[0])
#         y_dot = y_pred*y_true
#         y_d = y_dot-self.d
#         y_square= y_d*y_d
#         index_replace = y_dot>self.d
#         idx_replace=tf.where(index_replace==True)
#         y_loss = tf.tensor_scatter_nd_update(y_square, idx_replace, tf.zeros(len(idx_replace)))
#         return tf.divide(K.sum(K.sum(y_loss, axis=1)),tf.cast(N*L, tf.float32))


# seed(1)
# tf.random.set_seed(2)
# zero = tf.keras.initializers.Zeros()
# model2 = Sequential()
# initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
# b = np.ones(20)
# model2.add(Dense(20, input_dim=9, name='hidden', kernel_initializer=initializer, 
#                  bias_initializer=initializer, activation = lambda x: K.tanh(b*x)))#activation='tanh'))
# model2.add(Dense(2, activation='linear', name='output', use_bias=False, trainable=False,kernel_initializer= lambda shape, 
#                  dtype: initializeOutputWeights(shape, dtype)))
# startingWeights = model2.get_weights()
# model2.compile(loss=customLoss(d=16), optimizer='adam', metrics=(['accuracy']))
# model2.fit(X_train,y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))
# # epochs = 150
# # for epoch in range(epochs):
#     # print("\nStart of epoch %d" % (epoch,))
    




# accuracy = model2.evaluate(X_test,y_test, verbose=1)
# y_predict = model2.predict(X_test)
# lossvalue = customLoss(d=16).call(y_test, y_predict)
# #print('Accuracy: %.2f' %(accuracy*100))
# #tf.enable_eager_execution()
# outputWeights2 = model2.get_weights()
# outputs2 = [K.function([model2.input], [layer.output])(Xscaled) for layer in model2.layers]
# print(outputs2)
# #%%
# outputs=[]
# for layer in model2.layers:
#     Keras_func = K.function([model2.input], [layer.output])
#     outputs.append(Keras_func([X_t,0]))
# print(outputs)
    
    
# #%%
# x = np.ones((5,5))
# b=[0.5,0.5,0.5,0.5,2]
# model = tf.keras.Sequential([tf.keras.layers.Dense(5, kernel_initializer=tf.initializers.Ones, activation=lambda x: K.tanh(b*x))])
# model.build(input_shape=x.shape)
# model(x)
# model.get_weights()
# #%%
# layer_name = 'my_layer'
# intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)
# #%%
# y_times = y_predict*y_test
# y_tf = tf.convert_to_tensor(y_times)
# y_squaredd = K.square(y_tf-16)
# indices_replace = y_tf>2
# idx_replace = tf.where(indices_replace== True)
# y_replaced = tf.tensor_scatter_nd_update(y_squaredd, idx_replace, tf.zeros(len(idx_replace), dtype=tf.float64))
# #%%
# '''
# Rewrite predictions to 1 and -1
# '''
# tf.enable_eager_execution()
# y_predict = outputs2[1][0]
# indexesMax = tf.reshape(tf.cast(tf.argmax(y_predict, axis=-1), "int32"),shape=(-1,1))
# indexesMin = tf.reshape(tf.cast(tf.argmin(y_predict, axis=-1), "int32"),shape=(-1,1))
# rangeind = tf.reshape(tf.range(len(y_predict)), shape=(-1,1))
# full_indexMax = tf.concat([rangeind, indexesMax], 1)
# full_indexMin = tf.concat([rangeind, indexesMin], 1)
# y_new2 = tf.tensor_scatter_nd_update(y_predict, full_indexMax, tf.ones(len(y_predict)))
# y_new2 = tf.tensor_scatter_nd_update(y_new2, full_indexMin, -tf.ones(len(y_predict)))

# import pandas as pd
# import numpy as np
# from math import exp
# from random import random
# a = np.array([[1,2,3],[2,1,4],[8,1,7]])
# a = pd.DataFrame(a)
# b = np.array([[0,1,0]]).reshape(3,1)
# b = pd.DataFrame(b)

# def initializeGVM(x,y,HiddenNodes, OutputNodes=1, rangeW = (-1,1), rangeO = (-1,1), rangeb = (-1,1), rangebeta = (-1,1), method='Standard'):
#     #need to initialize the network with hidden weights and bias between -1 and +1
#     #Output weight is set to -1 or +1, transfer function coefficient to 1
#     #Set number of nodes in hidden layer
#     inputNodes = len(x.columns)
#     if method == 'Standard':
#         weightH = np.random.uniform(low=rangeW[0], high=rangeW[1], size=(inputNodes,HiddenNodes))
#         biasH = np.random.uniform(low=rangeb[0], high=rangeb[1], size=HiddenNodes)
#         weightO = np.random.randint(low=2, size=(HiddenNodes, OutputNodes))
#         beta = np.ones(HiddenNodes)
#     elif method == 'Custom':
#         weightH = np.random.uniform(low=rangeW[0], high=rangeW[1], size=(inputNodes,HiddenNodes))
#         biasH = np.random.uniform(low=rangeb[0], high=rangeb[1], size=HiddenNodes)
#         weightO = np.random.uniform(low=rangeO[0], high=rangeO[1], size=(HiddenNodes, OutputNodes))
#         beta = np.random.uniform(low=rangebeta[0], high = rangebeta[1], size = HiddenNodes)
        
#     dictionary = {"Hidden Weights": weightH, "Output Weights" : weightO, 
#                   "Bias": biasH, "Transfer coefficient": beta}
#     return dictionary

# def tanh(x):
#     t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#     dt=1-t**2
#     return t,dt
# def hiddenlayer(inputs, weights, bias):
    
# network = initializeGVM(a, b, 4, 1)
    
    
