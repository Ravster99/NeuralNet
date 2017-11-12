
# coding: utf-8

# In[2]:



# coding: utf-8

# In[1]:


#Your code is intended for binary classification problems.
#All of the attributes are numeric.
#The neural network has connections between input and the hidden layer,
#and between the hidden and output layer and one bias unit and one output node.
#The number of units in the hidden layer should be equal to the number of input units.
#For training the neural network, use n-fold stratified cross validation.
#Use sigmoid activation function and train using stochastic gradient descent.
#Randomly set initial weights for all units including bias in the range (-1,1).
#Use a threshold value of 0.5. If the sigmoidal output is less than 0.5, 
#take the prediction to be the class listed first in the ARFF file in the class attributes section;
#else take the prediction to be the class listed second in the ARFF file.


# In[2]:


#%reset
#Importing important libraries.
#Import arff for Windows.
import arff
import pandas as pd
import numpy as np
from numpy import random
#import scipy.io as sio
#import sys
from numpy import linalg as LA
#from scipy.io import arff
import math


# In[3]:


########Loading of data in Linux.###################
########Training data set.############################
#raw_data = arff.loadarff(open(sys.argv[1]))
#feature_list = [] 
#for x in raw_data['attributes']:
#    feature_list.append(x[0])
#training_data = pd.DataFrame(raw_data[0])
#training_data = training_data.apply(pd.to_numeric, errors='ignore')
########Loading of data in Windows.###################
########Training data set.############################
raw_data = arff.load(open('sonar.arff')) # Loading the training datasets
#Creating a list of the features
feature_list = [] 
for x in raw_data['attributes']:
    feature_list.append(x[0])
training_data = pd.DataFrame(np.array(raw_data['data']), columns = feature_list) # Saving as a Pandas Dataframe
training_data = training_data.apply(pd.to_numeric, errors='ignore') # Converting the releavent columns to numeric


# In[4]:


global weight_hidden, weight_output, bias_hidden, accuracy


# In[5]:


#Compute and return the sigmoid activation value for a
#Given input value
def sigmoid_activation(x):
    hidden_output_calculation = 1.0 / (1 + np.exp(-x))
    return hidden_output_calculation
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


# In[6]:


def initialization(number_of_inputlayer_neurons):
    #Initialization of weights for hidden layer and output layer.
    number_of_hiddenlayer_neurons = number_of_inputlayer_neurons
    weight_hidden = 2*np.random.uniform(size=(number_of_inputlayer_neurons, number_of_hiddenlayer_neurons))-1
    bias_hidden = 2*np.random.uniform(size=(1, number_of_hiddenlayer_neurons))-1
    weight_output = 2*np.random.uniform(size=(number_of_hiddenlayer_neurons, 1))-1
    return weight_output, weight_hidden, bias_hidden


# In[7]:


def NeuralNetwork (ita, epoch, X, Y, weight_output, weight_hidden, bias_hidden):
    for j in range(0, epoch):
        #Forward propagation.
        hiddenlayer_input = np.dot(X, weight_hidden) + bias_hidden
        hiddenlayer_output = sigmoid_activation(hiddenlayer_input)
        outputlayer_input = np.dot(hiddenlayer_output, weight_output)
        outputlayer_output = sigmoid_activation(outputlayer_input)        
        error = np.zeros(shape=(len(Y), 1))
        for i in range(0, len(outputlayer_output)):
                if (Y[i] == 'Rock'):
                    error [i] = -outputlayer_output[i]
                if (Y[i] == 'Mine'):
                    error [i] = 1- outputlayer_output[i]
        #Backward propagation.
        slope_output_layer = derivatives_sigmoid(outputlayer_output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_output)
        delta_output = error * slope_output_layer
        Error_at_hidden_layer = delta_output.dot(weight_output.T)
        delta_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        weight_output = weight_output + hiddenlayer_output.T.dot(delta_output) * ita
        weight_hidden = weight_hidden + X.T.dot(delta_hiddenlayer) * ita
    return weight_output, weight_hidden, bias_hidden, outputlayer_output


# In[8]:


def stratification (data, n):
    #n-fold cross validation.
    #Create a empty dataframe to contain the strata of the data.
    stratified_data = pd.DataFrame(columns=feature_list)
    stratified_data['Fold_of_instance'] = np.nan
    #Find number of data-points to go into each strata.
    data_count = data['Class'].value_counts()
    rock_data_count = int(data_count['Rock']/n)
    mine_data_count = int(data_count['Mine']/n)
    #Create two partition of the dataset (Rock and Mine).
    rock_data = data[data['Class'] == 'Rock'].reset_index(drop = 'True')
    mine_data = data[data['Class'] == 'Mine'].reset_index(drop = 'True')
    #Appending the split data into a new dataframe with the strata number.
    for i in range(0, n-1):
        #Appending for rock data.
        index_for_rock_data = np.random.choice(rock_data.index.values, rock_data_count, replace = False)
        rock_dataframe  = rock_data.iloc[index_for_rock_data]
        rock_data = rock_data[~rock_data.isin(rock_dataframe)].dropna()
        rock_data = rock_data.reset_index(drop = True)
        rock_dataframe['Fold_of_instance'] = i + 1
        stratified_data = stratified_data.append(rock_dataframe, ignore_index=True)
        #Appending for mine data.
        index_for_mine_data = random.choice(mine_data.index.values, mine_data_count, replace = False)
        mine_dataframe  = mine_data.iloc[index_for_mine_data]
        mine_data = mine_data[~mine_data.isin(mine_dataframe)].dropna()
        mine_data = mine_data.reset_index(drop = True)
        mine_dataframe['Fold_of_instance'] = i + 1
        stratified_data = stratified_data.append(mine_dataframe, ignore_index=True)
    #We might have ignored certain values due to roundin off. To consider all the remaining values.
    #Appending for rock data.
    index_for_rock_data = np.random.choice(rock_data.index.values, len(rock_data), replace = False)
    rock_dataframe  = rock_data.iloc[index_for_rock_data]
    rock_data = rock_data[~rock_data.isin(rock_dataframe)].dropna()
    rock_data = rock_data.reset_index(drop = True)
    rock_dataframe['Fold_of_instance'] = n
    stratified_data = stratified_data.append(rock_dataframe, ignore_index=True)
    #Appending for mine data.
    index_for_mine_data = random.choice(mine_data.index.values, len(mine_data), replace = False)
    mine_dataframe  = mine_data.iloc[index_for_mine_data]
    mine_data = mine_data[~mine_data.isin(mine_dataframe)].dropna()
    mine_data = mine_data.reset_index(drop = True)
    mine_dataframe['Fold_of_instance'] = n
    stratified_data = stratified_data.append(mine_dataframe, ignore_index=True)
    #Returning the n - fold dataset.
    return stratified_data


# In[9]:


def prediction (weight_hidden, bias_hidden, weight_output, X, Y):
    #Function to predict based on trained weights.
    outputlayer_output = np.zeros(X.shape[0])
    hiddenlayer_input = np.dot(X, weight_hidden) + bias_hidden
    hiddenlayer_output = sigmoid_activation(hiddenlayer_input)
    outputlayer_input = np.dot(hiddenlayer_output, weight_output)
    outputlayer_output = sigmoid_activation(outputlayer_input)
    predicted_value = []
    for i in range (0, len(X)):
        if (outputlayer_output[i] <= 0.50):
            predicted_value.append('Rock')
        else:
            predicted_value.append('Mine')
    return predicted_value, outputlayer_output    


# In[11]:


# Main output.
# Creating the stratified dataset
#n = int(sys.argv[2]) # Number of strata to be created
n = 10
stratified_training_data = stratification (training_data, n)
number_of_inputlayer_neurons = stratified_training_data.shape[1] - 2
# Adding the confidence of prediction and class of prediction.
print_dataset = stratified_training_data.copy()
print_dataset['Confidence_of_prediction'] = np.nan
print_dataset['Predicted Class'] = np.nan
(weight_output, weight_hidden, bias_hidden) = initialization(number_of_inputlayer_neurons)
# Architecture.
#ita = float(sys.argv[3]) #Learning rate of the algorithm.
#epoch = int(sys.argv[4]) #Number of epochs considered.
ita = 0.01
epoch = 50
for iteration in range(n):
    #Training over each stratified dataset.
    substratified_training_data = stratified_training_data[stratified_training_data['Fold_of_instance'] != (iteration + 1)]
    substratified_training_data = substratified_training_data.reset_index(drop = True)
    batch_size = substratified_training_data['Class'].value_counts()
    X_training = substratified_training_data.iloc[:, 0:60]
    Y_training = substratified_training_data['Class']    
    (weight_output, weight_hidden, bias_hidden, outputlayer_output) = NeuralNetwork (ita, epoch, X_training, Y_training, weight_output, weight_hidden, bias_hidden)
#Working on testing dataset.
substratified_testing_data = stratified_training_data[stratified_training_data['Fold_of_instance'] == (iteration + 1)]
substratified_testing_data = substratified_testing_data.reset_index(drop = True)
X_testing = substratified_testing_data .iloc[:, 0:60]
Y_testing = substratified_testing_data['Class']
#Prediction.
(predicted_value, outputlayer_output) = prediction(weight_hidden, bias_hidden, weight_output, X_testing, Y_testing)
predicted_value_column = pd.DataFrame({'Prediction': predicted_value})
predicted_value_column  = predicted_value_column['Prediction']
#Determining accuracy.
sum = 0.0
for i in range (0, len(Y_testing)):
    if (Y_testing[i] == predicted_value_column[i]):
        sum+= 1        
accuracy = sum/len(Y_testing)
print('Accuracy is: ', accuracy)
predicted_value_column = predicted_value_column.values
#To disable the warning.
pd.options.mode.chained_assignment = None  # default='warn'
print_dataset['Predicted Class'][print_dataset['Fold_of_instance'] == (iteration + 1)] = predicted_value_column
print_dataset['Confidence_of_prediction'][print_dataset['Fold_of_instance'] == (iteration + 1)] = outputlayer_output
#Printing the dataset.
print(print_dataset)
#Printing of ROC curve.
#Plot ROC curve for the neural network constructed with the following parameters: 
#(With learning rate = 0.1, number of epochs = 50, number of folds = 10)
print_dataset = print_dataset.sort_values('Confidence_of_prediction', ascending = False)
num_neg = len(print_dataset[print_dataset['Class'] == 'Rock'])
num_pos = len(print_dataset[print_dataset['Class'] == 'Mine'])
TP, FP, last_TP = 0, 0, 0
ROC_curve_data = pd.DataFrame(columns = ['TPR', 'FPR'])
for iteration in range(len(print_dataset)):
    if ((iteration > 0) & (print_dataset.iloc[iteration]['Confidence_of_prediction'] != print_dataset.iloc[iteration-1]['Confidence_of_prediction']) & (TP > last_TP) & (print_dataset.iloc[iteration]['Class'] == 'Rock')):
        TPR = TP / num_pos       
        FPR = FP / num_neg
        ROC_data = {'TPR': [TPR], 'FPR': [FPR]}
        ROC_data = pd.DataFrame(ROC_data, columns = ['TPR', 'FPR'])
        ROC_curve_data = ROC_curve_data.append(ROC_data, ignore_index = True)
        last_TP = TP
    if (print_dataset.iloc[iteration]['Class'] == 'Mine'):
        TP = TP + 1
    else:
        FP = FP + 1  
TPR = TP / num_pos
FPR = FP / num_neg
ROC_data = {'TPR': [TPR], 'FPR': [FPR]}
ROC_data = pd.DataFrame(ROC_data, columns = ['TPR', 'FPR'])
ROC_curve_data = ROC_curve_data.append(ROC_data, ignore_index = True)


