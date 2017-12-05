# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:55:09 2017

@author: zhang
"""


import pandas as Panda
import time
from sklearn.linear_model import LogisticRegression

# load the data from the url 
    
train_data = Panda.read_csv("C:\\Users\\zhang\\Desktop\\513data\\train.csv")
test_data  = Panda.read_csv("C:\\Users\\zhang\\Desktop\\513data\\test.csv")
# Normalization the data 
train_data.loc[:,'pixel0':] =  train_data.loc[:,'pixel0':] * 1.0/255.0
test_data.loc[:,'pixel0':] =  test_data.loc[:,'pixel0':] * 1.0/255.0
data =  train_data.loc[:,'pixel0':] * 1.0/255.0
# use the sklearn LogisticRegression package to train the model 
main = LogisticRegression(penalty = 'l2' , solver = 'sag',C=15)
# Use different parameter to train the predict model 
#solver: Choose the gradient decent optimization method  
#      sag:Stochastic Gradient Descent
#      liblinear: Coordinate Descent
#C: Inverseof regularization strength; must be a positive float.
#Like in support vectormachines, smaller values specify stronger regularization.
#
# The penalty parameter is on the left 
main.fit(train_data.loc[:,'pixel0':], train_data.loc[:,'label'])
# Use the model and predict the data 
predictions = main.predict(test_data)

# Show the predict result 
print(predictions)
# Establish acsv file and write the data
import csv
csvfile = open('submit_regre(penalty = l2 , solver = sag,C=15).csv','w')
csvfile.write('ImageId,Label\n')
n = len(predictions)
for i in range(n):
    csvfile.write('%d,%d \n' % (i + 1, predictions[i])) 
    # All the result need to fit the Kaggle submission format "ImageId , Label " the data need to be Integer type
csvfile.close()
 