# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 16:17:36 2017

@author: zmx
"""
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

def data_load():

    # use pandas to read csv file
    training_total=pd.read_csv('C:\\Users\zmx\\Desktop\\CS 513\\project\\train.csv')
    training_label=pd.DataFrame(training_total['label'])
    training_data=pd.DataFrame(training_total.ix[:,1:])
    testing_data=pd.read_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\test.csv')

    # dataframe normalization
    testing_data[testing_data!=0]=1

    # train_data[train_data!=0]=1
    m, n = training_data.shape  # dataframe is too big, use for loop
    for i in range(m):
        for j in range(n):
            if training_data.ix[i, j] != 0:
                training_data.ix[i, j] = 1

    return training_data,training_label,testing_data

#use Sklearn library，classify the test set
def rf_classify(training_data,training_label,testing_data):

    rf_clf = RandomForestClassifier() # set functions and parameters
    rf_clf.fit(training_data,training_label.values.ravel()) #train the training data
    rf_result=rf_clf.predict(testing_data) #predict the testing data

    return rf_result

if __name__=='__main__':
    start = time.clock()
    training_data,training_label,testing_data=data_load()#load the raw data

    m,n=testing_data.shape
    result_labels=rf_classify(training_data,training_label,testing_data)

    # convert result into dataframe
    result={}
    Image_ID=np.arange(m)+1
    result['Label']=result_labels
    result_frame=pd.DataFrame(result,index=Image_ID)

    # write the result to a csv file
    result_frame.to_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\result_rf.csv')
    end = time.clock()
    print('Total time: ', (end - start) / 60) # 75 minutes
