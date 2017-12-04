# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 16:17:36 2017

@author: zmx
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def data_load():
    print("data loaded")
    # use pandas to read csv file
    training_total=pd.read_csv('C:\\Users\zmx\\Desktop\\CS 513\\project\\train.csv')
    training_label=pd.DataFrame(training_total['label'])
    # training data is the data in train.csv except the first column
    training_data=pd.DataFrame(training_total.iloc[:,1:])

    testing_data=pd.read_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\test.csv')   
    # dataframe normalization
    testing_data[testing_data!=0]=1

    m, n = training_data.shape  # dataframe is too big, use for-loop to normalize the training data to avoid overflow
    for i in range(m):
        for j in range(n):
            if training_data.iloc[i, j] != 0:
                training_data.iloc[i, j] = 1
    return training_data,training_label,testing_data

# use sklearn library，classify the testing set
def rf_classify(training_data,training_label,testing_data,depth,cri,estimators):
    # set functions and parameters
    print("Classify started")
    rf_clf = RandomForestClassifier(criterion=str(cri),max_depth=depth, n_estimators=estimators)
    # train the training data, use values.ravel() to convert 2D array into 1D
    rf_clf.fit(training_data,training_label.values.ravel())
    # predict the testing data
    rf_result=rf_clf.predict(testing_data) 
    return rf_result

if __name__=='__main__':
    training_data,training_label,testing_data=data_load() # load the raw data
    print("Load success!")
    m, n = testing_data.shape # m = 42000, n = 784
    criteria = ["gini", "entropy"]
    depth_list = [32, 15, 5]
    estimators_list = [120, 300, 500, 800]
    # automate the test process
    for cri in criteria:
        for depth in depth_list:
            for estimators in estimators_list:
                result_labels=rf_classify(training_data,training_label,testing_data, depth, cri, estimators)
                result={}
                # np.arange distribute the values evenly, m=42000, which is the number of rows(digits)
                Image_ID=np.arange(m)+1
                result['Label']=result_labels
                result_frame=pd.DataFrame(result,index=Image_ID)
            
                # write the result dataframe to a csv file
                result_frame.to_csv('result_rf_'+ cri + str(depth) +'_'+ str(estimators)+'.csv')
                
