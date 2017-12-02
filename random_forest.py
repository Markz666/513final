# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:17:36 2017

@author: zmx
"""
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

def data_load():

    # use pandas to read csv file
    train_ttl=pd.read_csv('C:\\Users\zmx\\Desktop\\CS 513\\project\\train.csv')
    train_label=pd.DataFrame(train_ttl['label'])
    train_data=pd.DataFrame(train_ttl.ix[:,1:])
    test_data=pd.read_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\test.csv')

    # dataframe normalization
    test_data[test_data!=0]=1

    # train_data[train_data!=0]=1
    m, n = train_data.shape  # dataframe is too big, use for loop
    for i in range(m):
        for j in range(n):
            if train_data.ix[i, j] != 0:
                train_data.ix[i, j] = 1

    return train_data,train_label,test_data

#use Python Sklearn，classify the test set
def rf_classify(traindata,trainlabel,testdata):

    rf_clf = RandomForestClassifier() # set functions and parameters
    rf_clf.fit(traindata,trainlabel.values.ravel()) #train the traindata
    rf_result=rf_clf.predict(testdata) #predict the testdata

    return rf_result

if __name__=='__main__':
    start = time.clock()
    traindata,trainlabel,testdata=data_load()#load the raw data

    m,n=testdata.shape
    result_labels=rf_classify(traindata,trainlabel,testdata)

    # convert result into dataframe
    result={}
    ImageId=np.arange(m)+1
    result['Label']=result_labels
    result_frame=pd.DataFrame(result,index=ImageId)

    # generate the result
    result_frame.to_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\result_rf.csv')
    end = time.clock()
    print('Total time: ', (end - start)/60)#1.5 hours
