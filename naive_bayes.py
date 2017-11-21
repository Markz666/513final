# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:03:46 2017

@author: zmx
"""
import pandas as pd
import numpy as np
import time
from sklearn import naive_bayes

def data_load():

    # use pandas to read csv file
    train_ttl=pd.read_csv('C:\Users\zmx\Desktop\CS 513\project\\train.csv')
    train_label=pd.DataFrame(train_ttl['label'])
    train_data=pd.DataFrame(train_ttl.ix[:,1:])
    test_data=pd.read_csv('C:\Users\zmx\Desktop\CS 513\project\\test.csv')

    # dataframe normalization
    test_data[test_data!=0]=1

    # train_data[train_data!=0]=1
    m,n=train_data.shape #data frame too big, use for loop
    for i in range(m):
        for j in range(n):
            if train_data.ix[i,j]!=0:
                train_data.ix[i,j]=1

    return train_data,train_label,test_data

# use sklearn to classify
def bayes_classify(traindata,trainlabel,testdata):

    bayes_clf = naive_bayes.MultinomialNB() #set function and parameters
    bayes_clf.fit(traindata,trainlabel.values.ravel())#train the data
    bayes_result=bayes_clf.predict(testdata) #predict the test data

    return bayes_result

if __name__=='__main__':
    start = time.clock()
    traindata,trainlabel,testdata=data_load() #load raw data

    m,n=testdata.shape
    result_labels=bayes_classify(traindata,trainlabel,testdata)

    #put result as dataframe
    result={}
    ImageId=np.arange(m)+1
    result['Label']=result_labels
    result_frame=pd.DataFrame(result,index=ImageId)

    # write result to csv
    result_frame.to_csv('C:\Users\zmx\Desktop\CS 513\project\\result_bayes.csv')
    end = time.clock()
    print('Total time:', (end - start)/3600.0)
