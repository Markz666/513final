# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 16:17:36 2017

@author: zmx
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def data_load():
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

#def plotNum(ind, data):
#    plt.imshow(np.reshape(np.array(data.iloc[ind,1:]), (28, 28)), cmap="gray")
#    plt.figure()
#    for ii in range(1,17):
#        plt.subplot(4,4,ii)
#        plotNum(ii)
#    plt.show()
    
# use sklearn library，classify the testing set
def rf_classify(training_data,training_label,testing_data):
    # set functions and parameters
    rf_clf = RandomForestClassifier()
    # train the training data, use values.ravel() to convert 2D array into 1D
    rf_clf.fit(training_data,training_label.values.ravel())
    # predict the testing data
    rf_result=rf_clf.predict(testing_data) 
    return rf_result

if __name__=='__main__':
    start = time.clock() # get start time
    training_data,training_label,testing_data=data_load() # load the raw data

    m, n = testing_data.shape
    result_labels=rf_classify(training_data,training_label,testing_data)

    # convert the result into dataframe
    result={}
    # np.arange distribute the values evenly, m=42000, which is the number of rows(digits)
    Image_ID=np.arange(m)+1
    result['Label']=result_labels
    result_frame=pd.DataFrame(result,index=Image_ID)

    # write the result dataframe to a csv file
    result_frame.to_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\result_rf.csv')
    end = time.clock() # get end time
    print('Total time: ', (end - start) / 60) # 75 minutes
