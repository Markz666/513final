# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:03:46 2017

@author: zmx
"""
#import the libraries
import numpy as np
import pandas as pd
import time
from sklearn import naive_bayes

def data_load():
	 # use pandas to read csv file
    training_total=pd.read_csv('C:\\Users\zmx\\Desktop\\CS 513\\project\\train.csv')
    training_label=pd.DataFrame(training_total['label'])
    training_data=pd.DataFrame(training_total.iloc[:,1:])
    testing_data=pd.read_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\test.csv')

    # dataframe normalization
    testing_data[testing_data!=0]=1
    m,n=training_data.shape  # dataframe is too big, use for-loop to normalize the training data to avoid overflow
    for i in range(m):
        for j in range(n):
            if training_data.iloc[i, j] != 0:
                training_data.iloc[i, j] = 1
    return training_data,training_label,testing_data

# use sklearn to classify
def bayes_classify(training_data,training_label,testing_data):
    bayes_clf = naive_bayes.MultinomialNB() #set functions and parameters
    bayes_clf.fit(training_data,training_label.values.ravel()) # train the training data
    bayes_result=bayes_clf.predict(testing_data) #predict the testing data
    return bayes_result

if __name__=='__main__':
    start = time.clock() # get start time
    training_data,training_label,testing_data=data_load() #load the raw data

    m,n=testing_data.shape
    result_labels=bayes_classify(training_data,training_label,testing_data)

    # convert the result into dataframe
    result={}
    Image_ID=np.arange(m)+1
    result['Label']=result_labels
    result_frame=pd.DataFrame(result,index=Image_ID)

    # write the result dataframe to a csv file
    result_frame.to_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\result_bayes.csv')
    end = time.clock() # get end time
    print('Total time:', (end - start)/3600.0) # 1.2 hours
