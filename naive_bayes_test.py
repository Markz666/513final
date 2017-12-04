# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:03:46 2017

@author: zmx
"""
#import the libraries
import numpy as np
import pandas as pd
from sklearn import naive_bayes
from sklearn.decomposition import RandomizedPCA

def data_load():
    print("data loaded")
	 # use pandas to read csv file
    training_total=pd.read_csv('C:\\Users\zmx\\Desktop\\CS 513\\project\\train.csv')
    training_label=pd.DataFrame(training_total['label'])
    # training data is the data in train.csv except the first column
    training_data=pd.DataFrame(training_total.iloc[:,1:])
    pca = RandomizedPCA(n_components=50, whiten=True).fit(training_data)
    training_data=pca.transform(training_data)

    testing_data=pd.read_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\test.csv')
    testing_data=pca.transform(testing_data)

    # dataframe normalization
    testing_data[testing_data!=0]=1
    m,n=training_data.shape  # dataframe is too big, use for-loop to normalize the training data to avoid overflow
    for i in range(m):
        for j in range(n):
            if training_data.iloc[i, j] != 0:
                training_data.iloc[i, j] = 1
    return training_data,training_label,testing_data

# use sklearn to classify
def bayes_classify(training_data,training_label,testing_data, alpha, flag):
    print("classify started")
    #set functions and parameters
    bayes_clf = naive_bayes.MultinomialNB(alpha = float(alpha_value), fit_prior = bool(flag))
    # train the training data, use values.ravel() to convert 2D array into 1D
    bayes_clf.fit(training_data,training_label.values.ravel())
    #predict the testing data
    bayes_result=bayes_clf.predict(testing_data) 
    return bayes_result

if __name__=='__main__':

    training_data,training_label,testing_data=data_load() #load the raw data

    m,n=testing_data.shape # m = 42000, n = 784
    alpha_list = ["0.0", "1.0", "5.0", "10.0", "20.0", "40.0"]
    flag_list = ["True", "False"]
    for alpha_value in alpha_list:
        for flag in flag_list:
            result_labels = bayes_classify(training_data,training_label,testing_data, alpha_value, flag)
            result={}
            # np.arange distribute the values evenly, m=42000, which is the number of rows(digits)
            Image_ID=np.arange(m)+1
            result['Label']=result_labels
            result_frame=pd.DataFrame(result,index=Image_ID)

            # write the result dataframe to a csv file
            result_frame.to_csv('C:\\Users\\zmx\\Desktop\\CS 513\\project\\result_bayes_' +str(alpha_value)+'_'+str(flag) +'.csv')

