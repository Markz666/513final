from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import csv
import pandas as pd
import sys
stdout = sys.stdout  
stdin = sys.stdin  
stderr = sys.stderr
reload(sys)
sys.setdefaultencoding('utf8')
sys.stdout = stdout  
sys.stdin = stdin  
sys.stderr = stderr 



def opencsv():#use pandas to read csv file    
    data_train = pd.read_csv('train.csv')      
    data_test=pd.read_csv('test.csv')
    train_data = data_train.values[0:,1:]#Read all train data
    train_label = data_train.values[0:,0]#Read label
    test_data=data_test.values[0:,0:]#Read test data
    return train_data,train_label,test_data


def nomalizing(array):#dataframe normalization
             x,y=np.shape(array)
             for i in range(x):
                  for j in range(y):
                       if array[i,j]!=0:#Test each column of each row, if the gray scale value is not zero, then set it as 1.
                          array[i,j]=1
             return array
def knnClassify(trainData,trainLabel,testData):
              knnClf=KNeighborsClassifier(10)#In default k = 5
              knnClf.fit(trainData,np.ravel(trainLabel))#Bulid a model
              testLabel=knnClf.predict(testData)#Predict the class labels for the provided data
              np.savetxt('sklearn_knn_Result.csv', testLabel, delimiter=',')

if __name__ == "__main__":
    train_data,train_label,test_data=opencsv()
    trainData=nomalizing(train_data)
    testData=nomalizing(test_data)
    knnClassify(trainData,train_label,testData)

