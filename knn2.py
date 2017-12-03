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



def opencsv():#使用pandas打开      
    data = pd.read_csv('train.csv')      
    data1=pd.read_csv('test.csv')
    train_data = data.values[0:,1:]#读入全部训练数据
    train_label = data.values[0:,0]#读入标签
    test_data=data1.values[0:,0:]#测试全部测试个数据
    return train_data,train_label,test_data


def nomalizing(array):#归一化数据
             m,n=np.shape(array)
             for i in range(m):
                  for j in range(n):
                       if array[i,j]!=0:#每一行的每一列进行检测，如果非零则置为1
                          array[i,j]=1
             return array
def knnClassify(trainData,trainLabel,testData):
              knnClf=KNeighborsClassifier(10)#k=5   KNN中邻值为5，
              knnClf.fit(trainData,np.ravel(trainLabel))#ravel->降维数组.Fit the model using X as training data and y as target values
              testLabel=knnClf.predict(testData)#Predict the class labels for the provided data
              np.savetxt('sklearn_knn10_Result.csv', testLabel, delimiter=',')

if __name__ == "__main__":
    train_data,train_label,test_data=opencsv()
    trainData=nomalizing(train_data)
    testData=nomalizing(test_data)
    knnClassify(trainData,train_label,testData)
    
#    print(shape(train_data))
#    print(shape(train_label))
#    print(shape(test_data))
