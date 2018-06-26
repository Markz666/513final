import pandas as pd
import numpy as np
import csv as csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# Load the data 
train_raw=pd.read_csv("C:\\Users\\zhang\\Desktop\\513data\\train.csv",header=0)
test_raw=pd.read_csv("C:\\Users\\zhang\\Desktop\\513data\\test.csv",header=0)
train = train_raw.values
test = test_raw.values

print ('Start PCA, reduce the dimension to 50')
train_x=train[0::,1::]
train_label=train[::,0]
pca = RandomizedPCA(n_components=50, whiten=True).fit(train_x)
train_x_pca=pca.transform(train_x)
test_x_pca=pca.transform(test)

a_train, b_train, a_label, b_label = train_test_split(train_x_pca, train_label, test_size=0.33, random_state=23323)

print (a_train.shape)
print (a_label.shape)

print ('Start training')
rbf_svc = svm.SVC(C=0.001, class_weight=None,gamma='auto',kernel = 'rbf')
# C means Penalty parameter C of the error term. 
# 
# kernel = 'rbf' radial basis function 
# gamma = 'auto' means gamma is auto scale

rbf_svc.fit(a_train,a_label)
# Fit the model 
print ('Start predicting')
#Prediction part
b_predict=rbf_svc.predict(b_train)


score=accuracy_score(b_label,b_predict)
print ("The accruacy socre is : ", score)

print ('Start writing!')


out=rbf_svc.predict(test_x_pca)

import csv
# Write the result to the CSV file
csvfile = open('C=0.001, class_weight=None,gamma=auto,kernel = rbf.csv','w')
csvfile.write('ImageId,Label\n')
n = len(out)
for i in range(n):
    csvfile.write('%d,%d\n' % (i + 1, out[i]))
csvfile.close()

print ('All is done')
