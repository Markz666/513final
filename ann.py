
#import Pandas
import pandas as pd
import numpy as np
import tensorflow as tf


#Read csv dataset for training and validation
train_data=pd.read_csv('/users/haodong/desktop/train.csv')

#Read dataset for Kaggle submission
test_data=pd.read_csv('/users/haodong/desktop/test.csv')

#X=independant variables=the image data + basic feature scaling
X=train_data.iloc[:,1:].values / 255.0
X_submit=test_data.iloc[:,0:].values / 255.0

#y=dependant variable= the written digit label 0-9
y=train_data.iloc[:,:1].values

# Normalization
# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
Onehotencoder = OneHotEncoder(categorical_features = [0])
y = Onehotencoder.fit_transform(y).toarray()

#Grab the automatic data splitter from sklearn
from sklearn.model_selection import train_test_split

#Create the Training/Testing split for my cross validation (No cross validation at present)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.001,random_state=0)


#Import keras to create the sequential network structure
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Creat model
#Initialize the network
ANN_model = Sequential()

#Add the first hidden layer, and specifying #inputs mean(10,784)=397)
ANN_model.add(Dense(units=382,kernel_initializer='uniform',activation='relu',input_dim=784))
ANN_model.add(Dropout(0.3))

ANN_model.add(Dense(units=191,kernel_initializer='uniform',activation='relu',input_dim=784))
ANN_model.add(Dropout(0.3))
#Add the output layer, an analog digit value
ANN_model.add(Dense(units=10,kernel_initializer='uniform',activation='sigmoid'))
# Compiling the ANN
ANN_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
ANN_model.fit(X_train, y_train, batch_size = 64, epochs = 20, verbose=1) #500
#Predicting the Kaggle submission
y_predict_submit=ANN_model.predict(X_submit)

#Create the Kaggle submission
y_pred_submit_int=[['ImageId','Label']]
for i in range(len(X_submit)):
    summedVal=int(round((y_predict_submit[i][0]*0)+(y_predict_submit[i][1]*1)+(y_predict_submit[i][2]*2)+(y_predict_submit[i][3]*3)+(y_predict_submit[i][4]*4)+(y_predict_submit[i][5]*5)+(y_predict_submit[i][6]*6)+(y_predict_submit[i][7]*7)+(y_predict_submit[i][8]*8)+(y_predict_submit[i][9]*9),0))
    if(summedVal>9):
        summedVal=9
    pair=[i+1,summedVal]
    y_pred_submit_int.append(pair)

#Write submission to storage
raw_data = y_pred_submit_int
df = pd.DataFrame(raw_data)
df.to_csv(path_or_buf = 'submissionbs64ep20ver1T.csv', index=None, header=False)
