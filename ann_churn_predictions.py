# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:14:33 2023

@author: ATISHKUMAR
"""

import pandas as pd
import numpy as np
import tensorflow as tf
tf.__version__

dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\sept_DL\7th\ANN_ 1st\Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#encoding catagorical data
#label encoding for the "Gender" column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

#one hot encoding for " geography "column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#apply feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#splittting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)


#part 2-

#building ANN

#inittialize ANN
ann=tf.keras.models.Sequential()

#adding the input layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#adding the seacond hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


#adding the ouput layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


#part-3
#training ANN

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#training the ann on training set

ann.fit(X_train,y_train, batch_size=32, epochs=200)

#part 4
#making predictions and evaluating the model

#predict the test reasults
y_pred=ann.predict(X_test)
y_pred=(y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac